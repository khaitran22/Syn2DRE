from utils import evaluate, report
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
from transformers.optimization import get_linear_schedule_with_warmup
from evaluation import *
from ramps import *

"""
    Work for both multi or mono GPU
"""

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

class Trainer:
    def __init__(self, rank, args, main_net1, main_net2, teacher_main_net1, teacher_main_net2, 
                 train_loader, distant_loader, dev_loader, test_loader):
        self.rank = rank
        self.args = args
        self.main_net1 = main_net1
        self.main_net2 = main_net2
        self.teacher_main_net1 = teacher_main_net1
        self.teacher_main_net2 = teacher_main_net2
        self.distant_loader = distant_loader
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.num_classes = self.args.num_class
        self.uncert_th_1 = torch.zeros(self.num_classes)
        self.uncert_th_2 = torch.zeros(self.num_classes)
        self._setup_training()

    def _setup_training(self):
        args = self.args
        
        ### main1 opt
        main1_new_layer = ["extractor", "bilinear"]
        main1_optimizer_grouped_parameters = [
            {"params": [p for n, p in self.main_net1.named_parameters() if not any(nd in n for nd in main1_new_layer)], },
            {"params": [p for n, p in self.main_net1.named_parameters() if any(nd in n for nd in main1_new_layer)], "lr": args.main1_classifier_lr},
        ]
        self.main1_opt = torch.optim.AdamW(main1_optimizer_grouped_parameters, lr=args.main1_lr, eps=args.adam_epsilon)
        
        ### main2 opt
        main2_new_layer = ["extractor", "bilinear", "classifier",  "projection"]
        main2_optimizer_grouped_parameters = [
            {"params": [p for n, p in self.main_net2.named_parameters() if not any(nd in n for nd in main2_new_layer)], },
            {"params": [p for n, p in self.main_net2.named_parameters() if any(nd in n for nd in main2_new_layer)], "lr": args.main2_classifier_lr},
        ]
        self.main2_opt = torch.optim.AdamW(main2_optimizer_grouped_parameters, lr=args.main2_lr, eps=args.adam_epsilon)

        ### general
        total_steps = int(len(self.train_loader) * self.args.num_train_epochs // self.args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        self.main1_schdlr = get_linear_schedule_with_warmup(self.main1_opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        self.main2_schdlr = get_linear_schedule_with_warmup(self.main2_opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    def _prepare_input(self, batch, model_name="2"):
        inputs = {'input_ids': batch[0].to(self.rank),
                'attention_mask': batch[1].to(self.rank),
                'labels': batch[2],
                'entity_pos': batch[3],
                'hts': batch[4],
                'Adj': batch[14].to(self.rank)}
        inputs['negative_mask'] = batch[13].to(self.rank) if (model_name == "2" and batch[13] != None) else None
        return inputs
    
    def _update_net1(self, loss):
        loss.backward()
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.main_net1.parameters(), self.args.max_grad_norm)
        self.main1_opt.step()
        self.main1_schdlr.step()
        self.main_net1.zero_grad()
        self.teacher_main_net1.update()

    def _update_net2(self, loss):
        loss.backward()
        if self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.main_net2.parameters(), self.args.max_grad_norm) 
        self.main2_opt.step()
        self.main2_schdlr.step()
        self.main_net2.zero_grad()
        self.teacher_main_net2.update()

    def _pretrain(self):
        main1_pt_losses, main2_pt_losses = [], []
        print("Pre-training!")
        for step_t, batch_t in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.epoch}")):
            self.main_net1.train()
            self.main_net2.train()
            
            if (step_t+1) % 2 == 0:
                inputs = self._prepare_input(batch_t)
                loss1, *_ = self.main_net1(**inputs)
                self._update_net1(loss1)
                main1_pt_losses.append(loss1.detach().cpu().item())
                
            else:
                inputs = self._prepare_input(batch_t, model_name="2")
                loss2, *_ = self.main_net2(**inputs)
                self._update_net2(loss2)
                main2_pt_losses.append(loss2.detach().cpu().item())
            
        main1_loss = np.around(np.array(main1_pt_losses).mean(), 2)
        main2_loss = np.around(np.array(main2_pt_losses).mean(), 2)
        
        return main1_loss, main2_loss
    
    def _get_pseudo_labels_from_teacher_net(self,  inputs, rounds, model_num=1):
        model = self.teacher_main_net1 if model_num == 1 else self.teacher_main_net2
        with torch.no_grad():
            model.apply(apply_dropout)
            logits = []
            for r in range(rounds):
                _, logit, _ = model(**inputs)
                logits.append(logit.cpu())

            logits = torch.stack(logits, dim=1)
            avg_loghts = logits.mean(dim=1)
            pseudo_labels = model.model.loss_fnt.get_label(avg_loghts, num_labels=self.args.num_labels)
            t_uncert_score = logits.std(dim=1).mul(pseudo_labels).mean(1)
            
        return avg_loghts, t_uncert_score, pseudo_labels
    
    def _get_s_preds(self, inputs, rounds, model_num=1):
        model = self.main_net1 if model_num == 1 else self.main_net2
        model.apply(apply_dropout)
        logits = []
        for r in range(rounds):
            _, logit, _ = model(**inputs)
            logits.append(logit.cpu())

        logits = torch.stack(logits, dim=1)

    def _get_disagreement_idx(self, idx_1, idx_2):
        union, counts = torch.cat([idx_1, idx_2]).unique(return_counts=True)
        agreement_index = union[torch.where(counts.gt(1))]
        union_1, counts_1 = torch.cat([idx_1, agreement_index]).unique(return_counts=True)
        disagreement_idx_1 = union_1[torch.where(counts_1 == 1)]
        union_2, counts_2 = torch.cat([idx_2, agreement_index]).unique(return_counts=True)
        disagreement_idx_2 = union_2[torch.where(counts_2 == 1)]
        return disagreement_idx_1, disagreement_idx_2

    def _train_distant(self, inputs, rounds=10, step=0):
        device = inputs['input_ids'].device

        # generate pseudo labels
        t_avg_loghts_1, t_uncertainty_score_1, pseudo_labels_1 = self._get_pseudo_labels_from_teacher_net(inputs, rounds=rounds, model_num=1)
        t_avg_loghts_2, t_uncertainty_score_2, pseudo_labels_2 = self._get_pseudo_labels_from_teacher_net(inputs, rounds=rounds, model_num=2)

        # get pos pseudo-labels
        pos_hts_mask_1 = pseudo_labels_1[:, 0] != 1
        pos_hts_mask_2 = pseudo_labels_2[:, 0] != 1
        pos_idx_1 = pos_hts_mask_1.nonzero().squeeze(1)
        pos_idx_2 = pos_hts_mask_2.nonzero().squeeze(1)
        pos_t_uncertainty_score_1 = t_uncertainty_score_1[pos_idx_1]
        pos_t_uncertainty_score_2 = t_uncertainty_score_2[pos_idx_2]
        _, pos_hicon_idx_1 = torch.topk(pos_t_uncertainty_score_1, k=int(0.3 * pos_t_uncertainty_score_1.size(0)), largest=False)
        _, pos_hicon_idx_2 = torch.topk(pos_t_uncertainty_score_2, k=int(0.3 * pos_t_uncertainty_score_2.size(0)), largest=False)
        pos_selected_idx_1 = pos_idx_1[pos_hicon_idx_1].to(device)
        pos_selected_idx_2 = pos_idx_2[pos_hicon_idx_2].to(device)

        # get neg pseudo-labels
        hts_tensor = torch.vstack([torch.tensor(hts) for hts in inputs['hts']])
        self_tensor_mask = hts_tensor[:, 0] == hts_tensor[:, 1]
        t_uncertainty_score_1 = t_uncertainty_score_1 + self_tensor_mask.int() * 1e30
        neg_hts_mask_1 = torch.logical_and(~pos_hts_mask_1, ~self_tensor_mask)
        neg_hts_mask_2 = torch.logical_and(~pos_hts_mask_2, ~self_tensor_mask)
        neg_idx_1 = neg_hts_mask_1.nonzero().squeeze(1)
        neg_idx_2 = neg_hts_mask_2.nonzero().squeeze(1)
        neg_t_uncertainty_score_1 = t_uncertainty_score_1[neg_idx_1]
        neg_t_uncertainty_score_2 = t_uncertainty_score_2[neg_idx_2]
        _, neg_hicon_idx_1 = torch.topk(neg_t_uncertainty_score_1, k=int(0.1 * neg_t_uncertainty_score_1.size(0)), largest=False)
        _, neg_hicon_idx_2 = torch.topk(neg_t_uncertainty_score_2, k=int(0.1 * neg_t_uncertainty_score_2.size(0)), largest=False)
        neg_selected_idx_1 = neg_idx_1[neg_hicon_idx_1].to(device)
        neg_selected_idx_2 = neg_idx_2[neg_hicon_idx_2].to(device)

        selected_idx_1 = torch.cat((pos_selected_idx_1, neg_selected_idx_1))
        selected_idx_2 = torch.cat((pos_selected_idx_2, neg_selected_idx_2))

        # student predictions
        _, logits1_u, _ = self.main_net1(**inputs)
        _, logits2_u, _ = self.main_net2(**inputs)

        # self-update from teacher net
        if self.args.self_training:
            sub_logits1_u = logits1_u[selected_idx_1, :]
            sub_logits2_u = logits2_u[selected_idx_2, :]
            sub_pseudo_labels_1 = pseudo_labels_1.to(device)[selected_idx_1, :]
            sub_pseudo_labels_2 = pseudo_labels_2.to(device)[selected_idx_2, :]
            self_sup_loss_1 = self.main_net1.loss_fnt(sub_logits1_u, sub_pseudo_labels_1.float())
            self_sup_loss_2 = self.main_net2.loss_fnt(sub_logits2_u, sub_pseudo_labels_2.float())
        else:
            self_sup_loss_1 = torch.tensor(0.0).to(logits1_u.device)
            self_sup_loss_2 = torch.tensor(0.0).to(logits2_u.device)

        # cross-label supervision (co-training)
        if (step + 1) % self.args.cross_update_step == 0 and self.args.cross_update:
            disagreement_idx_1, disagreement_idx_2 = self._get_disagreement_idx(selected_idx_1, selected_idx_2)
            selected_t_avg_loghts_1 = t_avg_loghts_1.to(device)[disagreement_idx_1, :]
            selected_t_avg_loghts_2 = t_avg_loghts_2.to(device)[disagreement_idx_2, :]
            sub_logits1_u_2 = logits1_u[disagreement_idx_2, :]
            sub_logits2_u_2 = logits2_u[disagreement_idx_1, :]
            kd_loss_1 = F.mse_loss(sub_logits1_u_2, selected_t_avg_loghts_2)
            kd_loss_2 = F.mse_loss(sub_logits2_u_2, selected_t_avg_loghts_1)
            loss_1 = self_sup_loss_1 + 0.1 * kd_loss_1 if self.args.self_training else kd_loss_1
            loss_2 = self_sup_loss_2 + 0.1 * kd_loss_2 if self.args.self_training else kd_loss_2
            return loss_1, self_sup_loss_1, kd_loss_1, loss_2, self_sup_loss_2, kd_loss_2
        
        return self_sup_loss_1, torch.zeros_like(self_sup_loss_1), torch.zeros_like(self_sup_loss_1), self_sup_loss_2, torch.zeros_like(self_sup_loss_2), torch.zeros_like(self_sup_loss_2)
    
    def _cotrain(self):
        losses_s_1, losses_s_2 = [], []
        self_sup_losses_1, self_sup_losses_2 = [], []
        kd_losses_1, kd_losses_2 = [], []
        for step_t, batch_t in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.epoch}")):
            inputs_t = self._prepare_input(batch_t)
            loss_u_1 = torch.zeros(1).to(self.rank)
            loss_u_2 = torch.zeros(1).to(self.rank)
            
            # for i in range(self.args.step_ahead):
            batch_dst = next(self.distant_loader)
            inputs = self._prepare_input(batch_dst)
            loss_u1, self_sup_loss_1, kd_loss_1, loss_u2, self_sup_loss_2, kd_loss_2 = self._train_distant(inputs, step=step_t)
            loss_u_1 = loss_u_1 + loss_u1
            loss_u_2 = loss_u_2 + loss_u2
            
            # normal training
            loss_s_1, *_ = self.main_net1(**inputs_t)
            loss_s_2, *_ = self.main_net2(**inputs_t)
                        
            # update net1
            loss_1 = loss_s_1 + 0.5 * loss_u_1
            self._update_net1(loss_1)
            # losses_1.append(loss_1.detach().cpu().item())
            losses_s_1.append(loss_s_1.detach().cpu().item())
            self_sup_losses_1.append(self_sup_loss_1.detach().cpu().item())
            kd_losses_1.append(kd_loss_1.detach().cpu().item())
            
            # update net2
            loss_2 = loss_s_2 + 0.5 * loss_u_2
            self._update_net2(loss_2)
            # losses_2.append(loss_2.detach().cpu().item())
            losses_s_2.append(loss_s_2.detach().cpu().item())
            self_sup_losses_2.append(self_sup_loss_2.detach().cpu().item())
            kd_losses_2.append(kd_loss_2.detach().cpu().item())

        losses_s_1 = np.around(np.array(losses_s_1).mean(), 2)
        losses_s_2 = np.around(np.array(losses_s_2).mean(), 2)
        self_sup_losses_1 = np.around(np.array(self_sup_losses_1).mean(), 2)
        self_sup_losses_2 = np.around(np.array(self_sup_losses_2).mean(), 2)
        kd_losses_1 = np.around(np.array(kd_losses_1).mean(), 2)
        kd_losses_2 = np.around(np.array(kd_losses_2).mean(), 2)
        return losses_s_1, losses_s_2, self_sup_losses_1, self_sup_losses_2, kd_losses_1, kd_losses_2

    def train(self):
        self.best_val_s_1 = 0
        self.best_val_s_2 = 0
        self.best_val_t_1 = 0
        self.best_val_t_2 = 0
        self.epoch = 0
        self.args.steps = 0
        
        for epoch in range(int(self.args.num_train_epochs)):
            self.epoch += 1
            if self.epoch <= self.args.warmup_epochs:
                losses1, losses2 = self._pretrain()
                print(f"Epoch {self.epoch}: Main Net 1 Training Losses: {losses1} - Main Net 2 Training Losses: {losses2}")
            else:
                losses_s_1, losses_s_2, self_sup_losses_1, self_sup_losses_2, kd_losses_1, kd_losses_2 = self._cotrain()
                print(f"Epoch {self.epoch}: Main Net 1 Supervised Losses: {losses_s_1} - Main Net 2 Supervised Losses: {losses_s_2}")
                print(f"Epoch {self.epoch}: Main Net 1 Self-sup Losses: {self_sup_losses_1} - Main Net 2 Self-sup Losses: {self_sup_losses_2}")
                print(f"Epoch {self.epoch}: Main Net 1 KD Losses: {kd_losses_1} - Main Net 2 KD Losses: {kd_losses_2}")
        
            self._epoch_evaluation()
            
    def _epoch_evaluation(self):
        print(f"Epoch {self.epoch}:")
        print("Evaluating Student Net 1...")
        val_f1_s_1, val_output_1 = evaluate(self.rank, self.main_net1, self.dev_loader, self.args, tag="dev")
        print(f"Student Net 1: {val_output_1}")

        print("Evaluating Teacher Net 1...")
        val_f1_t_1, val_t_output_1 = evaluate(self.rank, self.teacher_main_net1, self.dev_loader, self.args, tag="dev")
        print(f"Teacher Net 1: {val_t_output_1}")
        
        print("Evaluating Student Net 2...")
        val_f1_s_2, val_output_2 = evaluate(self.rank, self.main_net2, self.dev_loader, self.args, tag="dev")
        print(f"Student Net 2: {val_output_2}")

        print("Evaluating Teacher Net 2...")
        val_f1_t_2, val_t_output_2 = evaluate(self.rank, self.teacher_main_net2, self.dev_loader, self.args, tag="dev")
        print(f"Teacher Net 2: {val_t_output_2}")
        
        if self.epoch > self.args.warmup_epochs:
            _, test_output_s_1 = evaluate(self.rank, self.main_net1, self.test_loader, self.args, tag="test")
            print(f"Student Net 1: {test_output_s_1}")

            _, test_output_t_1 = evaluate(self.rank, self.teacher_main_net1, self.test_loader, self.args, tag="test")
            print(f"Teacher Net 1: {test_output_t_1}")
            
            _, test_output_s_2 = evaluate(self.rank, self.main_net2, self.test_loader, self.args, tag="test")
            print(f"Student Net 2: {test_output_s_2}")

            _, test_output_t_2 = evaluate(self.rank, self.teacher_main_net2, self.test_loader, self.args, tag="test")
            print(f"Teacher Net 2: {test_output_t_2}")
            
        if val_f1_s_1 > self.best_val_s_1:
            self.best_val_s_1 = val_f1_s_1
            print("Best score Student 1!")
            if self.args.save_path_main_1 != "":
                torch.save(self.main_net1.state_dict(), self.args.save_path_main_1)

        if val_f1_t_1 > self.best_val_t_1:
            self.best_val_t_1 = val_f1_t_1
            print("Best score Teacher 1!")
            if self.args.save_path_teacher_1 != "":
                torch.save(self.teacher_main_net1.state_dict(), self.args.save_path_teacher_1)
                
        if val_f1_s_2 > self.best_val_s_2:
            self.best_val_s_2 = val_f1_s_2
            print("Best score Student 2!")
            if self.args.save_path_main_2 != "":
                torch.save(self.main_net2.state_dict(), self.args.save_path_main_2)

        if val_f1_t_2 > self.best_val_t_2:
            self.best_val_t_2 = val_f1_t_2
            print("Best score Teacher 2!")
            if self.args.save_path_teacher_2 != "":
                torch.save(self.teacher_main_net2.state_dict(), self.args.save_path_teacher_2)
        print("-------------")
        
    def report(self, model):
        ans = report(self.rank, model, self.test_loader, self.args, tag="test")
        return ans