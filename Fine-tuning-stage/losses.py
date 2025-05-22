import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels, weights=None):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()
        return loss

    def get_uncertainty_score(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        uncertainty_score = loss1 + loss2
        return uncertainty_score

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output

    
class AFLoss(nn.Module):
    def __init__(self, gamma_pos, gamma_neg):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, labels, weights=None):        
        
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        n_mask = 1 - labels
        _, num_class = labels.size()

        # Rank each class to threshold class TH
        th_mask = torch.cat(num_class * [logits[:, :1]], dim=1)
        logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
        log_probs = F.log_softmax(logit_th, dim=1)
        probs = torch.exp(F.log_softmax(logit_th, dim=1))
        prob_0 = probs[:, 1 ,:]
        prob_0_gamma = torch.pow(prob_0, self.gamma_pos)
        log_prob_1 = log_probs[:, 0 ,:]
        
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        rank2 = F.log_softmax(logit2, dim=-1)
        loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels).sum(1) 
        loss2 = -(rank2 * th_label).sum(1)
        loss =  (1.0 * loss1 + 1.0 * loss2)
        if weights is not None:
            loss = loss * weights
        loss = loss.mean()
        
        return loss

    def get_uncertainty_score(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0
        n_mask = 1 - labels
        _, num_class = labels.size()

        # Rank each class to threshold class TH
        th_mask = torch.cat(num_class * [logits[:, :1]], dim=1)
        logit_th = torch.cat([logits.unsqueeze(1), 1.0 * th_mask.unsqueeze(1)], dim=1) 
        log_probs = F.log_softmax(logit_th, dim=1)
        probs = torch.exp(F.log_softmax(logit_th, dim=1))
        prob_0 = probs[:, 1 ,:]
        prob_0_gamma = torch.pow(prob_0, self.gamma_pos)
        log_prob_1 = log_probs[:, 0 ,:]
        
        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        rank2 = F.log_softmax(logit2, dim=-1)
        loss1 = - (log_prob_1 * (1 + prob_0_gamma ) * labels).sum(1)
        loss2 = -(rank2 * th_label).sum(1)
        uncertainty_score =  (1.0 * loss1 + 1.0 * loss2)
        return uncertainty_score

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1) * 1.0
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output
    

class SCLoss(nn.Module):
    def __init__(self, tau=2.0, tau_base=1.0):
        super().__init__()
        self.tau = tau
        self.tau_base = tau_base

    def forward(self, features, labels, weights=None):
        labels = labels.long()
        mask_s = torch.any((labels.unsqueeze(1) & labels).bool(), dim=-1).float().fill_diagonal_(0)

        sims = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2).div(self.tau)
        
        logits_max, _ = torch.max(sims, dim=1, keepdim=True)
        logits = sims - logits_max.detach()
        logits_mask = torch.ones_like(mask_s).fill_diagonal_(0)

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        denom = mask_s.sum(1)
        denom[denom == 0] = 1     # avoid div by zero

        log_prob1 = (mask_s * log_prob).sum(1) / denom

        loss = - (self.tau/self.tau_base) * log_prob1
        loss = loss.mean()

        return loss


class NCRLoss(nn.Module):
    def __init__(self, shift=0.05, isReg=True, eps=1e-8, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.isReg = isReg
        self.eps = eps
        self.reduction = reduction

    def compute_CE(self, x, y, weights=None):
        """
        Adapted from "Asymmetric Loss For Multi-Label Classification"
        https://arxiv.org/abs/2009.14119
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Margin Shifting
        if self.shift is not None and self.shift > 0:
            xs_neg = (xs_neg + self.shift).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if weights is not None:
            loss = loss * weights

        return -loss.sum()

    def forward(self, logits, labels, weights=None):
        # Logit margin for pre-defined relations 
        rel_margin = logits[:,1:] - logits[:,0].unsqueeze(1)
        loss = self.compute_CE(rel_margin.float(), labels[:,1:].float(), weights=weights)
        
        if self.isReg: # Enable margin regularization
            # Logit margin for the none class label
            na_margin = logits[:,0] - logits[:,1:].mean(-1) 
            loss += self.compute_CE(na_margin.float(), labels[:,0].float(), weights=weights)
        
        if self.reduction == "mean":
            loss /= labels.shape[0]

        return loss

    def get_label(self, logits, num_labels=-1):
        """Copied from https://github.com/wzhouad/ATLOP/blob/main/losses.py#L32"""
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class PMTEMloss(nn.Module):
    def __init__(self, lambda_1, lambda_2):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2


    def forward(self, logits, labels):
        label_mask = (labels[:, 0] != 1.)

        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

 
        count = labels.sum(1, keepdim=True)
        count[count==0] = 1


        th = logits[:, :1].expand(logits.size(0), logits.size(1))

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        probs1 = F.softmax(torch.cat([th.unsqueeze(1), logit1.unsqueeze(1)], dim=1), dim=1)
        loss1 = -(torch.log(probs1[:, 1] + 1e-30) * labels).sum(1)
        loss3 = -(((probs1 * torch.log(probs1 + 1e-30)).sum(1)) * labels ).sum(1) / count


        count = (1-p_mask).sum(1, keepdim=True)
        count[count==0] = 1


        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        probs2 = F.softmax(torch.cat([th.unsqueeze(1), logit2.unsqueeze(1)], dim=1), dim=1)
        loss2 = -(torch.log(probs2[:, 0] + 1e-30) * (1 - p_mask)).sum(1)
        loss4 = -(((probs2 * torch.log(probs2 + 1e-30)).sum(1)) * (1 - p_mask)).sum(1) / count

            
        # Sum  parts
        loss = loss1 + loss2 + self.lambda_1*loss3 + self.lambda_2*loss4
        loss = loss.mean()
        return loss


    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output