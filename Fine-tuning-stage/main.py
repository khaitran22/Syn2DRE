import argparse
from models import *
from utils import *
from trainer import Trainer
import ujson as json

import autoRun
autoRun.choose_gpu(n_gpus=1, retry=True, min_gpu_memory=10000, sleep_time=30)

def main():
    parser = argparse.ArgumentParser()

    # setup
    parser.add_argument("--data_dir", default="./dataset/docred", type=str)

    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--train_distant_file", default="train_distant.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    
    parser.add_argument("--save_path_main_1", default="", type=str)
    parser.add_argument("--save_path_main_2", default="", type=str)
    parser.add_argument("--save_path_teacher_1", default="", type=str)
    parser.add_argument("--save_path_teacher_2", default="", type=str)
    
    parser.add_argument("--save_last", default="", type=str)
    
    parser.add_argument("--load_path_main_1", default="", type=str)
    parser.add_argument("--load_path_main_2", default="", type=str)
    parser.add_argument("--load_path_teacher_1", default="", type=str)
    parser.add_argument("--load_path_teacher_2", default="", type=str)
    
    parser.add_argument("--output_name", default="result.json", type=str)

    parser.add_argument("--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int, help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--distant_batch_size", default=4, type=int, help="Batch size for training.")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int, help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int, help="Max number of labels in prediction.")

    parser.add_argument("--main1_lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--main1_classifier_lr", default=1e-4, type=float, help="The initial learning rate for Classifier.")
    
    parser.add_argument("--main2_lr", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--main2_classifier_lr", default=1e-4, type=float, help="The initial learning rate for Classifier.")

    # models
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--drop_prob", default=0.0, type=float, help="Negative Sample Discard rate.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int, help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97, help="Number of relation types in dataset.")
    parser.add_argument("--warmup_epochs", default=5, type=int, help="tau")
    parser.add_argument("--dropout", default=0.25, type=float, help="dropout")
    parser.add_argument("--agree_weight", default=0.1, type=float, help="")
    parser.add_argument("--cross_update_step", default=2, type=int, help="")
    parser.add_argument("--gamma_pos", default=1.0, type=float, help="Gamma for positive class")
    parser.add_argument("--gamma_neg", default=1.0, type=float, help="Gamma for negative class")
    
    # enable/disable Disagreement-based Cross-update and Self-supervised steps
    parser.add_argument('--cross_update', action='store_true')
    parser.add_argument('--self_training', action='store_true')
    
    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    rank = 0
    set_seed(args)
    train_distant_loader, train_loader, dev_loader, test_loader, tokenizer = prepare_data(args)
    main_net1, main_net2, teacher_main_net1, teacher_main_net2 = build_models(rank, args, tokenizer)
    trainer = Trainer(rank, args, main_net1, main_net2, teacher_main_net1, teacher_main_net2, train_loader, train_distant_loader, dev_loader, test_loader)
    if args.load_path_teacher_1 == "" or args.load_path_teacher_2 == "":
        trainer.train()
    else:
        ans = trainer.report()
        with open('./checkpoints/result.json') as f:
            json.dump(ans, f)

if __name__ == "__main__":
    main()