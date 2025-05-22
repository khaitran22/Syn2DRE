import torch
import logging
import os
import math

from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from prepro import *
from transformers import AutoConfig, AutoModel, AutoTokenizer

from models import *
from evaluation import *

from copy import deepcopy
import random
from ema_pytorch import EMA

def get_logger(filename, local_rank):
    formatter = logging.Formatter(fmt='[%(asctime)s %(levelname)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if filename is not None and local_rank <=0: # only log to file for first GPU
        f_handler = logging.FileHandler(filename, 'a')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

        stdout_handler = logging.StreamHandler()
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.INFO)
        logger.addHandler(stdout_handler)
    else: # null handlers for other GPUs
        null_handler = logging.NullHandler()
        null_handler.setLevel(logging.INFO)
        logger.addHandler(null_handler)
    
    return logger

def ddp_setup(args, rank: int, world_size: int, runid=None):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    torch.backends.cudnn.benchmark = True
    init_process_group(
        backend="nccl", 
        rank=rank, 
        world_size=world_size
    )

def tocuda(rank, data):
    if type(data) is list:
        if len(data) == 1:
            return data[0].to(rank)
        else:
            return [x.to(rank) for x in data]
    else:
        return data.to(rank)

def totensor(rank, data):
    d = [torch.tensor(d) for d in data]
    d = torch.cat(d, dim=0).to(rank)
    return d

def evaluate(rank, model, loader, args, tag="dev"):
        ''' 
            Evaluate some model on some data (modified version for single GPU)
            TODO:   Move to multiple GPU when finished
        '''
        # ncorrect = 0
        # nsamples = 0

        model.eval()
        preds = []
        for batch in loader:
            model.eval()

            inputs = {
                # 'input_ids': batch[0].to(args.device),
                # 'attention_mask': batch[1].to(args.device),
                'input_ids': batch[0].to(rank),
                'attention_mask': batch[1].to(rank),
                'entity_pos': batch[3],
                'hts': batch[4],
                'negative_mask': batch[13]
            }

            with torch.no_grad():
                pred, *_ = model(**inputs)
                pred = pred.cpu().numpy()
                pred[np.isnan(pred)] = 0
                preds.append(pred)

        preds = np.concatenate(preds, axis=0).astype(np.float32)
        ans = to_official(preds, loader.dataset)
        best_f1, _, best_f1_ign, _ = official_evaluate(ans, args.data_dir, tag) if len(ans) > 0 else (0, 0, 0, 0)
        output = {
            tag + "_F1": best_f1 * 100,
            tag + "_F1_ign": best_f1_ign * 100,
        }
        return best_f1, output
    
class DataIterator(object):
    def __init__(self, dataloader):
        assert isinstance(dataloader, torch.utils.data.DataLoader), 'Wrong loader type'
        self.loader = dataloader
        self.iterator = iter(self.loader)
        
    def __len__(self):
        return len(self.loader.dataset)

    def __next__(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            data = next(self.iterator)

        return data

def get_random_mask(train_features, drop_prob):
    new_features = []  
    n_e = 42
    for old_feature in train_features:
        feature = deepcopy(old_feature)
        neg_labels = torch.tensor(feature['labels'])[:, 0]
        neg_index = torch.where(neg_labels==1)[0]
        pos_index = torch.where(neg_labels==0)[0]
        perm = torch.randperm(neg_index.size(0))
        sampled_negative_index = neg_index[perm[:int(drop_prob * len(neg_index))]]
        neg_mask = torch.ones(len(feature['labels']))
        neg_mask[sampled_negative_index] = 0    
        pad_neg = torch.zeros((n_e, n_e))
        num_e = int(math.sqrt(len(neg_mask)))
        pad_neg[:num_e,:num_e] = neg_mask.view(num_e,num_e)
        feature['negative_mask'] = pad_neg
        new_features.append(feature)
    return new_features

def prepare_data(args):

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    
    suffix = '.{}.pt'.format(args.model_name_or_path)
    read = read_docred

    ## DATA PROCESS AND LOADING
    # train distant
    if os.path.exists(os.path.join(args.data_dir, args.train_distant_file + suffix)):
        train_distant_features = torch.load(os.path.join(args.data_dir, args.train_distant_file + suffix))
        print('Loaded train distant features')
    else:
        train_distant_file = os.path.join(args.data_dir, args.train_distant_file)
        train_distant_features = read(train_distant_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(train_distant_features, os.path.join(args.data_dir, args.train_distant_file + suffix))
        print('Created and saved new train distant features')
    
    # train revised
    if os.path.exists(os.path.join(args.data_dir, args.train_file + suffix)):
        train_features = torch.load(os.path.join(args.data_dir, args.train_file + suffix))
        print('Loaded train features')
    else:
        train_file = os.path.join(args.data_dir, args.train_file)
        train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(train_features, os.path.join(args.data_dir, args.train_file + suffix))
        print('Created and saved new train features')
    # masked_train_features = get_random_mask(train_features, args.drop_prob)
    
    # dev revised
    if os.path.exists(os.path.join(args.data_dir, args.dev_file + suffix)):
        dev_features = torch.load(os.path.join(args.data_dir, args.dev_file + suffix))
        print('Loaded dev features')
    else:
        dev_file = os.path.join(args.data_dir, args.dev_file)
        dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(dev_features, os.path.join(args.data_dir, args.dev_file + suffix))
        print('Created and saved new dev features')
    
    # test revised
    if os.path.exists(os.path.join(args.data_dir, args.test_file + suffix)):
        test_features = torch.load(os.path.join(args.data_dir, args.test_file + suffix))
        print('Loaded test features')
    else:
        test_file = os.path.join(args.data_dir, args.test_file)
        test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
        torch.save(test_features, os.path.join(args.data_dir, args.test_file + suffix))
        print('Created and saved new test features')
    
    # DATA LOADER
    train_distant_loader = DataIterator(DataLoader(train_distant_features, batch_size=args.distant_batch_size, shuffle=True, collate_fn=collate_fn_kd, drop_last=True))
    train_loader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn_kd, drop_last=True)
    dev_loader = DataLoader(dev_features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn_kd, drop_last=False)
    test_loader = DataLoader(test_features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn_kd, drop_last=False)

    return train_distant_loader, train_loader, dev_loader, test_loader, tokenizer

def build_models(rank, args, tokenizer):
    # Main 1
    main_config1 = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    main_encoder1 = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=main_config1,
    )
    main_config1.cls_token_id = tokenizer.cls_token_id
    main_config1.sep_token_id = tokenizer.sep_token_id
    main_config1.transformer_type = args.transformer_type
    main_net1 = DocREModel(main_config1, args, main_encoder1, num_labels=args.num_labels)
    main_net1.to(rank)
    if args.load_path_main_1 != "":
        main_net1.load_state_dict(torch.load(args.load_path_main_1))
        print(f"Loaded main_net1 from {args.load_path_main_1}")
    teacher_main_net1 = EMA(
        main_net1,
        beta=0.995,
        update_after_step=10,
        update_every=5
    )
    if args.load_path_teacher_1 != "":
        teacher_main_net1.load_state_dict(torch.load(args.load_path_teacher_1))
    teacher_main_net1.to(rank)
    
    # Main 2
    main_config2 = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    main_encoder2 = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=main_config2,
    )
    main_config2.cls_token_id = tokenizer.cls_token_id
    main_config2.sep_token_id = tokenizer.sep_token_id
    main_config2.transformer_type = args.transformer_type
    main_net2 = KD_DocRE(args, main_config2, main_encoder2, num_labels=args.num_labels)
    main_net2.to(rank)
    if args.load_path_main_2 != "":
        main_net2.load_state_dict(torch.load(args.load_path_main_2))
        print(f"Loaded main_net2 from {args.load_path_main_2}")
    teacher_main_net2 = EMA(
        main_net2,
        beta=0.999,
        update_after_step=10,
        update_every=5
    )
    if args.load_path_teacher_2 != "":
        teacher_main_net2.load_state_dict(torch.load(args.load_path_teacher_2))
    teacher_main_net2.to(rank)

    return main_net1, main_net2, teacher_main_net1, teacher_main_net2

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    sentid_mask = [ torch.cat([f["sentid_mask"] , torch.zeros((max_len - len(f["sentid_mask"])))]) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    mention_pos = [f["mention_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    mention_hts = [f["mention_hts"] for f in batch]
    padded_mention = [f["padded_mention"] for f in batch]
    padded_mention_mask = [f["padded_mention_mask"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    sentid_mask = torch.stack(sentid_mask)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    #labels = [torch.tensor(label) for label in labels]
    output = (input_ids, input_mask, labels, entity_pos, hts, mention_pos, mention_hts, padded_mention, padded_mention_mask, sentid_mask)
    return output

def collate_fn_kd(batch):
    teacher_logits = None
    segment_spans = None
    entity_types = None
    max_len = max([len(f["input_ids"]) for f in batch])
    max_ent_len = max([len(f["entity_pos"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    sentid_mask = None
    neg_masks = None
    #sentid_mask = [ torch.cat([f["sentid_mask"] , torch.zeros((max_len - len(f["sentid_mask"])))]) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    mention_pos = [f["mention_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    mention_hts = [f["mention_hts"] for f in batch]
    padded_mention = [f["padded_mention"] for f in batch]
    padded_mention_mask = [f["padded_mention_mask"] for f in batch]
    
    Adjs = []
    for f in batch:
        Adj = []
        for i in range(0, max_len):
            Adj.append([0] * max_len)
        A = f['Adj']
        for i in range(0, len(A)):
            for j in range(0, len(A)):
                Adj[i][j] = A[i][j]
        Adjs.append(Adj)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    
    Adj = torch.tensor(Adjs, dtype=torch.float)
    
    if "teacher_logits" in batch[0]:
        teacher_logits = [f["teacher_logits"] for f in batch]
    if "segment_span" in batch[0]:
        segment_spans = [f["segment_span"] for f in batch]
    if "entity_types" in batch[0]:
        entity_types = [x for f in batch for x in f['entity_types'] ]
        entity_types = [f['entity_types'] for f in batch ]
    if "negative_mask" in batch[0]:
        neg_masks = torch.stack([f['negative_mask'] for f in batch], dim=0)
    output = (input_ids, input_mask, labels, entity_pos, hts, mention_pos, mention_hts, padded_mention, padded_mention_mask, sentid_mask, teacher_logits, entity_types, segment_spans, neg_masks, Adj)
    return output

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
def evaluate(rank, model, loader, args, tag="dev", pretrain=False):
        ''' 
            Evaluate some model on some data (modified version for single GPU)
        '''

        model.eval()
        preds = []
        for batch in loader:
            model.eval()

            inputs = {
                'input_ids': batch[0].to(rank),
                'attention_mask': batch[1].to(rank),
                'entity_pos': batch[3],
                'hts': batch[4],
                'negative_mask': batch[13],
                'Adj': batch[14].to(rank)
            }

            with torch.no_grad():
                pred, *_ = model(**inputs)
                pred = pred.cpu().numpy()
                pred[np.isnan(pred)] = 0
                preds.append(pred)

        preds = np.concatenate(preds, axis=0).astype(np.float32)
        ans = to_official(preds, loader.dataset)
        dev_file = args.dev_file if tag == 'dev' else args.test_file
        best_f1, _, best_f1_ign, _, re_p, re_r, best_f1_freq, best_f1_long_tail, _, _, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail = official_evaluate_benchmark(ans, args.data_dir, args.train_file, dev_file) if len(ans) > 0 else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        output = {
            tag + "_ign_F1": np.around(best_f1_ign * 100, decimals=2),
            tag + "_F1": np.around(best_f1 * 100, decimals=2),
            tag + "_p": np.around(re_p * 100, decimals=2),
            tag + "_r": np.around(re_r * 100, decimals=2),
            tag + "_F1_freq": np.around(best_f1_freq * 100, decimals=2),
            tag + "_p_freq": np.around(re_p_freq * 100, decimals=2),
            tag + "_r_freq": np.around(re_r_freq * 100, decimals=2),
            tag + "_F1_LT": np.around(best_f1_long_tail * 100, decimals=2),
            tag + "_p_LT": np.around(re_p_long_tail * 100, decimals=2),
            tag + "_r_LT": np.around(re_r_long_tail * 100, decimals=2),
        }
        return best_f1, output
    
def report(rank, model, loader, args, tag="dev", pretrain=False):
    model.eval()
    preds = []
    for batch in loader:
        model.eval()

        inputs = {
            'input_ids': batch[0].to(rank),
            'attention_mask': batch[1].to(rank),
            'entity_pos': batch[3],
            'hts': batch[4],
            'negative_mask': batch[13],
            'Adj': batch[14].to(rank)
        }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, loader.dataset)
    return ans