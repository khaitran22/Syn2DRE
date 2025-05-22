from tqdm import tqdm
import ujson as json
import torch
import numpy as np
import copy

docred_rel2id = json.load(open('/scratch/user/uqptran9/Code/Work_3/dev/dataset/re_docred/meta/rel2id.json', 'r'))
docred_ner2id = json.load(open('/scratch/user/uqptran9/Code/Work_3/dev/dataset/re_docred/meta/ner2id.json', 'r'))
# docred_rel2id = json.load(open('./dataset/re_docred/meta/rel2id.json', 'r'))
# docred_ner2id = json.load(open('./dataset/re_docred/meta/ner2id.json', 'r'))

cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}

def batch_index(tensor, index, pad=False):
    if tensor.shape[0] != index.shape[0]:
        raise Exception()

    if not pad:
        return torch.stack([tensor[i][index[i]] for i in range(index.shape[0])])
    else:
        return padded_stack([tensor[i][index[i]] for i in range(index.shape[0])])

def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor

def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked

def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def addEntitySentence(entities, curSent):
    vertexSentence = {}
    for enti in entities:
        for menti in enti:
            key = menti['sent_id'] * 10 + menti['pos'][0]
            vertexSentence[key] = copy.deepcopy(menti)
    vertexlists = sorted(vertexSentence.items(), key=lambda item: item[0])
    vertexsentencelist = []
    for item in vertexlists:
        vertexsentencelist += curSent[item[1]['sent_id']
                                      ][item[1]['pos'][0]:item[1]['pos'][1]]
    return vertexsentencelist

def read_docred(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []
        words = []
        token_map = []
        lengthofPice = 0
        mentions = []
        inter_mask = []
        
        for i in range(0, len(sample['vertexSet'])):
            inter_mask.append([1] * len(sample['vertexSet']))
        for i in range(0, len(sample['vertexSet'])):
            inter_mask[i][i] = 0
        for i, ent1 in enumerate(sample['vertexSet']):
            for j, ent2 in enumerate(sample['vertexSet']):
                if i != j:
                    breakFlag = 0
                    for men1 in ent1:
                        for men2 in ent2:
                            if men1['sent_id'] == men2['sent_id']:
                                inter_mask[i][j] = 0
                                inter_mask[j][i] = 0
                                breakFlag = 1
                                break
                        if breakFlag == 1:
                            break

        entities = sample['vertexSet']
        vertexsentencelist = addEntitySentence(entities, sample['sents'])
        entity_start, entity_end = {}, {}
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                mentions.append(pos)

                entity_start[(sent_id, pos[0],)] = "*"
                entity_end[(sent_id, pos[1] - 1,)] = "*"
            
        newsents = sample['sents']
        newsents.append(vertexsentencelist)
        
        for i_s, sent in enumerate(newsents):
            new_map = {}
            for i_t, token in enumerate(sent):

                oneToken = []
                words.append(token)
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start.keys():
                    tokens_wordpiece = [
                        entity_start[(i_s, i_t)]] + tokens_wordpiece
                    oneToken.append(lengthofPice + 1)
                    lengthofPice += len(tokens_wordpiece)
                    oneToken.append(lengthofPice)

                elif (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + \
                        [entity_end[(i_s, i_t)]]
                    oneToken.append(lengthofPice)
                    lengthofPice += len(tokens_wordpiece)
                    oneToken.append(lengthofPice-1)
                else:
                    oneToken.append(lengthofPice)
                    lengthofPice += len(tokens_wordpiece)
                    oneToken.append(lengthofPice)
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                token_map.append(oneToken)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)
        
        sent_starts = [x[0] for x in sent_map]
        sentid_mask = torch.zeros( (len(sents) + 2))
        for sent_id in range(len(sent_starts)):
            sentid_mask[sent_starts[sent_id]:] = (sent_id+1)
        
        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                if label['r'] not in docred_rel2id.keys():
                    continue
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r})

        entity_pos = []
        entity_types = []
        mention_pos = []
        ent2mention = []
        mention2ent = []
        mention_id = 0
        entity_id = 0
        sent_occur = {}
        for e in entities:
            entity_pos.append([])
            name_lens = np.array([len(x['name']) for x in e])
            long_idx = np.argmax(name_lens)
            entity_types.append(docred_ner2id[e[long_idx]['type']])
            ent2mention.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
                mention_pos.append((start,end))
                mention2ent.append(entity_id)
                ent2mention[-1].append(mention_id)
                mention_id += 1
                
                if m["sent_id"] not in sent_occur.keys():
                    sent_occur[m["sent_id"]] = []
                if entity_id not in sent_occur[m["sent_id"]]:
                    sent_occur[m["sent_id"]].append(entity_id)
                    
            entity_id +=1 
        
        relations, hts = [], []
        mention_hts = []
        mention_hts_id = 0
        hts2mention_hts = []
        convert_lens = []
        for h in range(len(entities)):
            for t in range(len(entities)):
                hts2mention_hts.append([])
                for h_m in range(len(ent2mention[h])):
                    for t_m in range(len(ent2mention[t])):
                        mention_hts.append((ent2mention[h][h_m], ent2mention[t][t_m]))
                        hts2mention_hts[-1].append(mention_hts_id)
                        mention_hts_id += 1
                convert_lens.append(len(hts2mention_hts[-1]))
                if (h, t) in train_triple.keys():
                    relation = [0] * len(docred_rel2id)
                    for mention in train_triple[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h,t])
                    pos_samples += 1
                elif (h, t) not in train_triple.keys():
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1
        max_men = max(convert_lens)
        padded_mention = []
        padded_mention_mask = []
        for ent_idx in range(len(hts)):
            temp = torch.zeros(( max_men))
            temp_1 = torch.zeros(( max_men))
            temp[:len(hts2mention_hts[ent_idx])] = torch.tensor(hts2mention_hts[ent_idx])
            temp_1[:len(hts2mention_hts[ent_idx])] = 1
            padded_mention.append(temp)
            padded_mention_mask.append(temp_1)
        padded_mention = torch.stack(padded_mention, dim=0).long()
        padded_mention_mask = torch.stack(padded_mention_mask, dim=0).long()
        assert len(mention_hts) == len(mention_pos) * len(mention_pos)        
        assert len(relations) == len(entities) * len(entities)
        
        sents = sents[:max_seq_length - 2]
        sentid_mask = sentid_mask[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        
        A = []
        for i in range(0, len(input_ids)):
            A.append([0]*len(input_ids))
        offset = 1
        edges = 0
        for token_s in token_map:
            start = token_s[0]+offset
            end = token_s[1]+offset
            for i in range(start, end):
                for j in range(start, end):
                    if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
                        if A[i][j] == 0:
                            A[i][j] = 1
                            edges += 1
        mentionsofPice = []
        for ment in mentions:
            mentionsofPice.append(
                [token_map[ment[0]][0], token_map[ment[1]-1][1]])
        for ment in mentionsofPice:
            start = ment[0]+offset
            end = ment[1]+offset
            for i in range(start, end):
                for j in range(start, end):
                    if i < (len(input_ids) - 1) and j < (len(input_ids) - 1):
                        if A[i][j] == 0:
                            A[i][j] = 1
                            edges += 1
        entityofPice = []
        for ent in entity_pos:
            oneEntityP = []
            for ment in ent:
                if (ment[0]+offset) == (ment[1]-offset):
                    oneEntityP.append(ment[0]+offset)
                for i in range(ment[0]+offset, ment[1]-offset):
                    oneEntityP.append(i)
            entityofPice.append(oneEntityP)
        predicted_Doc2 = []
        for h in range(0, len(entities)):
            item = [0, h]
            predicted_Doc2.append(item)

        predictedEntityPairPiece = []
        for item in predicted_Doc2:
            one_predicted = entityofPice[item[0]]+entityofPice[item[1]]
            predictedEntityPairPiece.append(one_predicted)

        for line in predictedEntityPairPiece:
            for i in line:
                for j in line:
                    if A[i + offset][j + offset] == 0:
                        A[i + offset][j + offset] = 1
                        edges += 1

        for i in range(0, len(A)):
            A[i][i] = 1

        if 'title' not in sample.keys():
            sample['title'] = ' '.join(sample['sents'][0])
        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'sentid_mask':sentid_mask,
                   'mention_pos':mention_pos,
                   'mention_hts':mention_hts,
                   'entity_types': entity_types,
                   'padded_mention':padded_mention,
                   'padded_mention_mask':padded_mention_mask,
                   'Adj':A,
                   'inter_mask':inter_mask,
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features


def read_docred_distant_inference(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        '''
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1
        '''
        pos_idx = []
        for h in range(len(entities)):
            for t in range(len(entities)):
                if (h, t) in train_triple.keys():
                    relation = [0] * len(docred_rel2id)
                    for mention in train_triple[h, t]:
                        relation[mention["relation"]] = 1
                        evidence = mention["evidence"]
                    relations.append(relation)
                    hts.append([h,t])
                    pos_idx.append(len(relations))
                    pos_samples += 1
                elif (h, t) not in train_triple.keys():
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * len(entities)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   'pos_idx' : pos_idx
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features


def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
