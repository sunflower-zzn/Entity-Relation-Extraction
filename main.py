# -*- coding: UTF-8 -*-
# flask web-server
import itertools

import requests
import os
import json
from flask_cors import *

os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
from flask import Flask, request

app = Flask(__name__)

# entity identity
from LAC import LAC
import re

# 装载LAC模型
lac = LAC(mode='lac')

# relation-extraction
import numpy as np
import torch.nn.functional
import warnings
import torch
from transformers import BertTokenizer
import random
from loader import map_id_rel


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


warnings.filterwarnings("ignore")
setup_seed(44)
rel2id, id2rel = map_id_rel()


def rest_response(context, err=False):
    response = {}
    if err:
        response['errmsg'] = context
        response['context'] = None
    else:
        response['errmsg'] = None
        response['context'] = context
    return json.dumps(response, ensure_ascii=False, indent=4)


def extractRel(net_path, text_list, entity_list):
    max_length = 128
    net = torch.load(net_path, map_location=torch.device('cpu'))
    net.eval()
    result = []
    with torch.no_grad():
        for text, entitys in zip(text_list, entity_list):
            single_result = {'text': text}
            rel_list = []
            for ent1, ent2 in list(itertools.combinations(entitys, 2)):
                sent = ent1 + ent2 + text
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
                avai_len = len(indexed_tokens)
                while len(indexed_tokens) < max_length:
                    indexed_tokens.append(0)  # 0 is id for [PAD]
                indexed_tokens = indexed_tokens[: max_length]
                indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
                # Attention mask
                att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
                att_mask[0, :avai_len] = 1
                outputs = net(indexed_tokens, att_mask)
                logits = outputs[0]
                _, predicted = torch.max(logits.data, 1)
                rel_result = predicted.cpu().numpy().tolist()[0]
                rel_list.append({'ent1': ent1, 'ent2': ent2, 'rel': id2rel[rel_result]})
            single_result['relations'] = rel_list
            result.append(single_result)
    return result


@app.route('/relation/extraction', methods=['POST'])
def extractRelation():
    try:
        text_list = []
        entity_list = []
        for param in request.json:
            text_list.append(param.get('text'))
            entity_list.append(param.get('entitys'))
        context = extractRel('net.pth', text_list, entity_list)
        return rest_response(context)
    except Exception as e:
        print(e.args)
        return rest_response(e.args, True)


@app.route('/identity', methods=['POST'])
def entityIdentity():
    try:
        ignore_short_sent = request.json.get('is_ignore')
        # 段落分句
        paragraph = str(request.json.get('text')).replace('\t', '').replace(' ', '')
        sentences = re.split('[。！？\n]', paragraph)
        new_sents = []
        for s in sentences:
            if len(s) > 10:  # 太短的句子抛弃
                new_sents.append(s)
        # 按句子输入LAC命名实体识别模型
        result = []
        for i in lac.run(new_sents):
            entity = []
            uniq_set = []
            for j in range(len(i[0])):
                if not uniq_set.__contains__(j):
                    uniq_set.append(j)
                    if i[1][j] == 'PER':
                        entity.append({'ent': i[0][j], 'type': '人名'})
                    elif i[1][j] == 'LOC':
                        entity.append({'ent': i[0][j], 'type': '地名'})
                    elif i[1][j] == 'ORG':
                        entity.append({'ent': i[0][j], 'type': '组织名'})
                    elif i[1][j] == 'TIME':
                        entity.append({'ent': i[0][j], 'type': '时间'})
                    elif i[1][j] == 'nz':
                        entity.append({'ent': i[0][j], 'type': '专有名词'})
                    elif i[1][j] == 'nw':
                        entity.append({'ent': i[0][j], 'type': '作品名'})
            ner = {'text': ''.join(i[0]), 'entity': entity}
            if ignore_short_sent:
                if len(entity) < 2:
                    continue
            result.append(ner)
        return rest_response(result)
    except Exception as e:
        print(e.args)
        return rest_response(e.args, True)


@app.route('/types', methods=['GET'])
def getRelTypes():
    return json.dumps(id2rel, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5590)
