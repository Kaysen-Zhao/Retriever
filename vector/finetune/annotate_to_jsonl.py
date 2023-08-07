# -*- coding: UTF-8 -*-
import json
import random
from tqdm import tqdm
from bson.objectid import ObjectId

import sys, os
sys.path.append(os.getcwd())

from mongo.connect import get_segment_co
from es.connect import access_es_api


# dev_num: 129, train_num: 1277, all: 1406
data_path = 'data/datasets/annotate100/data_expand.json'
institution = '安徽大学'
neg_method = 'title__article_name__content'
neg_top_n = 200
dev_rate = 0.1


with open(data_path, 'r', encoding='utf8') as r:
    data = json.load(r)
    
def doc_to_text(doc):
    text = ''
    for f in ['title', 'article_name', 'content']:  # 多个领域
        if doc[f].strip():
            text += f'{f}: ' + doc[f].strip() + '\n'
    text = text.strip()
    return text

w_train = open(os.path.join(os.path.dirname(data_path), 'train.jsonl'), 'w', encoding='utf8')
w_dev = open(os.path.join(os.path.dirname(data_path), 'dev.jsonl'), 'w', encoding='utf8')
dev_num = train_num = 0

for one in tqdm(data, '样本转换'):
    # 正例
    positive_passages = []
    positive_passages_ids = set()
    for doc_id in one['doc_id']:
        _id = ObjectId(doc_id)
        doc = get_segment_co().find_one({'_id': _id})
        positive_passages.append({
            'docid': doc_id,
            'text': doc_to_text(doc),
        })
        positive_passages_ids.add(doc_id)
    for q, qe in zip(one['query'], one['query_expand']):
        q = q.strip()
        qe = qe["chatglm-6b"].strip()
        for query in [q, f'{q}\n{qe}']:
            # 负例
            negative_passages = []
            ret_data = access_es_api({
                "query": query,
                "filter" : {"institution": f'"{institution}"'},
                "settings": {
                    neg_method: True,
                    "top_n": neg_top_n,
                }
            }, institution)
            for docid_sim in ret_data['results'][neg_method]:
                if docid_sim['_id'] in positive_passages_ids:
                    continue
                _id = ObjectId(docid_sim['_id'])
                doc = get_segment_co().find_one({'_id': _id})
                negative_passages.append({
                    'docid': docid_sim['_id'],
                    'text': doc_to_text(doc),
                })
            # 切分 train/dev
            if random.random() < dev_rate:
                n = dev_num
                dev_num += 1
                w = w_dev
            else:
                n = train_num
                train_num += 1
                w = w_train
            w.write(json.dumps({
                'query_id': n,
                'query': query,
                'positive_passages': positive_passages,
                'negative_passages': negative_passages,
            }, ensure_ascii=False) + '\n')
w_dev.close()
w_train.close()
print(f'dev_num: {dev_num}, train_num: {train_num}, all: {dev_num + train_num}')
