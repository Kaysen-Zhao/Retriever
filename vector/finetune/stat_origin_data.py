import json
import os
from tqdm import tqdm
from pprint import pprint


path = 'data/datasets/zyznull_dureader-retrieval-ranking/train.jsonl'
'''统计结果 (这样的数据可得到 22230586 个最终 Dataset 样本, 7.82→42GB)
{'negative_passages': {'avg': 99.98839053185948,
                       'max': 100,
                       'min': 97,
                       'num': 86395,
                       'text': {'avg': 340.0095246893065,
                                'max': 122596,
                                'min': 1,
                                'num': 8638497}},
 'positive_passages': {'avg': 2.5741651715955784,
                       'max': 130,
                       'min': 1,
                       'num': 86395,
                       'text': {'avg': 358.5845590053733,
                                'max': 107567,
                                'min': 1,
                                'num': 222395}},
 'query': {'avg': 9.512610683488628, 'max': 56, 'min': 2, 'num': 86395}}
'''

path = 'data/datasets/zyznull_dureader-retrieval-ranking/dev.jsonl'
'''统计结果 (这样的数据可得到 279050 个最终 Dataset 样本, 87.5→540MB)
{'negative_passages': {'avg': 45.8445,
                       'max': 49,
                       'min': 28,
                       'num': 2000,
                       'text': {'avg': 322.51188255952184,
                                'max': 42467,
                                'min': 2,
                                'num': 91689}},
 'positive_passages': {'avg': 3.1555,
                       'max': 21,
                       'min': 0,
                       'num': 2000,
                       'text': {'avg': 409.3459039771827,
                                'max': 26006,
                                'min': 6,
                                'num': 6311}},
 'query': {'avg': 9.289, 'max': 55, 'min': 3, 'num': 2000}}
'''

def stat_path(path):
    stat = {
        'query': {
            'max': 0,
            'min': 10**10,
            'avg': 0,
            'num': 0,
        },
        'negative_passages': {
            'max': 0,
            'min': 10**10,
            'avg': 0,
            'num': 0,
            'text': {
                'max': 0,
                'min': 10**10,
                'avg': 0,
                'num': 0,
            }
        },
        'positive_passages': {
            'max': 0,
            'min': 10**10,
            'avg': 0,
            'num': 0,
            'text': {
                'max': 0,
                'min': 10**10,
                'avg': 0,
                'num': 0,
            }
        },
    }

    with open(path, 'r', encoding='utf8') as r:
        for line in tqdm(r):
            line = json.loads(line)
            
            q_len = len(line['query'])
            if q_len > stat['query']['max']:
                stat['query']['max'] = q_len
            if q_len < stat['query']['min']:
                stat['query']['min'] = q_len
            stat['query']['avg'] += q_len
            stat['query']['num'] += 1
            
            neg_n = len(line['negative_passages'])
            if neg_n > stat['negative_passages']['max']:
                stat['negative_passages']['max'] = neg_n
            if neg_n < stat['negative_passages']['min']:
                stat['negative_passages']['min'] = neg_n
            stat['negative_passages']['avg'] += neg_n
            stat['negative_passages']['num'] += 1
            
            pos_n = len(line['positive_passages'])
            if pos_n > stat['positive_passages']['max']:
                stat['positive_passages']['max'] = pos_n
            if pos_n < stat['positive_passages']['min']:
                stat['positive_passages']['min'] = pos_n
            stat['positive_passages']['avg'] += pos_n
            stat['positive_passages']['num'] += 1
            
            for doc in line['negative_passages']:
                _len = len(doc['text'])
                if _len > stat['negative_passages']['text']['max']:
                    stat['negative_passages']['text']['max'] = _len
                if _len < stat['negative_passages']['text']['min']:
                    stat['negative_passages']['text']['min'] = _len
                stat['negative_passages']['text']['avg'] += _len
                stat['negative_passages']['text']['num'] += 1
            
            for doc in line['positive_passages']:
                _len = len(doc['text'])
                if _len > stat['positive_passages']['text']['max']:
                    stat['positive_passages']['text']['max'] = _len
                if _len < stat['positive_passages']['text']['min']:
                    stat['positive_passages']['text']['min'] = _len
                stat['positive_passages']['text']['avg'] += _len
                stat['positive_passages']['text']['num'] += 1
                
    stat['query']['avg'] /= stat['query']['num']
    stat['negative_passages']['avg'] /= stat['negative_passages']['num']
    stat['positive_passages']['avg'] /= stat['positive_passages']['num']
    stat['negative_passages']['text']['avg'] /= stat['negative_passages']['text']['num']
    stat['positive_passages']['text']['avg'] /= stat['positive_passages']['text']['num']

    pprint(stat)
    with open(os.path.splitext(path)[0] + '_stat.json', 'w', encoding='utf8') as w:
        json.dump(stat, w, ensure_ascii=False, indent=2)
    return stat

for p in ['dev.jsonl', 'train.jsonl']:
    path = os.path.join('data/datasets/annotate100', p)
    stat_path(path)
