from datasets import load_dataset, load_from_disk, Dataset
import json
from tqdm import tqdm


def generate_data(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf8') as r:
        for line in r:
            line = json.loads(line)
            for pos in line['positive_passages']:
                pos = pos['text']
                for neg in line['negative_passages']:
                    neg = neg['text']
                    yield {
                        'question': line['query'],
                        'positive_reference': pos,
                        'negative_reference': neg,
                    }

def jsonl_to_ds(jsonl_path, out_path):
    # 创建三个列表用于存储数据
    questions = []
    positive_references = []
    negative_references = []

    # 假设你想生成1000个样本
    for sample in tqdm(generate_data(jsonl_path), '解析样本'):
        print(sample)
        questions.append(sample["question"])
        positive_references.append(sample["positive_reference"])
        negative_references.append(sample["negative_reference"])

    # 一次性创建数据集
    data = Dataset.from_dict({
        "question": questions,
        "positive_reference": positive_references,
        "negative_reference": negative_references
    })

    print(data[0])

    # 将数据集保存到磁盘
    data.save_to_disk(out_path)
    print('完成:', out_path)
    print()


# dataset_path = "data/datasets/zyznull_dureader-retrieval-ranking"
# # dev 129349 train 1196898
# dataset_path = "data/datasets/annotate100"
dataset_path = "/data4t/zkx/uni_glm/data/dataset5"
jsonl_path = dataset_path + '/dev.jsonl'
out_path = dataset_path + "/dev"
jsonl_to_ds(jsonl_path, out_path)

jsonl_path = dataset_path + '/train.jsonl'
out_path = dataset_path + "/train"
jsonl_to_ds(jsonl_path, out_path)
