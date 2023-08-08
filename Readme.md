# Retriever 训练


## 环境依赖
`pip install -r requirements.txt`

## 数据集转换

/vector/finetune/dataset.py

jsonl格式的训练数据放于/data/datasets下，分别命名为train.jsonl和dev.jsonl

Line51 将dataset_path 修改为jsonl格式数据集所在目录

- 数据集转换 `python vector/finetune/dataset.py`

## 模型训练
/vector/finetune/trainer_contriever.py

预训练模型可放于/data/model/contriever/ckpt下

Line37 q_model_path 修改为预训练的question_encoder所在目录

Line38 ref_model_path 修改为预训练的reference_encoder所在目录

Line43 dataset_path修改为转换后的数据集所在目录


- 训练 `python -m torch.distributed.launch --nproc_per_node=8 vector/finetune/trainer_contriever.py`


## 预训练模型
模型地址：https://www.aliyundrive.com/s/kAxk4cdLpAB
