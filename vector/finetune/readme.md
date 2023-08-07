## Run
- 标注数据转换 `python vector/finetune/annotate_to_jsonl.py`
- 数据结果统计 `python vector/finetune/stat_origin_data.py`
- 数据集转换 `python vector/finetune/dataset.py`
- 训练 `python -m torch.distributed.launch --nproc_per_node=8 vector/finetune/trainer_contriever.py`
