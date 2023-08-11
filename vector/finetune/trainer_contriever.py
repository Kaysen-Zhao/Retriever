from transformers import RobertaTokenizer, RobertaModel, AutoModelWithLMHead, AutoTokenizer, Trainer, AutoModel, BertLMHeadModel, models
from datasets.load import load_dataset, load_from_disk
import torch
import os
import sys
import time
import random
import json
from rouge_score.rouge_scorer import RougeScorer
import logging
import datetime
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sys, os
sys.path.append('vector/contriever')
sys.path.append(os.getcwd())

from src.contriever import Contriever

# 测试集评估(bs=24*8) Acc:   0.710287 , Loss:   0.799917 , Sample: 38400
# 首轮训练 Step   50/578925, Avg Loss:   0.372914, Lr: 1.692807e-07
# q_model_path = "data/model/contriever/shuimu/question_encoder"
# ref_model_path = "data/model/contriever/shuimu/reference_encoder"
# q_model_path = "data/ckpt/ckpt_contriever/shuimu/step-3000-acc-0.8111-loss-0.5985/question_encoder"
# ref_model_path = "data/ckpt/ckpt_contriever/shuimu/step-3000-acc-0.8111-loss-0.5985/reference_encoder"

# 测试集评估(bs=24*8) Acc:   0.628255 , Loss:   1.204921 , Sample: 38400
# 首轮训练 Step   50/578925, Avg Loss:   1.316492, Lr: 1.692807e-07
# q_model_path = ref_model_path = 'data/model/contriever/mcontriever'

# 测试集评估(bs=24*8) Acc: 0.667318 , Loss: 1.711564 , Sample: 38400
# 首轮训练 Step   50/578925, Avg Loss:   1.186070, Lr: 1.692807e-07
q_model_path = 'data/model/contriever/ckpt/question_encoder'
ref_model_path = 'data/model/contriever/ckpt/reference_encoder'

# https://huggingface.co/datasets/zyznull/dureader-retrieval-ranking
# dataset_path = "data/datasets/zyznull_dureader-retrieval-ranking"
# dataset_path = "data/datasets/annotate100"
dataset_path = "./data/dataset5"
share_model = True
batch_size = 8
pretrain_tokenizer = q_model_path
load_checkpoint = '' # optional/data4t/zkx/uni_glm/data/ckpt/ckpt_contriever/20230711-170253/step-1500-acc-0.9650-loss-0.0995
save_checkpoint = "./data/ckpt/ckpt_contriever/%s" % (time.strftime("%Y%m%d-%H%M%S", time.localtime(time.time())))
max_epoch = 5
seed = 42
if q_model_path != ref_model_path:
    share_model = False


def get_logging(logging_name='contriever.log'):
    logger = logging.getLogger("train-contriever")
    os.makedirs(save_checkpoint, exist_ok=True)
    logging.basicConfig(filename=os.path.join(save_checkpoint, logging_name), filemode='a')
    logger.setLevel(logging.INFO)
    rf_handler = logging.StreamHandler(sys.stderr)  # 默认是sys.stderr
    rf_handler.setLevel(logging.DEBUG)
    #rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))

    f_handler = logging.FileHandler('error.log')
    f_handler.setLevel(logging.ERROR)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
    return logger

logger = get_logging()
logger.info('Para: ' + str({
    'batch_size': batch_size,
    'seed': seed,
    'share_model': share_model,
    'q_model_path': q_model_path,
    'ref_model_path': ref_model_path,
    'dataset_path': dataset_path,
    'load_checkpoint': load_checkpoint,
}))


class Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.tensor):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather(x: torch.tensor):
    if not dist.is_initialized():
        return x
    x_gather = Gather.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather


class QuestionReferenceDensity_fromContriever(torch.nn.Module):
    def __init__(self, q_model_path, ref_model_path) -> None:
        super().__init__()
        self.question_encoder = Contriever.from_pretrained(q_model_path)
        if share_model:
            self.reference_encoder = self.question_encoder
        else:
            self.reference_encoder = Contriever.from_pretrained(ref_model_path)
        self.share_model = share_model

        total = sum([param.nelement() for param in self.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def forward(self, question, pos, neg, return_acc=False):
        global temp, device

        cls_q = self.question_encoder(**question)
        cls_r_pos = self.reference_encoder(**pos)
        cls_r_neg = self.reference_encoder(**neg)
        cls_q /= temp

        bsz = cls_q.shape[0]

        kemb = torch.cat([cls_r_pos, cls_r_neg])  # [bs * 2, emb_dim]
        gather_kemb = gather(kemb)  # [bs * 2 * word_size, emb_dim]

        # 相当于把负例数量扩展到 bs*2*word_siz-1 个, 但可能有小概率其他样本的正负例中有真正例
        scores = torch.matmul(cls_q, torch.transpose(gather_kemb, 0, 1))  # [bs, bs * 2 * word_size]

        labels = torch.arange(0, bsz, dtype=torch.long, device=device)
        labels = labels + dist.get_rank() * len(kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels)
        
        if return_acc:
            _, preds = torch.max(scores, dim=1)
            acc = torch.mean((preds == labels).float())
            return loss, acc

        return loss


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )

#todo

train_set = load_from_disk(dataset_path + "/train").shuffle(seed=seed)
dev_set = load_from_disk(dataset_path + "/dev").shuffle(seed=seed)
tokenizer = AutoTokenizer.from_pretrained(pretrain_tokenizer)


def collate(data):
    question = tokenizer([item["question"] for item in data], return_tensors="pt", padding=True, truncation=True)
    positive_reference = tokenizer([item["positive_reference"] for item in data],
                                   return_tensors="pt", padding=True, truncation=True)
    negative_reference = tokenizer([item["negative_reference"] for item in data],
                                   return_tensors="pt", padding=True, truncation=True)

    for key in question:
        question[key] = question[key].to(device)
    for key in positive_reference:
        positive_reference[key] = positive_reference[key].to(device)
    for key in negative_reference:
        negative_reference[key] = negative_reference[key].to(device)

    return question, positive_reference, negative_reference

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

torch.distributed.init_process_group(backend="nccl",world_size=1,rank=0)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate, sampler=sampler)
sampler = DistributedSampler(dev_set)
dev_loader = DataLoader(dev_set, batch_size=batch_size, collate_fn=collate, sampler=sampler)
total_step = len(train_loader) * max_epoch

model = QuestionReferenceDensity_fromContriever(q_model_path, ref_model_path)

model = model.to(device)
if load_checkpoint:
    ckpt = torch.load(load_checkpoint)
    model.load_state_dict({key.replace('module.', ''): ckpt[key] for key in ckpt.keys()})

model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
opt = AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
scheduler_args = {
    "warmup": int(total_step / 1000 * 25),  # 2.5% warm-up
    "total": total_step,
    "ratio": 0.0,
}
scheduler = WarmupLinearScheduler(opt, **scheduler_args)
temp = 0.05


def eval(eval_steps=-1):
    model.eval()
    total_acc = 0
    total_loss = 0
    with torch.no_grad():
        n = eval_steps * batch_size * dist.get_world_size()
        if eval_steps != -1 and n < len(dev_set):
            indices = random.sample(range(len(dev_set)), n)
            dev_subset = Subset(dev_set, indices)
            sampler = DistributedSampler(dev_subset)
            dev_loader_ = DataLoader(dev_subset, batch_size=batch_size, collate_fn=collate, sampler=sampler)
        else:
            dev_loader_ = dev_loader
        total_step = len(dev_loader_)
        if dist.get_rank() == 0:
            bar = tqdm(dev_loader_, 'EVAL')
        else:
            bar = dev_loader_
        for q, pos, neg in bar:
            loss, acc = model(q, pos, neg, return_acc=True)
            dist.all_reduce(loss)
            loss /= dist.get_world_size()
            dist.all_reduce(acc)
            acc /= dist.get_world_size()
            total_acc += acc
            total_loss += loss
        total_acc /= total_step
        total_loss /= total_step
        
        if dist.get_rank() == 0:
            logger.info("EVALUATION, Acc: %10.6f , Loss: %10.6f , Sample: %4d" % (total_acc, total_loss, total_step * batch_size * dist.get_world_size()))
    return total_acc, total_loss

    
def save_coder(model: QuestionReferenceDensity_fromContriever, path: str, tokenizer: models.bert.tokenization_bert_fast.BertTokenizerFast):
    while not hasattr(model, 'question_encoder'):
        model = model.module
    model.question_encoder.save_pretrained(os.path.join(path, 'question_encoder'))
    tokenizer.save_pretrained(os.path.join(path, 'question_encoder'))
    if not share_model:
        model.reference_encoder.save_pretrained(os.path.join(path, 'reference_encoder'))
        tokenizer.save_pretrained(os.path.join(path, 'reference_encoder'))


def save(name):
    os.makedirs(save_checkpoint, exist_ok=True)
    path = os.path.join(save_checkpoint, name)
    # torch.save(model.state_dict(), path + '.ckpt')
    save_coder(model, path, tokenizer)
    logger.info("Sussessfully saved " + "%s.ckpt" % (name))


def train(max_epoch=5, eval_step=300, start_save_step=500, print_step=50):
    step = 0
    total_step = len(train_loader) * max_epoch
    all_loss = 0
    max_dev_acc = 0
    for epoch in range(0, max_epoch):
        logger.info("EPOCH %d" % epoch)
        for q, pos, neg in train_loader:
            if step % eval_step == 0:
                acc, loss = eval(eval_steps = 200)
                if start_save_step <= step and acc > max_dev_acc and dist.get_rank() == 0:
                    save("step-%d-acc-%.4f-loss-%.4f" % (step, acc, loss))
                    max_dev_acc = acc
            if step % len(train_loader) == 0:
                save("Epoch-%d-step-%d" % (epoch, step))
                
            model.train()
            step += 1

            loss = model(q, pos, neg)
            reduced_loss = loss.detach().clone().view(1)
            dist.all_reduce(reduced_loss.data)
            all_loss += reduced_loss.data / dist.get_world_size()

            if step % print_step == 0 and dist.get_rank() == 0:
                logger.info("Step %4d/%4d, Avg Loss: %10.6f, Lr: %10.6e" % (step, total_step, all_loss/step, scheduler.get_lr()[0]))

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            model.zero_grad()
    save("End")

if __name__ == "__main__":
    train(max_epoch=max_epoch)
