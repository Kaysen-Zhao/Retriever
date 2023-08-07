import sys, os
sys.path.append('vector/contriever')
sys.path.append(os.getcwd())

from typing import List, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Extra
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, status, HTTPException
import time

from utiles import get_global_config, torch_gc
from src.contriever import Contriever
from transformers import AutoTokenizer


# 初始化 contriever 模型
device = get_global_config().get(['vector', 'device'])  # cuda cpu
methods_models = {}  # 直接调用的模型
path_models = {}
path_tokenizers = {}
for method, paras in get_global_config().get(['vector', 'methods']).items():
    if not paras.get('load', False):
        continue
    path = paras['model_path']['question_encoder']
    path_ref = paras['model_path']['reference_encoder']
    # 加载模型
    for p in [path, path_ref]:
        if p not in path_models:
            path_models[p] = Contriever.from_pretrained(p).to(device)
        if p not in path_tokenizers:
            path_tokenizers[p] = AutoTokenizer.from_pretrained(p)
    # 具体的模型
    model = path_models[path]
    model_ref = path_models[path_ref]
    tokenizer = path_tokenizers[path]
    tokenizer_ref = path_tokenizers[path_ref]
    # 直接调用的模型
    methods_models[method] = {
        'question_encoder': lambda text: model(**tokenizer(
            text, padding=True, truncation=True, return_tensors="pt").to(device)),
        'reference_encoder': lambda text: model_ref(**tokenizer_ref(
            text, padding=True, truncation=True, return_tensors="pt").to(device)),
    }
    print(f'初始化 {method} 方法和模型成功!')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class Body(BaseModel):
    method: Optional[str] = 'contriever'
    model: Optional[str] = 'reference_encoder'
    texts: Optional[List[str]] = []


@app.post(get_global_config()['vector']['api']['path'])
async def completions(body: Body, request: Request):
    if request.headers.get("Authorization").split(" ")[1] != get_global_config()['vector']['api']['token']:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")
    
    if body.method not in methods_models:
        raise HTTPException(status.HTTP_405_METHOD_NOT_ALLOWED, "method 不存在!")
    
    if body.model not in methods_models[body.method]:
        raise HTTPException(status.HTTP_405_METHOD_NOT_ALLOWED, "model 不存在!")
    
    ret = {
        "method": body.method,  # 使用的方法
        "model": body.model,  # 使用的模型
        "similarity_method": "dotProduct",  # 相似度计算方式，例如点乘dotProduct、余弦相似度cosineSimilarity
        "dim": 768,  # 每个向量的维度
        "time_consuming": 0,  # 耗时多少秒
        "vectors": [],
        'success': True,
    }
    if body.texts:
        start_time = time.time()
        torch_gc()
        vectors = methods_models[body.method][body.model](body.texts)
        ret['vectors'] = vectors.tolist()
        ret['dim'] = len(vectors[0])
        ret['time_consuming'] = time.time() - start_time
    return JSONResponse(content=ret)


if __name__ == '__main__':
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host=get_global_config()['vector']['api']['host'], 
                     port=get_global_config()['vector']['api']['port'],
                     log_config=log_config)


'''测试
curl -vvv http://127.0.0.1:37018/v1/get_text_vector \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 8aGfJp0UMfQe2ZVSJQloxpqrS" \
  -d '{ "texts": ["content", "写一首夏天的诗"],
    "model": "question_encoder",
    "method": "安徽大学"
  }'
'''
