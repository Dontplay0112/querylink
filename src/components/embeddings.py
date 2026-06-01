import logging
import multiprocessing
import os
import subprocess
import threading
from time import sleep
from langchain_core.runnables.config import run_in_executor
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from components.config import MyConfig
from components.utils import select_gpu_with_min_free

_gpu_lock = multiprocessing.Lock()
class MyMultiLevelEmbeddings(Embeddings):
    def __init__(self, config: MyConfig, method: str = "method1"):
        if config.agent.embed_model_source == "openai":
            self.ebd = OpenAIEmbeddings(base_url=config.agent.embed_base_url)
        elif config.agent.embed_model_source == "ollama":
            model = config.agent.embed_model_name
            self.ebd = OllamaEmbeddings(model=model, base_url=config.agent.embed_base_url)
        elif config.agent.embed_model_source == "huggingface":
            model = config.agent.embed_model_name
            with _gpu_lock:
                try:
                    gpu_idx = select_gpu_with_min_free(min_free_gb=2)
                    logging.info(f"Selected GPU {gpu_idx} for HuggingFace embeddings (CUDA_VISIBLE_DEVICES={gpu_idx})")
                except Exception as e:
                    raise RuntimeError(f"无法分配 GPU 给 HuggingFace embeddings: {e}")
                self.ebd = HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": f"cuda:{gpu_idx}"})
                # self.ebd.multi_process = True
        self.method = method

    def embed_documents(self, texts: list[list[str]] | list[str]) -> list[list[float]]:
        if isinstance(texts[0], str):
            texts = [texts]
        length = len(texts)
        # embeds = self.ebd.embed_documents([text for sublist in texts for text in sublist])
        texts = [text for sublist in texts for text in sublist]
        embeds = []
        for t in texts:
            embeds.append(self.ebd.embed_query(t))
        text_num = int(len(embeds) / length)
        embeds = [embeds[i * text_num:(i + 1) * text_num] for i in range(length)]
        return self.get_method(self.method)(embeds)

    def embed_query(self, text: list[str] | str) -> list[float]:
        if isinstance(text, str):
            text = [text]
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[list[str]] | list[str]) -> list[list[float]]:
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: list[str] | str) -> list[float]:
        return await run_in_executor(None, self.embed_query, text)


    def method1(self, embeds: list[list[float]], kwargs: dict = {}):
        for ebds in embeds:
            for ebd in ebds[1:]:
                for i in range(len(ebds[0])):
                    ebds[0][i] += ebd[i]
        # 对每一行求向量均值
        return [[e / len(ebds) for e in ebds[0]] for ebds in embeds]

    def method2(self, embeds: list[list[float]], kwargs: dict = {}):
        # 如果有权重矩阵在kwargs中，可以用权重矩阵加权平均
        weights = kwargs.get("weights", [1] * len(embeds))
        weighted_embeds = []
        for ebds, w in zip(embeds, weights):
            weighted_ebd = []
            for i in range(len(ebds[0])):
                val = 0.0
                for ebd in ebds:
                    val += ebd[i] * w
                weighted_ebd.append(val / (len(ebds) * w))
            weighted_embeds.append(weighted_ebd)
        return weighted_embeds
    def get_method(self, method: str):
        if method == "method1":
            return self.method1
        elif method == "method2":
            return self.method2
        else:
            raise ValueError(f"Unknown method: {method}")
