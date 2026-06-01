from copy import Error
import os
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
import logging
import time
from typing import Optional, Any
import threading

from tqdm import tqdm

from components.config import MyConfig
from components.utils import select_gpu_with_min_free


class LLMProxy:
    """
    LLM proxy that wraps langchain.init_chat_model.
    """

    def __init__(self, config: MyConfig):
        self.config = config
        self.max_retries = self.config.agent.max_retries
        self.backoff_factor = self.config.agent.backoff_factor
        self.llm = None
        # add a lock to protect access / reinitialization of self.llm
        self._llm_lock = threading.RLock()
        self.llm = self._init_chat_model()

    def _init_chat_model(self):
        source = self.config.agent.llm_model_source
        name = self.config.agent.llm_model_name
        temperature = float(self.config.agent.llm_temperature+1e-8)
        base_url = self.config.agent.llm_base_url
        # Allow passing extra model kwargs via config
        model_kwargs = {}
        model_kwargs["max_retries"] = self.max_retries

        if source == "huggingface" and name.lower().startswith("qwen"):
            if name == "Qwen/Qwen2.5-1.5B":
                name = "qwen2.5-1.5b-instruct"
            if name == "Qwen/Qwen2.5-3B":
                name = "qwen2.5-3b-instruct"
            return init_chat_model(
                model=name, model_provider="openai",
                temperature=temperature,
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        elif source == "huggingface":
            llm = HuggingFacePipeline.from_model_id(
                model_id=name,
                task="text-generation",
                device=select_gpu_with_min_free(14),
                pipeline_kwargs={
                    "temperature": temperature,
                    "return_full_text": False,
                    "do_sample": True,
                    "max_new_tokens": 2048,
                    "trust_remote_code": True,
                },
            )
            return ChatHuggingFace(llm=llm, **model_kwargs)
        elif source == "ollama":
            # Ollama models are hosted locally; no API key or base_url needed
            return init_chat_model(model=name, model_provider="ollama", temperature=temperature, **model_kwargs)
        else:
            # init_chat_model accepts a single `model` parameter such as "openai:gpt-4"
            return init_chat_model(model=name, model_provider=source, temperature=temperature, base_url=base_url, **model_kwargs)

    def set_llm_proxy(self):
        """(Re)initialize underlying llm and return it."""
        # serialize reinitialization so only one thread recreates the client at a time
        with self._llm_lock:
            self.llm = self._init_chat_model()
        return self

    def _extract_content(self, response: AIMessage | Any) -> str:
        # Try to extract textual content in a forgiving way.
        try:
            # Common pattern in repo: response may be dict-like or have .content
            if response is None:
                return ""
            if isinstance(response, str):
                return response
            if isinstance(response, dict):
                return response.get("content", response.get("text", str(response)))
            if hasattr(response, "content"):
                return getattr(response, "content")
            # Some langchain wrappers return objects with .generations or .message
            if hasattr(response, "generations"):
                gens = getattr(response, "generations")
                # gens could be a list of Generation objects
                try:
                    return str(gens[0][0].text)
                except Exception:
                    return str(gens)
            if hasattr(response, "text"):
                return getattr(response, "text")
            return str(response)
        except Exception:
            logging.exception("Failed to extract content from response: %s", response)
            raise ValueError("Failed to extract content from response")

    def ask(self, prompt, **kwargs) -> str:
        """Synchronous ask with optimistic concurrency: call without lock; on error reinit under lock and retry."""
        
        # 如果是huggingface模型，设置system提示词
        if self.config.agent.llm_model_source == "huggingface":
            system_prompt = "You are a helpful assistant. Answer the question based on the context provided or summarize the information as needed."
            if isinstance(prompt, list):
                prompt = [{"role": "system", "content": system_prompt}] + prompt
            else:
                prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        
        attempt = 0
        while attempt <= self.max_retries:
            try:
                # ensure llm exists, but avoid holding lock during invoke if possible
                if self.llm is None:
                    with self._llm_lock:
                        if self.llm is None:
                            self.set_llm_proxy()
                llm_local = self.llm  # snapshot reference # ?

                # call without holding the lock to allow concurrency on the common success path
                if hasattr(llm_local, "invoke"):
                    resp = llm_local.invoke(prompt, **kwargs)
                else:
                    resp = llm_local(prompt, **kwargs)
                # 不extract_content 直接返回原始响应,从而计算token数
                # return self._extract_content(resp).strip()
                return resp
            except Exception as e:
                attempt += 1
                logging.exception("LLM call failed (attempt %d): %s", attempt, e)
                # serialize reinitialization so only one thread recreates the client at a time
                try:
                    with self._llm_lock:
                        # always try to (re)create a fresh llm instance
                        self.set_llm_proxy()
                except Exception:
                    logging.exception("LLM reinitialization failed after exception: %s", e)
                if attempt > self.max_retries and self.max_retries >= 0:
                    tqdm.write(f"\nLLM ask failed after {self.max_retries} attempts\n")
                    raise e
                backoff = self.backoff_factor * (2 ** (attempt - 1))
                time.sleep(backoff)
        

    # convenience aliases
    def invoke(self, prompt: str, **kwargs) -> str:
        return self.ask(prompt, **kwargs)

