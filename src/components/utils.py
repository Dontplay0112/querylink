from contextlib import nullcontext
import json
import logging
import multiprocessing
import pickle
import os
import csv
import threading
import threading
from typing import Any, List
import subprocess
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from tqdm import tqdm

from components.config import MyConfig
from components.types import ResultItem, Results

temp_result_list = []
def append_result_to_csv(file_path: str, results: Results | ResultItem):
    global temp_result_list
    if isinstance(results, ResultItem):
        results = Results(items=[results])
    results_list = []
    for result in results.items:
        res = {
            "tid": result.tid,
            "questionID": result.questionID,
            "question": result.question,
            "questionType": result.questionType,
            "answer": result.answer,
            "response": result.response,
            "judgment": result.judgment,
            **result.additional_info
        }
        results_list.append(res)
    try:
        file_exists = os.path.isfile(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile: 
            results_list.extend(temp_result_list)
            temp_result_list.clear()
            writer = csv.DictWriter(csvfile, fieldnames=results_list[0].keys())
            if not file_exists:
                writer.writeheader()
            for result in results_list:
                writer.writerow(result)
    except Exception:
        temp_result_list.extend(results_list)
        logging.exception(f"Failed to write results to CSV at {file_path}. Results have been cached in memory.")

def get_msg_content(response: AIMessage | Any) -> str:
        # Try to extract textual content in a forgiving way.
        res = ""
        try:
            # Common pattern in repo: response may be dict-like or have .content
            if response is None:
                res = ""
            if isinstance(response, str):
                res = response
            if isinstance(response, dict):
                res = response.get("content", response.get("text", str(response)))
            if hasattr(response, "content"):
                res = getattr(response, "content")
            # Some langchain wrappers return objects with .generations or .message
            if hasattr(response, "generations"):
                gens = getattr(response, "generations")
                # gens could be a list of Generation objects
                try:
                    res = str(gens[0][0].text)
                except Exception:
                    res = str(gens)
            if hasattr(response, "text"):
                res = getattr(response, "text")
            return str(res).strip()
        except Exception:
            raise ValueError("Failed to extract content from response")
        
def get_msg_token(response: AIMessage | Any):
    # usage = response.response_metadata["token_usage"]
    if hasattr(response, "response_metadata"):
        usage = response.response_metadata.get("token_usage", {"completion_tokens":0, "prompt_tokens":0, "total_tokens":0})
    else:
        return 0, 0, 0
    return usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0), usage.get("total_tokens", 0)
    # return response.usage_metadata.get("input_tokens", 0), response.usage_metadata.get("output_tokens", 0), response.usage_metadata.get("total_tokens", 0)

    
def hit_evidence(contentID: str, evidenceCIDs: List[str]) -> bool:
    for ecid in evidenceCIDs:
        if ecid.endswith("*"):
            if contentID.startswith(ecid[:-1]):
                return True
        else:
            if contentID == ecid:
                return True
    return False
        
def judge_answer(question:str, answer: str, ground_truth: str, config: MyConfig, llm = None) -> str:
    if llm is None:
        llm = init_chat_model(
            model='gpt-4o',
            model_provider="openai",
            temperature=0.0,
            base_url=config.agent.llm_base_url,
        )
    # Adapted from the Mem0 long-term-memory evaluation prompt.
    # License and attribution: THIRD_PARTY_NOTICES.md
    PROMPT = f"Your task is to label an answer to a question as \"CORRECT\" or \"WRONG\". You will be given the following data: (1) a question (posed by one user to another user), (2) a 'gold' (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.  The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example: Question: Do you remember what I got the last time I went to Hawaii? Gold answer: A shell necklace The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.  For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like 'last Tuesday' or 'next month'), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it CORRECT if it's the same date.  Now it's time for the real question: Question: {question} Gold answer: {ground_truth} Generated answer: {answer}  First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.  Just return the label CORRECT or WRONG in a json format with the key as \"label\"."
    
    response = llm.invoke(PROMPT)
    res = get_msg_content(response)
    return "WRONG" if "WRONG" in res.upper() else "CORRECT"

# 互斥锁
_gpu_lock = multiprocessing.Lock()
# _gpu_lock = nullcontext()
def select_gpu_with_min_free(min_free_gb: int = 10) -> int:
    """Return the GPU index which has >= min_free_gb free memory (MB). Raise if none."""
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    gpu_list = None
    if CUDA_VISIBLE_DEVICES is not None:
        gpu_list = [int(idx) for idx in CUDA_VISIBLE_DEVICES.split(",") if idx.strip().isdigit()]
    
    with _gpu_lock:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free,index", "--format=csv,noheader,nounits"],
                encoding="utf-8",
                stderr=subprocess.STDOUT,
            )
            lines = [ln.strip() for ln in out.strip().splitlines() if ln.strip()]
            best_idx = None
            best_free = -1
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                free_mb = int(parts[0])
                idx = int(parts[1])
                if (gpu_list is None or idx in gpu_list) and free_mb > best_free and free_mb >= min_free_gb * 1024:
                    best_free = free_mb
                    best_idx = idx
            if best_idx is None:
                raise RuntimeError(f"No GPU with >= {min_free_gb}GB free. Found: {lines}")
            
            # 根据 CUDA_VISIBLE_DEVICES 环境变量重新映射 GPU 索引
            if gpu_list is not None:
                best_idx_idx = gpu_list.index(best_idx)
            else:
                best_idx_idx = best_idx
            
            return best_idx_idx
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"nvidia-smi query failed: {e.output}") from e
        except FileNotFoundError:
            raise RuntimeError("nvidia-smi not found; cannot select GPU for HuggingFaceEmbeddings")
