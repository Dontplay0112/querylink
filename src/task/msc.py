# import os
# import json
# import pickle
# import csv
# from typing import Dict, List
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from tqdm import tqdm
# from agent.baseagent import BaseAgent
# from task.basetask import BaseTaskWorker, BaseTask
# from components.evaluation import calculate_metrics
# from components.types import QA, DataItem, DataSet, History, HistoryItem, ResultItem, Results
# from datasets import load_dataset
# from components.config import Config


# class MSC(BaseTask):


# def __init__(self, config: Config):
#         super().__init__(config, BaseTaskWorker)
    
#     def _load_data(self, full_path: str) -> DataSet:
#         ds = load_dataset("nayohan/multi_session_chat")
#         print(ds)
#         # 保存
#         dataset = DataSet(item=[])
            
#         return dataset
    
    
if __name__ == "__main__":
    from datasets import load_dataset
    
    ds = load_dataset("nayohan/multi_session_chat")
    print(ds)