from __future__ import annotations

import os
import csv
from multiprocessing import Queue
import pickle
from tqdm import tqdm
from agent.querylink import QueryLinkAgent
from components.config import MyConfig
from components.llm import LLMProxy
from components.types import QA, DataItem, DataSet, History, ResultItem, Results
from typing import Annotated, Any, Dict, List, Tuple, override

from components.memory import MyDocument, MyMemory, DocFilter
from components.embeddings import MyMultiLevelEmbeddings
from agent.querylink import InvokeGraph, InvokeState

class QueryLinkMotivation(QueryLinkAgent):
    def __init__(self, config: MyConfig, msg_queue: Queue, res_queue: Queue):
        super().__init__(config, msg_queue, res_queue)
        self.ebd = MyMultiLevelEmbeddings(self.config)
        self.llm = LLMProxy(self.config)
        
    @override
    def init(self, dataItem: DataItem):
        self.tid: str = dataItem.tid
        self.qaList: List[QA] = dataItem.qaList
        
        # NOTE DEBUG 取一部分数据
        # tempList = []
        # for qa in self.qaList:
        #     if qa.questionID == "conv-44_q56":
        #         tempList.append(qa)
        # self.qaList = tempList
        
        self.memory = MyMemory(
            config=self.config,
            tid=self.tid,
            msg_queue=self.msg_queue,
            embedding=self.ebd,
            llm=self.llm,
            history=dataItem.history
            )
        self.invoke_graph = MotivationInvokeGraph(self.llm, self.memory)
        self._build_memory()
    
    @override
    def _evaluate_item(self, qa: QA, response: InvokeState) -> ResultItem:
        # # NOTE DEBUG 暂时不做自动评测
        judgment = "Not Judged"
        
        # sim_res = {"retrieved_side": {}, "doc_type_side": {}}
        sim_res = response.get("sim_res", {})
        flat_sim_res = {}
        for retrieved_by in self.all_retrieved_bys:
            for doc_type in self.all_types:
                key_score = f"sim_{retrieved_by}_{doc_type}_score"
                key_count = f"sim_{retrieved_by}_{doc_type}_count"
                if retrieved_by in sim_res and doc_type in sim_res[retrieved_by]:
                    score, count = sim_res[retrieved_by][doc_type]
                    flat_sim_res[key_score] = score
                    flat_sim_res[key_count] = count
                else:
                    flat_sim_res[key_score] = 0.0
                    flat_sim_res[key_count] = 0
                    
        # 统计各个retrieved_by总的平均值
        for retrieved_by in self.all_retrieved_bys:
            key_score = f"sim_{retrieved_by}_all_score"
            key_count = f"sim_{retrieved_by}_all_count"
            total_score = 0.0
            total_count = 0
            for doc_type in self.all_types:
                if retrieved_by in sim_res and doc_type in sim_res[retrieved_by]:
                    score, count = sim_res[retrieved_by][doc_type]
                    total_score += score
                    total_count += count
            flat_sim_res[key_score] = total_score
            flat_sim_res[key_count] = total_count
        
        # 统计各个类别总的平均值
        for doc_type in self.all_types:
            key_score = f"sim_all_{doc_type}_score"
            key_count = f"sim_all_{doc_type}_count"
            total_score = 0.0
            total_count = 0
            for retrieved_by in self.all_retrieved_bys:
                if retrieved_by in sim_res and doc_type in sim_res[retrieved_by]:
                    score, count = sim_res[retrieved_by][doc_type]
                    total_score += score
                    total_count += count
            flat_sim_res[key_score] = total_score
            flat_sim_res[key_count] = total_count
        
        result = ResultItem(
            tid=self.tid,
            questionID=qa.questionID,
            question=qa.question,
            questionType=qa.questionType,
            answer=qa.answer,
            response=response["answer"],
            judgment=judgment,
            additional_info={
                # flatten sim_res into additional_info
                **flat_sim_res
            }
        )
        return result
    
    @override
    @classmethod
    def evaluate_all(cls, config: MyConfig, **kwargs):
        data_path = config.exp.data_path
        file_name = config.exp.data_file_name
        full_path = os.path.join(data_path, file_name)
        full_path_without_ext = os.path.splitext(full_path)[0]
        if os.path.exists(full_path_without_ext + ".pkl"):
            with open(full_path_without_ext + ".pkl", 'rb') as f:
                data: DataSet = pickle.load(f)
        else:
            raise ValueError("DataSet pkl file does not exist for evaluation.")
        
        # 读取结果文件
        result_path = os.path.join(config.exp.output_path, 'results.csv')
        if not os.path.exists(result_path):
            tqdm.write(f"Results file {result_path} does not exist.")
            return
        # 读取列名及内容
        results = []
        with open(result_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            with tqdm(total=0, desc="Evaluating results", ascii=".>") as pbar:
                for row in reader:
                    pbar.total += 1
                    pbar.update(1)
                    results.append(row)
        
        sim_col_names = [col for col in results[0].keys() if col.startswith("sim_")]
        summary: Dict[str, Dict[str, float]] = {}
        data.questionTypeList.sort()
        for qtype in data.questionTypeList + ["avg"]:
            summary[str(qtype)] = {
                "questionType": str(qtype),
                "count": 0,
            }
            for col in sim_col_names:
                summary[str(qtype)][col] = 0.0
                
        for row in results:
            qtype = row["questionType"]
            if qtype not in summary:
                continue
            summary[qtype]["count"] += 1
            for col in sim_col_names:
                summary[qtype][col] += float(row[col])
                summary["avg"][col] += float(row[col])
            summary["avg"]["count"] += 1
            
        for qtype in summary.keys():
            for col in sim_col_names:
                if col.endswith("_count"):
                    continue
                count_key = col.replace("_score", "_count")
                avg_key = col.replace("_score", "_avg")
                if summary[qtype][count_key] > 0:
                    summary[qtype][avg_key] = summary[qtype][col] / summary[qtype][count_key]
            
        # 写入summary文件
        sum_results_path = os.path.join(config.exp.output_path, 'summary.csv')
        with open(sum_results_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = summary["avg"].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for qtype, metrics in summary.items():
                writer.writerow(metrics)
        

# ======================================================

class MotivationInvokeState(InvokeState):
    sim_res: Dict[str, Dict[str, Tuple[float, int]]] | Any
    
class MotivationInvokeGraph(InvokeGraph):
    @override
    def _retrieve(self, state: MotivationInvokeState):
        # 添加问题本身作为检索依据
        question = state["qa"].question
        state["retrieve_base"]["question"] = [question]
        
        mix_v = []
        for key, value in state["retrieve_base"].items():
            mix_v.extend(value)
        state["retrieve_base"]["mix"] = mix_v
        
        qa: QA = state["qa"]
        ecids = qa.evidenceCIDs
        def filter_func(doc: MyDocument) -> bool:
            return doc.metadata.get("contentID", None) in ecids
        
        # 计算retrieved_by和type交叉检索的平均相似度
        sim_res = {}
        for key, value in state["retrieve_base"].items():
            retrieved_by = key
            for v in value:
                if isinstance(v, str):
                    query_v = [v]
                else:
                    query_v = v
                query_results: List[Tuple[MyDocument, float]] = self.memory.get_by_filter_with_score(filter=filter_func, query=query_v)
                for doc, score in query_results:
                    doc_type = list(doc.metadata.get("type"))[0]
                    # retrieved_side
                    if retrieved_by not in sim_res:
                        sim_res[retrieved_by] = {}
                    if doc_type not in sim_res[retrieved_by]:
                        sim_res[retrieved_by][doc_type] = [score, 1]
                    else:
                        sim_res[retrieved_by][doc_type][0] += score
                        sim_res[retrieved_by][doc_type][1] += 1
        return {"sim_res": sim_res}
    
    @override
    def _generate_answer(self, state: MotivationInvokeState):
        return {"answer": ""}