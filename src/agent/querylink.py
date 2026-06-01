from __future__ import annotations

import os
import csv
import gc
import pickle
import random
import shutil
from pathlib import Path
from operator import add, or_
from multiprocessing import Queue
from typing import Annotated, Any, Dict, List, Tuple, override
from typing_extensions import TypedDict

import torch
from tqdm import tqdm
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from components.config import MyConfig
from components.llm import LLMProxy
from components.types import QA, DataItem, DataSet, ResultItem, Results
from components.evaluation import calculate_metrics
from components.utils import append_result_to_csv, get_msg_content, get_msg_token, judge_answer, hit_evidence
from components.memory import MyDocument, MyMemory
from components.embeddings import MyMultiLevelEmbeddings
from agent.baseagent import BaseAgent, ThreadTaskRunner


class QueryLinkAgent(BaseAgent):
    def __init__(self, config: MyConfig, msg_queue: Queue, res_queue: Queue):
        super().__init__(config, msg_queue, res_queue)
        self.ebd = MyMultiLevelEmbeddings(self.config)
        self.llm = LLMProxy(self.config)
        
    @override
    def init(self, dataItem: DataItem):
        self.tid: str = dataItem.tid
        self.qaList: List[QA] = dataItem.qaList
        
        self.memory = MyMemory(
            config=self.config,
            tid=self.tid,
            msg_queue=self.msg_queue,
            embedding=self.ebd,
            llm=self.llm,
            history=dataItem.history
            )
        if self.config.agent.agent == "QueryLinkLocomo":
            self.invoke_graph = LocomoInvokeGraph(self.llm, self.memory)
        else:
            self.invoke_graph = InvokeGraph(self.llm, self.memory)
        self._build_memory()
        
    @override
    def solve(self):
        ThreadTaskRunner(
            tid=self.tid,
            stage="qa",
            data_items=self.qaList,
            func=self._invoke_item,
            msg_queue=self.msg_queue,
            res_queue=self.res_queue
        ).run(max_workers=self.config.exp.max_thread_workers, max_inflight=self.config.exp.max_thread_inflight)
        
    @override
    @classmethod
    def res_process(cls, res: Results, config: MyConfig):
        append_result_to_csv(
            f"{os.path.join(config.exp.output_path, 'results.csv')}",
            res
        )
        
    @override
    def reset(self):
        if hasattr(self, "memory") and self.memory:
            del self.memory
        self.memory = None
        
    @override
    def close(self):
        if hasattr(self, "memory") and self.memory:
            del self.memory
        if hasattr(self, "llm") and self.llm:
            del self.llm
        if hasattr(self, "ebd") and self.ebd:
            del self.ebd
        gc.collect()
        torch.cuda.empty_cache()
    
    def _build_memory(self):
        # Build memory items in parallel using threads
        self.all_types = ["original", "event", "keywords", "info", "mix"]
        self.all_retrieved_bys = ["question", "keywords", "possible_answers", "mix"]
                
        store_output_path = os.path.join(self.config.exp.output_path, "vectorstore", f"{self.tid}.bin")
        store_output_path_base = Path(self.config.exp.output_path).parent
        past_store_output_path = os.path.join(store_output_path_base, self.config.agent.memory_version, "vectorstore", f"{self.tid}.bin")
        
        # If vector version is specified and exists, load it directly
        if self.config.agent.memory_version and os.path.exists(past_store_output_path):
            if not os.path.exists(os.path.dirname(store_output_path)):
                os.makedirs(os.path.dirname(store_output_path), exist_ok=True)
            try:
                shutil.copy(past_store_output_path, store_output_path)
            except shutil.SameFileError as e:
                pass  # 源文件和目标文件相同，忽略此错误
            self.memory.load(store_output_path)
        else:
            ThreadTaskRunner(
                tid=self.tid,
                stage="build_memory",
                data_items=self.memory.history.historyItems,
                func=self.memory.build_memory_by_item,
                msg_queue=self.msg_queue,
                res_queue=None
            ).run(max_workers=self.config.exp.max_thread_workers, max_inflight=self.config.exp.max_thread_inflight)
            
            self.memory.dump(store_output_path)
    
    def _invoke_item(self, qa: QA) -> Results:
        response = self.invoke_graph.invoke(qa)
        return Results(items=[self._evaluate_item(qa, response)])
    
    def _evaluate_item(self, qa: QA, response: InvokeState) -> ResultItem:
        # # NOTE DEBUG 暂时不做自动评测
        # judgment = "Not Judged"
        judgment = judge_answer(
            question=qa.question,
            answer=response["answer"],
            ground_truth=qa.answer,
            config=self.config,
        )
        
        hit_sessions = []
        retrieved_sessions = []
        hit_type_retrieved_pairs = []
        type_hit_nums = {}
        retrieved_by_hit_nums = {}
        for t in self.all_types:
            type_hit_nums[t] = 0
        for s in self.all_retrieved_bys:
            retrieved_by_hit_nums[s] = 0
        for doc in response.get("retrieved_docs", []):
            cid = doc.metadata.get("contentID", None)
            retrieved_sessions.append(cid)
            if cid and hit_evidence(cid, qa.evidenceCIDs):
                hit_sessions.append(cid)
                types = doc.metadata.get("type", set())
                retrieved_bys = doc.metadata.get("retrieved_by", set())
                for t in types:
                    type_hit_nums[t] += 1
                for s in retrieved_bys:
                    retrieved_by_hit_nums[s] += 1
                for t, s in zip(types, retrieved_bys):
                    hit_type_retrieved_pairs.append(f"{cid}:{t}-{s}")
        result = ResultItem(
            tid=self.tid,
            questionID=qa.questionID,
            question=qa.question,
            questionType=qa.questionType,
            answer=qa.answer,
            response=response["answer"],
            judgment=judgment,
            additional_info={
                "all_session_ids": ";".join(qa.evidenceCIDs),
                "retrieved_sessions": ";".join([s for s in retrieved_sessions if s]),
                "hit_sessions": ";".join(hit_sessions),
                "num_all_session_ids": len(qa.evidenceCIDs),
                "num_retrieved": len([s for s in retrieved_sessions if s]),
                "num_hit": len(hit_sessions),
                "retrieval_precision": len(hit_sessions) / len(retrieved_sessions) if len(retrieved_sessions) > 0 else 0.0,
                "retrieval_recall": len(hit_sessions) / len(qa.evidenceCIDs) if len(qa.evidenceCIDs) > 0 else 1.0,
                "hit_type_retrieved_pairs": ";".join(hit_type_retrieved_pairs),
                "input_tokens": response.get("input_tokens", 0),
                "output_tokens": response.get("output_tokens", 0),
                "total_tokens": response.get("total_tokens", 0)
            }
        )
        for s, num in retrieved_by_hit_nums.items():
            result.additional_info[f"retrieved_by_{s}_hit_rate"] = num / len(qa.evidenceCIDs) if len(qa.evidenceCIDs) > 0 else 0.0
        for t, num in type_hit_nums.items():
            result.additional_info[f"type_{t}_hit_rate"] = num / len(qa.evidenceCIDs) if len(qa.evidenceCIDs) > 0 else 0.0
            
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
                    res = row["response"]
                    if res.startswith("\""):
                        res = res[1:]
                    if res.endswith("\""):
                        res = res[:-1]
                    row["response"] = res
                    metrics: Dict[str, float] = calculate_metrics(row['response'], row['answer'])
                    row.update(metrics)
                    results.append(row)
            
        with open(result_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                writer.writerow(res)
        
        summary: Dict[str, Dict[str, float]] = {}
        data.questionTypeList.sort()
        for qtype in data.questionTypeList + ["avg"]:
            summary[str(qtype)] = {
                "questionType": str(qtype),
                "f1_sum": 0.0,
                "bleu1_sum": 0.0,
                "retrieval_recall_sum": 0.0,
                "correct_num": 0,
                "count": 0,
                "input_tokens_sum": 0,
                "output_tokens_sum": 0,
                "total_tokens_sum": 0
            }
        for res in results:
            for qtype in [res['questionType'], "avg"]:
                summary[qtype]["f1_sum"] += float(res.get("f1", 0.0))
                summary[qtype]["bleu1_sum"] += float(res.get("bleu1", 0.0))
                summary[qtype]["retrieval_recall_sum"] += float(res.get("retrieval_recall", 0.0))
                if res['judgment'] == "CORRECT":
                    summary[qtype]["correct_num"] += 1
                summary[qtype]["count"] += 1
                summary[qtype]["input_tokens_sum"] += int(res.get("input_tokens", 0))
                summary[qtype]["output_tokens_sum"] += int(res.get("output_tokens", 0))
                summary[qtype]["total_tokens_sum"] += int(res.get("total_tokens", 0))
        
        # 计算avg
        for qtype, metrics in summary.items():
            count = metrics["count"]
            metrics["f1"] = metrics["f1_sum"] / count if count > 0 else 0.0
            metrics["bleu1"] = metrics["bleu1_sum"] / count if count > 0 else 0.0
            metrics["retrieval_recall"] = metrics["retrieval_recall_sum"] / count if count > 0 else 0.0
            metrics["accuracy"] = metrics["correct_num"] / count if count > 0 else 0.0
            metrics["avg_input_tokens"] = metrics["input_tokens_sum"] / count if count > 0 else 0.0
            metrics["avg_output_tokens"] = metrics["output_tokens_sum"] / count if count > 0 else 0.0
            metrics["avg_total_tokens"] = metrics["total_tokens_sum"] / count if count > 0 else 0.0
            # 删除sum字段
            del metrics["f1_sum"]
            del metrics["bleu1_sum"]
            del metrics["retrieval_recall_sum"]
            del metrics["correct_num"]
            del metrics["input_tokens_sum"]
            del metrics["output_tokens_sum"]
            del metrics["total_tokens_sum"]
            
        # 写入summary文件
        sum_results_path = os.path.join(config.exp.output_path, 'summary.csv')
        with open(sum_results_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = summary["avg"].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for qtype, metrics in summary.items():
                writer.writerow(metrics)
        
# ======================================================

class InvokeState(TypedDict):
    qa: QA
    messages: Annotated[List[AnyMessage], add_messages]
    retrieve_base: Annotated[Dict[str, Any], or_]
    retrieved_docs: List[MyDocument]
    other_info: Dict[str, Any]
    answer: str
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]
    total_tokens: Annotated[int, add]
    
class InvokeGraph:
    """Top-level InvokeGraph that encapsulates the main question -> retrieval -> answer StateGraph.

    Nodes are implemented as methods that delegate to an `agent` object for LLM and
    retriever access.
    """
    def __init__(self, llm: LLMProxy, memory: MyMemory):
        self.llm: LLMProxy = llm
        self.memory: MyMemory = memory
        workflow = StateGraph(InvokeState)
        # workflow.add_node("summary_entity", self._summary_entity)
        workflow.add_node("summary_keywords", self._summary_keywords)
        workflow.add_node("guess_possible_answers", self._guess_possible_answers)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate_answer", self._generate_answer)

        # 定义流程：START 并行到三个节点 -> 三个节点到 retrieve -> retrieve 到 generate_answer -> END
        # workflow.add_edge(START, "summary_entity")
        workflow.add_edge(START, "summary_keywords")
        workflow.add_edge(START, "guess_possible_answers")
        # workflow.add_edge("summary_entity", "retrieve")
        workflow.add_edge("summary_keywords", "retrieve")
        workflow.add_edge("guess_possible_answers", "retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", END)

        self.graph = workflow.compile()

    def invoke(self, qa: QA):
        
        state: InvokeState = {
            "qa": qa,
            "messages": [],
            "retrieve_base": {},
            "retrieved_docs": [],
            "other_info": {},
            "answer": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
        
        res = self.graph.invoke(state)
        
        return res

    def _summary_keywords(self, state: InvokeState):
        PROMPT = f"Summarize the keywords and entities in the following question for retrieval:\n" + \
        f"\n\"{state['qa'].question}\"\n\n" + \
        f"Answer with a list of keywords and entities only, separated by commas.\n"
        msg = self.llm.invoke([{"role": "user", "content": PROMPT}])
        res = get_msg_content(msg)
        in_token, out_token, total_token = get_msg_token(msg)
        
        keywords: list = [res]
        
        return {"retrieve_base": {"keywords": keywords}, "input_tokens": in_token, "output_tokens": out_token, "total_tokens": total_token}
    
    def _guess_possible_answers(self, state: InvokeState):
        PROMPT = f"Try to guess the events or information related to the question:\n" + \
        f"\n\"{state['qa'].question}\"\n\n" + \
        f"Answer with a list of at most three possible relevant events or information only, each answer start with a hyphen.\n"
        msg = self.llm.invoke([{"role": "user", "content": PROMPT}])
        res = get_msg_content(msg)
        in_token, out_token, total_token = get_msg_token(msg)
        
        possible_answers: list = [e.strip() for e in res.split("-") if e.strip()]
        # possible_answers: list = [res]
        
        return {"retrieve_base": {"possible_answers": possible_answers}, "input_tokens": in_token, "output_tokens": out_token, "total_tokens": total_token}

    def _retrieve(self, state: InvokeState):
        # 添加问题本身作为检索依据
        question = state["qa"].question
        state["retrieve_base"]["question"] = [question]
        
        mix_v = []
        for key, value in state["retrieve_base"].items():
            mix_v.extend(value)
        state["retrieve_base"]["mix"] = mix_v
        
        # # w/o possible_answers
        # state["retrieve_base"]["mix"] = []
        def filter_func(doc: MyDocument) -> bool:
            # return True
            return "mix" not in doc.metadata.get("type", "")
        
        k=3 # default 4 per query
        c=2 # 5 total contexts per query, default 2
        unique_docs_dict: Dict[str, MyDocument] = {}
        for key, value in state["retrieve_base"].items():
            for v in value:
                if isinstance(v, str):
                    query_v = [v]
                else:
                    query_v = v
                query_results: List[Tuple[MyDocument, float]] = self.memory.similarity_search_with_score(query_v,k=k, filter=filter_func)
                for doc, score in query_results:
                    cid = doc.metadata.get("contentID", None)
                    if cid not in unique_docs_dict:
                        doc.metadata["retrieved_by"] = {f"{key}"}
                        unique_docs_dict[cid] = doc
                    else:
                        existing_doc = unique_docs_dict[cid]
                        existing_doc.metadata["type"] = existing_doc.metadata.get("type", set()) | doc.metadata.get("type", set())
                        existing_doc.metadata["retrieved_by"] = existing_doc.metadata.get("retrieved_by", set()) | {f"{key}"}
                        unique_docs_dict[cid] = existing_doc
        
        # 将上下文记忆也加入检索结果
        unique_cids = list(unique_docs_dict.keys())
        for ucid in unique_cids:
            before_after_k_items = self.memory.getBeforeAfterKItems(ucid, k=c)

            types = unique_docs_dict[ucid].metadata.get("type", set())
            retrieved_bys = unique_docs_dict[ucid].metadata.get("retrieved_by", set())
            
            for item in before_after_k_items:
                cid = item.contentID
                if cid not in unique_docs_dict:
                    doc_metadata = {"sessionID": item.sessionID, "date": item.date, "role": item.role, "contentID": item.contentID, "type": types, "retrieved_by": retrieved_bys}
                    content = self.memory.getContentByID(cid)
                    unique_docs_dict[cid] = MyDocument(page_content=[content], metadata=doc_metadata)
                else:
                    existing_doc = unique_docs_dict[cid]
                    existing_doc.metadata["type"] = existing_doc.metadata.get("type", set()) | types
                    existing_doc.metadata["retrieved_by"] = existing_doc.metadata.get("retrieved_by", set()) | retrieved_bys
                    unique_docs_dict[cid] = existing_doc
                
        sorted_docs = [unique_docs_dict[cid] for cid in sorted(unique_docs_dict.keys())]

        return {"retrieved_docs": sorted_docs}
    
    def _generate_answer(self, state: InvokeState):
        # 结合问题和检索到的文档生成答案
        infos = []
        info_dict = {}
        template = f"""
        - speaker {{role}}: {{content}}
        ......"""
        for i, doc in enumerate(state["retrieved_docs"]):
            sessionID = doc.metadata.get("sessionID", "no_session")
            if sessionID not in info_dict:
                info_dict[sessionID] = f"\n\n- Retrieved Information {len(info_dict)+1}:" + \
                                       f"\n    - date of dialog: {doc.metadata.get('date', '')}" + \
                                       template.format(role=doc.metadata.get("role", ""),
                                                       content=self.memory.getContentByID(doc.metadata.get("contentID")).strip().replace("\n", "    "))
            else:
                info_dict[sessionID] += template.format(role=doc.metadata.get("role", ""),
                                                       content=self.memory.getContentByID(doc.metadata.get("contentID")).strip().replace("\n", "    "))

        # NOTE DEBUG 按session合并
        for session_info in info_dict.values():
            infos.append(session_info)
        
        # # 先提问筛选相关的检索信息
        # PROMPT_FILTER = f"From the following retrieved information, select only the relevant information that can help answer the question. Discard any irrelevant information. Keep auxiliary date and speaker information. Keep entire session information intact.\n\n" + \
        # f"Respond with the relevant information only, without any additional explanation.\n\n" + \
        # f"{''.join(infos)}" + \
        # f"\n\nQuestion Info:" + \
        # (f"\ntoday is {state['qa'].questionDate}.\n" if state['qa'].questionDate else "") + \
        # f"\nQuestion: {state['qa'].question}\n" + \
        # f"\nRelevant Information:"
        # msg = self.llm.invoke([{"role": "user", "content": PROMPT_FILTER}])
        # filtered_info = get_msg_content(msg)
        # in_token, out_token, total_token = get_msg_token(msg)
    
        filtered_info = ''.join(infos)
        in_token, out_token, total_token = 0, 0, 0
    
        # # 根据检索信息生成详细回答
        # PROMPT = f"Based on the following filtered retrieved information, question date(if exists) and question, provide a comprehensive answer. If you can't reason about the answer, please just answer \"not mentioned\" or \"no answer\".\n" + \
        # f"The answer should be an absolute value, not a relative value. For example, the answer should not be \"last year\", but a specific year.\n\n" + \
        # f"Retrieved Information:\n{filtered_info}\n\n" + \
        # (f"Question Date: {state['qa'].questionDate}.\n" if state['qa'].questionDate else "") + \
        # f"Question: {state['qa'].question}\n\n" + \
        # f"Your Response:"
    
        # 根据检索信息生成详细回答
        PROMPT = f"Based on the following filtered retrieved information, question date(if exists) and question, provide a comprehensive answer. \n" + \
        f"The answer should be an absolute value, not a relative value. For example, the answer should not be \"last year\", but a specific year.\n\n" + \
        f"Retrieved Information:\n{filtered_info}\n\n" + \
        (f"Question Date: {state['qa'].questionDate}.\n" if state['qa'].questionDate else "") + \
        f"Question: {state['qa'].question}\n\n" + \
        f"Your Response:"

        msg = self.llm.invoke([{"role": "user", "content": PROMPT}])
        detailed_res = get_msg_content(msg)
        in_token2, out_token2, total_token2 = get_msg_token(msg)
        
        # 再次提问来获取简略回答
        PROMPT_BRIEF = f"Provide a brief and concise answer to the question based on the following detailed answer. Complete sentences are not required; just answer the question directly. \n\n" + \
        f"=================\n" + \
        f"Example: \n" + \
        f"- Question: \"When was the company founded?\"\nDetailed Answer: \"The company was founded in 1998 by a group of entrepreneurs.\"\nBrief Answer: \"1998\"\n\n" + \
        f"- Question: \"Who is the CEO of the company?\"\nDetailed Answer: \"The CEO of the company is John Doe, who has been leading the company since 2015.\"\nBrief Answer: \"John Doe\"\n\n" + \
        f"- Question: \"What is the main product of the company?\"\nDetailed Answer: \"The main product of the company is a software platform that provides cloud computing services to businesses worldwide.\"\nBrief Answer: \"cloud computing platform\"\n\n" + \
        f"=================\n" + \
        f"Now, based on the detailed answer below, provide a brief answer to the question.\n\n" + \
        f"Question: {state['qa'].question}\n\nDetailed Answer: {detailed_res}\n\nBrief Answer:"
        
        msg = self.llm.invoke([{"role": "user", "content": PROMPT_BRIEF}])
        brief_res = get_msg_content(msg)
        in_token3, out_token3, total_token3 = get_msg_token(msg)
        
        in_token = in_token + in_token2 + in_token3
        out_token = out_token + out_token2 + out_token3
        total_token = total_token + total_token2 + total_token3
        
        return {"answer": brief_res, "input_tokens": in_token, "output_tokens": out_token, "total_tokens": total_token}


class LocomoInvokeGraph(InvokeGraph):
    def __init__(self, llm: LLMProxy, memory: MyMemory):
        super().__init__(llm, memory)
        
    @override
    def _generate_answer(self, state: InvokeState):
        # 结合问题和检索到的文档生成答案
        infos = []
        info_dict = {}
        template = f"""
        - speaker {{role}}: {{content}}
        ......"""
        for i, doc in enumerate(state["retrieved_docs"]):
            sessionID = doc.metadata.get("sessionID", "no_session")
            if sessionID not in info_dict:
                info_dict[sessionID] = f"\n\n- Retrieved Information {len(info_dict)+1}:" + \
                                       f"\n    - date of dialog: {doc.metadata.get('date', '')}" + \
                                       template.format(role=doc.metadata.get("role", ""),
                                                       content=self.memory.getContentByID(doc.metadata.get("contentID")).strip().replace("\n", "    "))
            else:
                info_dict[sessionID] += template.format(role=doc.metadata.get("role", ""),
                                                       content=self.memory.getContentByID(doc.metadata.get("contentID")).strip().replace("\n", "    "))

        # NOTE DEBUG 按session合并
        for session_info in info_dict.values():
            infos.append(session_info)
            
        # if state["qa"].questionType in [1,2,3,4,5, "1", "2", "3", "4", "5"]:
        # if state["qa"].questionType in [2,3,5]:
        if state["qa"].questionType in []:
            PROMPT_FILTER = f"From the following retrieved information, select only the relevant information that can help answer the question. Discard any irrelevant information. Keep auxiliary date and speaker information. Keep entire session information intact.\n\n" + \
            f"Respond with the relevant information only, without any additional explanation.\n\n" + \
            f"{''.join(infos)}" + \
            f"\n\nQuestion Info:" + \
            (f"\ntoday is {state['qa'].questionDate}.\n" if state['qa'].questionDate else "") + \
            f"\nQuestion: {state['qa'].question}\n" + \
            f"\nRelevant Information:"
            msg = self.llm.invoke([{"role": "user", "content": PROMPT_FILTER}])
            filtered_info = get_msg_content(msg)
            in_token, out_token, total_token = get_msg_token(msg)
        else:
            filtered_info = ''.join(infos)
            in_token, out_token, total_token = 0, 0, 0
    
        if state["qa"].questionType in [5, "5"]:
            answer_tmp = list()
            if random.random() < 0.5:
                answer_tmp.append('not mentioned')
                answer_tmp.append(state["qa"].adversarial_answer)
            else:
                answer_tmp.append(state["qa"].adversarial_answer)
                answer_tmp.append('not mentioned')
            PROMPT = f"Based on the following filtered retrieved information, question date(if exists) and question, select the correct response between the two options: \n- {answer_tmp[0]}\n- {answer_tmp[1]}\n" + \
            f"Pay attention to the difference between question subject and speaker.\n\n"
        elif state["qa"].questionType in [2, "2"]:
            PROMPT = f"Based on the following filtered retrieved information, question date(if exists) and question, provide a comprehensive answer.\n" + \
            f"Use DATE of CONVERSATION to answer with an approximate date.\n" + \
            f"The answer should mostly be like an absolute value, not a relative value. For example, the answer should not be \"last year\", but a specific year if you can determine it.\n\n"
        else:
            PROMPT = f"Based on the following filtered retrieved information, question date(if exists) and question, provide a comprehensive answer.\n"
            
        PROMPT = PROMPT + f"Retrieved Information:\n{filtered_info}\n\n" + \
            (f"Question Date: {state['qa'].questionDate}.\n" if state['qa'].questionDate else "") + \
            f"Question: {state['qa'].question}\n\n" + \
            f"Your Response:"

        msg = self.llm.invoke([{"role": "user", "content": PROMPT}])
        detailed_res = get_msg_content(msg)
        in_token2, out_token2, total_token2 = get_msg_token(msg)
        
        # 再次提问来获取简略回答
        PROMPT_BRIEF = f"Provide a brief and concise answer to the question based on the following detailed answer. Complete sentences are not required; just answer the question directly. \n\n" + \
        f"=================\n" + \
        f"Example: \n" + \
        f"- Question: \"When was the company founded?\"\nDetailed Answer: \"The company was founded in 1998 by a group of entrepreneurs.\"\nBrief Answer: \"1998\"\n\n" + \
        f"- Question: \"Who is the CEO of the company?\"\nDetailed Answer: \"The CEO of the company is John Doe, who has been leading the company since 2015.\"\nBrief Answer: \"John Doe\"\n\n" + \
        f"- Question: \"What is the main product of the company?\"\nDetailed Answer: \"The main product of the company is a software platform that provides cloud computing services to businesses worldwide.\"\nBrief Answer: \"cloud computing platform\"\n\n" + \
        f"=================\n" + \
        f"Now, based on the detailed answer below, provide a brief answer to the question.\n\n" + \
        f"Question: {state['qa'].question}\n\nDetailed Answer: {detailed_res}\n\nBrief Answer:"
        msg = self.llm.invoke([{"role": "user", "content": PROMPT_BRIEF}])
        brief_res = get_msg_content(msg)
        in_token3, out_token3, total_token3 = get_msg_token(msg)
        
        in_token = in_token + in_token2 + in_token3
        out_token = out_token + out_token2 + out_token3
        total_token = total_token + total_token2 + total_token3
        
        return {"answer": brief_res, "input_tokens": in_token, "output_tokens": out_token, "total_tokens": total_token}

