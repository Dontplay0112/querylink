import json
from tqdm import tqdm
from typing import Dict, override
from components.config import MyConfig
from task.basetask import BaseTaskWorker, BaseTask
from components.types import QA, DataItem, DataSet, History, HistoryItem
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LongMemEvalTask(BaseTask):
    def __init__(self, config: MyConfig):
        super().__init__(config, BaseTaskWorker)
    
    @override
    def _load_data(self, full_path: str) -> DataSet:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=0, disallowed_special=()
        )
        dataset = DataSet(items=[])
        for item in tqdm(data):
            evidenceCIDs = item["answer_session_ids"]
            for idx, session_id in enumerate(evidenceCIDs):
                evidenceCIDs[idx] = session_id+"*"
            if item["question_type"] not in dataset.questionTypeList:
                dataset.questionTypeList.append(item["question_type"])
            dataItem = DataItem(
                tid=item["question_id"],
                history=History(historyItems=[], cid2content={}),
                qaList=[
                    QA(
                        questionID=item["question_id"],
                        question=item["question"],
                        questionDate=item["question_date"],
                        questionType=item["question_type"],
                        answer=item["answer"],
                        evidenceCIDs=evidenceCIDs,
                        adversarial_answer="",
                    )
                ]
            )
            cid2content: Dict[str, Dict[str, Dict[str, str]]] = {}
            for i in range(len(item["haystack_sessions"])):
                for j in range(len(item["haystack_sessions"][i])):
                    for doc_chunk_index, doc_chunk in enumerate(text_splitter.split_text(item["haystack_sessions"][i][j]["content"])):
                        dataItem.history.historyItems.append(
                            HistoryItem(
                                sessionID=item["haystack_session_ids"][i],
                                date=item["haystack_dates"][i],
                                role=item["haystack_sessions"][i][j]["role"],
                                contentID=item["haystack_session_ids"][i] + ":" + str(j) + ":" + str(doc_chunk_index),
                            )
                        )
                        if item["haystack_session_ids"][i] not in cid2content:
                            cid2content[item["haystack_session_ids"][i]] = {}
                        if str(j) not in cid2content[item["haystack_session_ids"][i]]:
                            cid2content[item["haystack_session_ids"][i]][str(j)] = {}
                        cid2content[item["haystack_session_ids"][i]][str(j)][str(doc_chunk_index)] = doc_chunk
            dataItem.history.cid2content = cid2content
            dataset.items.append(dataItem)
        return dataset