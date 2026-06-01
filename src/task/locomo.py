import json
from typing import List, override
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from components.config import MyConfig
from task.basetask import BaseTaskWorker, BaseTask
from components.types import QA, DataItem, DataSet, History, HistoryItem


class LocomoTask(BaseTask):
    def __init__(self, config: MyConfig):
        super().__init__(config, BaseTaskWorker)
    
    @override
    def _load_data(self, full_path: str) -> DataSet:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        dataset = DataSet(items=[])
        for item in tqdm(data):
            dataItem = DataItem(
                tid=item["sample_id"],
                history=History(historyItems=[], cid2content={}),
                qaList=[]
            )
            for qIndex, qa in enumerate(item["qa"]):
                if qa["category"] not in dataset.questionTypeList:
                    dataset.questionTypeList.append(qa["category"])
                qid = f"{item['sample_id']}_q{qIndex}"
                evi:List[str] = qa.get("evidence", [])
                for idx, cid in enumerate(evi):
                    sessionID = f"{cid.split(':')[0].split('D')[-1]}"
                    diaID = cid.split(':')[1]
                    evi[idx] = f"{sessionID.zfill(3)}:{diaID.zfill(3)}:0"
                qaItem = QA(
                    questionID=qid,
                    question=qa["question"],
                    questionDate=None,
                    questionType=qa["category"],
                    answer=qa.get("answer", "not mentioned"),
                    adversarial_answer=qa.get("adversarial_answer", ""),
                    evidenceCIDs=evi
                )
                dataItem.qaList.append(qaItem)
            history = History(historyItems=[], cid2content={})
            session_index = 1
            while item["conversation"].get(f"session_{session_index}"):
                session = item["conversation"].get(f"session_{session_index}")
                sessionID = f"{session_index}"
                sessionDate = item["conversation"].get(f"session_{sessionID}_date_time", "no date")
                for turn in session:
                    role = turn["speaker"]
                    diaID: str = turn["dia_id"]
                    did = diaID.split(':')[1]
                    contentID = f"{sessionID.zfill(3)}:{did.zfill(3)}:0"
                    historyItem = HistoryItem(
                        sessionID=sessionID,
                        date=sessionDate,
                        role=role,
                        contentID=contentID
                    )
                    
                    text = turn.get("text", "")
                    if "img_url" in turn and "blip_caption" in turn:
                        caption_text = f"[Image: {turn['blip_caption']}]"
                        text = f"{caption_text} {text}"
                    history.historyItems.append(historyItem)
                    if sessionID.zfill(3) not in history.cid2content:
                        history.cid2content[sessionID.zfill(3)] = {}
                    if did.zfill(3) not in history.cid2content[sessionID.zfill(3)]:
                        history.cid2content[sessionID.zfill(3)][did.zfill(3)] = {}
                    history.cid2content[sessionID.zfill(3)][did.zfill(3)]["0"] = text
                session_index += 1
            dataItem.history = history
            dataset.items.append(dataItem)
            
        return dataset