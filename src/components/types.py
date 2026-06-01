from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TypedDict

@dataclass
class HistoryItem:
    sessionID: str
    date: str
    role: str
    # contentID = sessionID:indexOfSession:docChunkIndex
    contentID: str

@dataclass
class History:
    historyItems: List[HistoryItem]
    cid2content: Dict[str, Dict[str, Dict[str, str]]]

    def getContentByID(self, contentID: str) -> str:
        index = contentID.split(":")
        return self.cid2content[index[0]][index[1]][index[2]]
    
    def getBeforeAfterKItems(self, contentID: List[str] | str, k: int, include_self: bool = False) -> List[HistoryItem]:
        if isinstance(contentID, str):
            contentID = [contentID]
        res = []
        for idx, item in enumerate(self.historyItems):
            if item.contentID in contentID:
                cid = item.contentID
                start = max(0, idx - k)
                end = min(len(self.historyItems), idx + k + 1)
                temp: List[HistoryItem] = []
                if include_self:
                    temp.extend(self.historyItems[start:end])
                else:
                    temp.extend(self.historyItems[start:idx])
                    temp.extend(self.historyItems[idx+1:end])
                for t in temp:
                    if t.contentID.startswith(cid.split(":")[0]) and t not in res:
                        res.append(t)
        return res

@dataclass
class QA:
    questionID: str
    question: str
    questionDate: str
    questionType: str
    answer: str
    adversarial_answer: str
    # can be end with * for prefix match, implemented in is_hit method of BaseAgent
    evidenceCIDs: List[str]
    
@dataclass
class DataItem:
    tid: str
    history: History
    qaList: List[QA]

@dataclass
class DataSet:
    items: List[DataItem]
    length: int = 0
    questionTypeList: list = field(default_factory=list)
    
# ======================================================
    
@dataclass
class ResultItem:
    tid: str
    questionID: str
    question: str
    questionType: str
    answer: str
    response: str
    judgment: str
    additional_info: Dict[str, str]
    
@dataclass
class Results:
    items: List[ResultItem]