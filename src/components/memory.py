"""In-memory vector store."""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import threading
import uuid
from operator import add
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict, override
from multiprocessing import Queue
import numpy as np
from collections.abc import Iterator, Sequence
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import _cosine_similarity as cosine_similarity
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langgraph.graph import StateGraph, START, END
from tqdm import tqdm

from components.config import MyConfig
from components.llm import LLMProxy
from components.types import History, HistoryItem
from components.utils import get_msg_content, get_msg_token

class MyDocument(Document):
    page_content: str | list[str]
    
class DocFilter:
    INCLUDE = "include"
    EXCLUDE = "exclude"
    EQUALS = "equals"
    MIN = "min"
    MAX = "max"
    
    def __new__(cls, conditions: dict = None):
        if conditions is None or not isinstance(conditions, dict) or len(conditions) == 0:
            return None
        return super(DocFilter, cls).__new__(cls)
    
    def __init__(self, conditions: dict = None):
        self.conditions = conditions
    
    def __call__(self, doc: MyDocument) -> bool:
        for key, value in self.conditions.items():
            assert isinstance(value, dict), "Condition value must be a dict"
            meta = doc.metadata.get(key, None)
            for cond, cond_value in value.items():
                if cond == self.INCLUDE:
                    if not meta or cond_value not in meta:
                        return False
                elif cond == self.EXCLUDE:
                    if meta and cond_value in meta:
                        return False
                elif cond == self.EQUALS:
                    if meta != cond_value:
                        return False
                elif cond == self.MIN:
                    if meta is None or meta < cond_value:
                        return False
                elif cond == self.MAX:
                    if meta is None or meta > cond_value:
                        return False
            if meta is None:
                return False
        return True

class MyMemory(VectorStore):
    def __init__(self, config: MyConfig, tid: str, msg_queue: Queue, embedding: Embeddings, llm: LLMProxy, history: History):
        """Initialize with the given embedding function.

        Args:
            embedding: embedding function to use.
        """
        self.config = config
        self.tid = tid
        self.msg_queue = msg_queue
        self.history = history
        self.store: dict[str, dict[str, Any]] = {}
        self.embedding = embedding
        self.llm = llm
        
        self.add_documents_lock = threading.Lock()
        
        # ========Tools========
        self.memory_summary_graph = MemorySummaryGraph(self.llm, self)

    @property
    @override
    def embeddings(self) -> Embeddings:
        return self.embedding

    @override
    def delete(self, ids: Sequence[str] | None = None, **kwargs: Any) -> None:
        if ids:
            with self.add_documents_lock:
                for _id in ids:
                    self.store.pop(_id, None)
    
    def getContentByID(self, contentID: str) -> str:
        return self.history.getContentByID(contentID)
    
    def getBeforeAfterKItems(self, contentID: List[str] | str, k: int, include_self: bool = False) -> List[HistoryItem]:
        return self.history.getBeforeAfterKItems(contentID, k, include_self=include_self)
                    
    def get_history_len(self) -> int:
        return len(self.history.historyItems)
        
    def build_memory_by_item(self, item: HistoryItem):
        """
        Build memory based on a single HistoryItem.
        """
        self.memory_summary_graph.remember(item)
            
    # ================================ VectorStore methods =================================
    @override
    def add_documents(
        self,
        documents: list[MyDocument],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = [doc.page_content for doc in documents]        
        vectors = self.embedding.embed_documents(texts)        

        with self.add_documents_lock:
            if ids and len(ids) != len(texts):
                msg = (
                    f"ids must be the same length as texts. "
                    f"Got {len(ids)} ids and {len(texts)} texts."
                )
                raise ValueError(msg)

            id_iterator: Iterator[str | None] = (
                iter(ids) if ids else iter(doc.id for doc in documents)
            )

            ids_ = []

            for doc, vector in zip(documents, vectors, strict=False):
                doc_id = next(id_iterator)
                doc_id_ = doc_id or str(uuid.uuid4())
                ids_.append(doc_id_)
                self.store[doc_id_] = {
                    "id": doc_id_,
                    "vector": vector,
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }

        return ids_

    @override
    def get_by_ids(self, ids: Sequence[str], /) -> list[MyDocument]:
        """Get documents by their ids.

        Args:
            ids: The IDs of the documents to get.

        Returns:
            A list of `MyDocument` objects.
        """
        documents = []

        for doc_id in ids:
            doc = self.store.get(doc_id)
            if doc:
                documents.append(
                    MyDocument(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=doc["metadata"],
                    )
                )
        return documents
    
    def get_by_filter(self, filter: Callable[[MyDocument], bool] | DocFilter) -> list[MyDocument]:
        docs = list(self.store.values())
        if filter is not None:
            docs = [
                doc
                for doc in docs
                if filter(
                    MyDocument(
                        id=doc["id"], page_content=doc["text"], metadata=doc["metadata"]
                    )
                )
            ]
        return [
            MyDocument(
                id=doc_dict["id"],
                page_content=doc_dict["text"],
                metadata=doc_dict["metadata"],
            )
            for doc_dict in docs
        ]
        
    def get_by_filter_with_score(
        self,
        filter: Callable[[MyDocument], bool] | DocFilter,
        query: str | list[str],
    ) -> list[tuple[MyDocument, float]]:
        docs = self.get_by_filter(filter)
        if not docs:
            return []
        if isinstance(query, str):
            query = [query]
        embedding = self.embedding.embed_query(query)
        similarity = cosine_similarity([embedding], [self.embedding.embed_query(doc.page_content) for doc in docs])[0]
        return [
            (doc, float(similarity[idx].item()))
            for idx, doc in enumerate(docs)
        ]

    def _similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable[[MyDocument], bool] | DocFilter | None = None,  # noqa: A002
    ) -> list[tuple[MyDocument, float, list[float]]]:
        # get all docs with fixed order in list
        docs = list(self.store.values())

        if filter is not None:
            docs = [
                doc
                for doc in docs
                if filter(
                    MyDocument(
                        id=doc["id"], page_content=doc["text"], metadata=doc["metadata"]
                    )
                )
            ]

        if not docs:
            return []

        similarity = cosine_similarity([embedding], [doc["vector"] for doc in docs])[0]

        # get the indices ordered by similarity score
        top_k_idx = similarity.argsort()[::-1][:k]

        return [
            (
                MyDocument(
                    id=doc_dict["id"],
                    page_content=doc_dict["text"],
                    metadata=doc_dict["metadata"],
                ),
                float(similarity[idx].item()),
                doc_dict["vector"],
            )
            for idx in top_k_idx
            # Assign using walrus operator to avoid multiple lookups
            if (doc_dict := docs[idx])
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable[[MyDocument], bool] | DocFilter | None = None,  # noqa: A002
        **_kwargs: Any,
    ) -> list[tuple[MyDocument, float]]:
        """Search for the most similar documents to the given embedding.

        Args:
            embedding: The embedding to search for.
            k: The number of documents to return.
            filter: A function to filter the documents.

        Returns:
            A list of tuples of MyDocument objects and their similarity scores.
        """
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, filter=filter
            )
        ]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[MyDocument, float]]:
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )

    @override
    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[MyDocument]:
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[MyDocument]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    @override
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        *,
        filter: Callable[[MyDocument], bool] | DocFilter | None = None,
        **kwargs: Any,
    ) -> list[MyDocument]:
        prefetch_hits = self._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
        )

        mmr_chosen_indices = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            [vector for _, _, vector in prefetch_hits],
            k=k,
            lambda_mult=lambda_mult,
        )
        return [prefetch_hits[idx][0] for idx in mmr_chosen_indices]

    @override
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[MyDocument]:
        embedding_vector = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding_vector,
            k,
            fetch_k,
            lambda_mult=lambda_mult,
            **kwargs,
        )

    @classmethod
    @override
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> MyMemory:
        store = cls(
            embedding=embedding,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    def load(self, path: str):
        """Load a vector store from a file.

        Args:
            path: The path to load the vector store from.
            embedding: The embedding to use.
            **kwargs: Additional arguments to pass to the constructor.

        Returns:
            A VectorStore object.
        """
        with open(path, "rb") as f:
            self.store = pickle.load(f)

    def dump(self, path: str) -> None:
        """Dump the vector store to a file.

        Args:
            path: The path to dump the vector store to.
        """
        # 保存向量库到output路径
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.store, f)


# ===========================================================================================

class MemorySummaryState(TypedDict):
    historyItem: HistoryItem
    content: str
    metadata: dict
    words: List[str]
    mem_item: Annotated[List[str], add]
    
    
class MemorySummaryGraph:
    def __init__(self, llm: LLMProxy, memory: MyMemory):
        self.llm: LLMProxy = llm
        self.memory: MyMemory = memory
        workflow = StateGraph(MemorySummaryState)
        workflow.add_node("preprocess_content", self._preprocess_content)
        # workflow.add_node("seperate_original_doc", self._seperate_original_doc)
        workflow.add_node("add_origin_doc", self._add_origin_doc)
        # workflow.add_node("summary_entity", self._summary_entity_for_memory)
        workflow.add_node("summary_all", self._summary_all_for_memory)
        workflow.add_node("get_mix_embedding", self._get_mix_embedding)

        # 并行处理，然后组合
        # 1
        # workflow.add_edge(START, "seperate_original_doc")
        # workflow.add_edge("seperate_original_doc", END)
        # 2
        workflow.add_edge(START, "preprocess_content")
        workflow.add_edge("preprocess_content", "add_origin_doc")
        # workflow.add_edge("preprocess_content", "summary_entity")
        workflow.add_edge("preprocess_content", "summary_all")
        workflow.add_edge("add_origin_doc", "get_mix_embedding")
        # workflow.add_edge("summary_entity", "get_mix_embedding")
        workflow.add_edge("summary_all", "get_mix_embedding")
        workflow.add_edge("get_mix_embedding", END)
        self.graph = workflow.compile()
        
        self.in_token = 0
        self.out_token = 0
        self.total_token = 0

    def remember(self, item: HistoryItem):
        state: MemorySummaryState = {
            "historyItem": item,
            "content": "",
            "metadata": "",
            # "docs": [],
            "words": [],
            "mem_item": []
        }
        self.graph.invoke(state)
    
    # Nodes
    def _preprocess_content(self, state: MemorySummaryState):
        item: HistoryItem = state["historyItem"]
        metadata = {"sessionID": item.sessionID, "date": item.date, "role": item.role, "contentID": item.contentID}
        
        # NOTE 记忆连贯性消融实验
        c = 2
        beforeAfterItems = self.memory.getBeforeAfterKItems([item.contentID], c, include_self=True)
        
        itemContent = ""
        for histItem in beforeAfterItems:
            content = self.memory.getContentByID(histItem.contentID)
            itemContent += f"{histItem.role} ({histItem.date}): {content}\n"
            
        return {"content": itemContent, "metadata": metadata}
        
    def _seperate_original_doc(self, state: MemorySummaryState):
        origin_content_id = state["metadata"].get("contentID", "no_id")
        origin_content = self.memory.getContentByID(origin_content_id)
        PROMPT = f"Break the following content down into words. If a word has changed due to tense or other reasons, restore it to its original form.\nFor example, 'running' should be restored to 'run'.'apples' should be restored to 'apple'. Ignore punctuation, duplicates and common words like 'the', 'is', 'on', etc" + \
        f".\n\nHere is the content: \"{origin_content}\"\n\nRespond with a list of words, separated by commas."
        msg = self.llm.invoke([{"role": "user", "content": PROMPT}])
        res = get_msg_content(msg)
        words = [e.strip() for e in res.split(",") if e.strip()]
        return {"words": words}
    
    def _add_origin_doc(self, state: MemorySummaryState):
        self.memory.add_documents([MyDocument(page_content=[state["content"]], metadata={**state["metadata"], "type": {"original"}})])
        state["mem_item"].append(state["content"])
        
    # def _summary_entity_for_memory(self, state: MemorySummaryState):
    #     PROMPT = f"Summarize the entities in the following content:\n\n\"{state["content"]}\"\n\nRespond with a list of entities, separated by commas."
    #     msg = self.llm.invoke([{"role": "user", "content": PROMPT}])
    #     res = get_msg_content(msg)
    #     self.memory.add_documents([MyDocument(page_content=[res], metadata={**state["metadata"], "type": {"entities"}})])
    #     state["mem_item"].append(res)

    def _summary_all_for_memory(self, state: MemorySummaryState):
        PROMPT = (
            "Summarize the following content into structured lists. "
            "Return a JSON object with keys: keywords, events, info. "
            "Each value must be a list of strings. "
            "Do not include any extra text.\n\n"
            f"Content:\n\"{state['content']}\"\n"
        )
        msg = self.llm.invoke([{"role": "user", "content": PROMPT}])

        in_token, out_token, total_token = get_msg_token(msg)
        self.in_token += in_token
        self.out_token += out_token
        self.total_token += total_token

        res = get_msg_content(msg)
        payload = self._parse_summary_payload(res)

        keywords = payload.get("keywords", [])
        if keywords:
            keywords_text = ", ".join(keywords)
            self.memory.add_documents(
                [
                    MyDocument(
                        page_content=[keywords_text],
                        metadata={**state["metadata"], "type": {"keywords"}},
                    )
                ]
            )
            state["mem_item"].append(keywords_text)

        for event in payload.get("events", []):
            if event.strip():
                self.memory.add_documents(
                    [
                        MyDocument(
                            page_content=[event.strip()],
                            metadata={**state["metadata"], "type": {"event"}},
                        )
                    ]
                )
                state["mem_item"].append(event.strip())

        for info in payload.get("info", []):
            if info.strip():
                self.memory.add_documents(
                    [
                        MyDocument(
                            page_content=[info.strip()],
                            metadata={**state["metadata"], "type": {"info"}},
                        )
                    ]
                )
                state["mem_item"].append(info.strip())
                
    def _parse_summary_payload(self, text: str) -> dict:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        for key in ("keywords", "events", "info"):
            value = payload.get(key, [])
            if isinstance(value, list):
                payload[key] = [str(item).strip() for item in value if str(item).strip()]
            elif isinstance(value, str) and value.strip():
                payload[key] = [value.strip()]
            else:
                payload[key] = []

        return payload
                
    def _get_mix_embedding(self, state: MemorySummaryState) -> list[float]:
        self.memory.add_documents([MyDocument(page_content=state["mem_item"], metadata={**state["metadata"], "type": {"mix"}})])