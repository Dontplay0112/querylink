from abc import ABC, abstractmethod
import logging
import multiprocessing as mp
from concurrent.futures import Future, as_completed, ThreadPoolExecutor
import os
import traceback
from typing import Any, Callable, List, Set
from components.config import MyConfig
from components.types import DataItem, Results
    
class ThreadTaskRunner:
    def __init__(self, 
                 tid: str, 
                 stage: str,
                 data_items: List[Any], 
                 func: Callable, 
                 msg_queue: mp.Queue = None, 
                 res_queue: mp.Queue = None):
        self.tid = tid
        self.stage = stage
        self.data_items = data_items
        self.func = func
        self.msg_queue = msg_queue
        self.res_queue = res_queue

    def _notify(self, completed: int, total: int):
        if self.msg_queue:
            try:
                self.msg_queue.put(("worker", self.stage, os.getpid(), self.tid, completed, total))
            except Exception as e:
                logging.error(f"Failed to put message to queue: {e}")

    def run(self, max_workers: int = 4, max_inflight: int = 8):
        total = len(self.data_items)
        completed = 0
        in_flight: Set[Future] = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for item in self.data_items:
                while len(in_flight) >= max_inflight:
                    done_futures = {f for f in in_flight if f.done()}
                    for done_future in done_futures:
                        completed += 1
                        self._notify(completed, total)
                        in_flight.remove(done_future)

                future: Future = executor.submit(self._wrapped_work, item)
                in_flight.add(future)
                
                done_count = sum(1 for f in in_flight if f.done())
                if done_count > 0:
                    pass

            for future in as_completed(in_flight):
                completed += 1
                self._notify(completed, total)

    def _wrapped_work(self, item):
        try:
            result = self.func(item)
            if self.res_queue:
                try:
                    self.res_queue.put(result)
                except Exception as e:
                    logging.error(f"Failed to put result to queue: {e}")
            return result
        except Exception as e:
            tb = traceback.format_exc()
            logging.error(f"Error processing item {item}: {e}\n{tb}")

class BaseAgent(ABC):
    """
    Base agent class.
    """
    def __init__(self, config: MyConfig, msg_queue: mp.Queue, res_queue: mp.Queue):
        super().__init__()
        
        self.config = config
        self.msg_queue = msg_queue
        self.res_queue = res_queue
        
    @abstractmethod
    def init(self, dataItem: DataItem):
        """
        读入数据并处理。
        """
        pass

    @abstractmethod
    def solve(self):
        """
        正式解决问题。
        """
        pass
    
    @classmethod
    @abstractmethod
    def res_process(cls, res: Results, config: MyConfig):
        """
        为了了多进程安全，结果处理函数必须是类方法，交给主进程去执行。
        """
        pass
    
    @classmethod
    @abstractmethod
    def evaluate_all(cls, config: MyConfig, **kwargs):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def close(self):
        pass