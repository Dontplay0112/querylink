from abc import ABC, abstractmethod
import csv
import os
import pickle
import logging
import traceback
import threading
import multiprocessing as mp
from multiprocessing import Queue
from time import sleep, time
from typing import List
from tqdm import tqdm
from functools import wraps

from agent import BaseAgent, get_agent
from components.config import MyConfig
from components.decorators import split_line
from components.logger import init_logging
from components.types import DataItem, DataSet, Results


class BaseTaskWorker(ABC):
    """
    Base task worker class.
    """

    def __init__(
        self, config: MyConfig, msg_queue: Queue, task_queue: Queue, res_queue: Queue
    ):
        super().__init__()
        self.config = config
        self.msg_queue = msg_queue
        self.task_queue = task_queue
        self.res_queue = res_queue

    def run(self):
        init_logging(self.config)
        logging.info(f"Worker {mp.current_process().name} started.")

        try:
            agent_class: BaseAgent = get_agent(self.config.agent.agent)
            agent: BaseAgent = agent_class(self.config, self.msg_queue, self.res_queue)
        except Exception as e:
            logging.error(f"Agent initialization failed: {e}\n{traceback.format_exc()}")
            return
        while True:
            try:
                item: DataItem = self.task_queue.get()
                if item is None:
                    break
                try:
                    logging.info(f"Worker {mp.current_process().name} processing item {item.tid}")
                    agent.init(item)
                    # NOTE DEBUG
                    logging.info("Token usage for item {}: in_token={}, out_token={}, total_token={}".format(item.tid, agent.memory.memory_summary_graph.in_token, agent.memory.memory_summary_graph.out_token, agent.memory.memory_summary_graph.total_token))
                    agent.solve()
                    
                    self.msg_queue.put(
                        ("main", "Distributing Tasks", os.getpid(), item.tid, -1, -1)
                    )
                except Exception as e:
                    logging.error(
                        f"Error while processing item {item.tid}: {e}\n{traceback.format_exc()}"
                    )
                finally:
                    agent.reset()
            except Exception as e:
                logging.error(f"Worker queue read error: {e}\n{traceback.format_exc()}")
                break


class TaskRunner:
    def __init__(
        self, config: MyConfig, data_items: List[DataItem], worker_class: BaseTaskWorker
    ):
        self.config = config
        self.data_items = data_items
        self.worker_class = worker_class

        self.msg_queue = mp.Queue()
        self.task_queue = mp.Queue()
        self.res_queue = mp.Queue()

    def _msg_listener(self, msg_queue: mp.Queue, data_length: int = 0):
        max_workers = self.config.exp.max_workers

        main_running_ascii = ".>"
        main_done_ascii = ">>"
        worker_running_ascii = "-" + "".join(map(chr, range(0x258F, 0x2587, -1)))
        worker_done_ascii = ".."
        time_running_ascii = "~~"
        time_stuck_ascii = "--"

        mainBar = tqdm(
            total=data_length,
            desc=f"[Main pid:{os.getpid()}] [Distributing Tasks]",
            position=max_workers,
            leave=True,
            ascii=main_running_ascii,
        )
        worker_bars: list[tqdm] = [None] * (max_workers + 1)
        worker_bars[max_workers] = mainBar
        pid2slot: dict[int, int] = {}
        slots_free = list(range(max_workers))

        start_time = time()
        last_message_time = start_time
        time_bar = tqdm(
            total=1,
            desc="Running Time",
            position=max_workers + 1,
            leave=True,
            ascii=time_running_ascii,
        )
        time_bar.n = 0
        stop_event = threading.Event()

        def refresh_all_bars():
            while not stop_event.is_set():
                elapsed = time() - start_time
                stuck = (time() - last_message_time) > 10

                msg = f"= Running Time: {int(elapsed) // 3600:02d}:{(int(elapsed) % 3600) // 60:02d}:{int(elapsed) % 60:02d} | Last Msg: {int(time() - last_message_time)}s ago {'(Stuck?)' if stuck else ''} ="
                time_bar.set_description_str(msg)
                time_bar.ascii = time_stuck_ascii if stuck else time_running_ascii
                time_bar.colour = "red" if stuck else "green"
                time_bar.refresh()

                if worker_bars[max_workers] is not None:
                    worker_bars[max_workers].colour = "red" if stuck else "green"

                # refresh worker bars
                for bar in worker_bars:
                    if bar is not None:
                        bar.refresh()
                sleep(1)

        time_thread = threading.Thread(target=refresh_all_bars, daemon=True)
        time_thread.start()

        while True:
            try:
                msg = msg_queue.get()
                if msg is None:
                    break

                last_message_time = time()
                src, stage, pid, tid, completed, total = msg

                if src == "main":
                    n = mainBar.n
                    if n + 1 < data_length:
                        mainBar.update(1)
                    else:
                        mainBar.n = data_length
                        mainBar.set_description(
                            f"[DONE][pid:{pid}] [Distributing Tasks]"
                        )
                        mainBar.ascii = main_done_ascii
                        mainBar.refresh()
                    continue

                if src == "worker":
                    if pid not in pid2slot:
                        slot = slots_free.pop(0) if slots_free else 0
                        bar = tqdm(
                            total=total,
                            desc=f"[Worker{max_workers - slot} pid:{pid}][task:{tid}] stage:{stage}",
                            position=slot,
                            leave=True,
                            ascii=worker_running_ascii,
                        )

                        worker_bars[slot] = bar
                        pid2slot[pid] = slot

                    slot = pid2slot[pid]
                    bar = worker_bars[slot]

                    bar.set_description(
                        f"[Worker{max_workers - slot} pid:{pid}][task:{tid} stage:{stage}]"
                    )

                    bar.ascii = worker_running_ascii
                    bar.n = completed
                    bar.total = total
                    bar.refresh()

                    if completed >= total and total > 0:
                        bar.set_description(f"[DONE][task:{tid}] stage:{stage}")
                        bar.ascii = worker_done_ascii
                        bar.n = total
                        bar.refresh()
            except Exception as e:
                logging.error(f"Listener Error: {e}")
                break

        stop_event.set()
        time_thread.join()
        for bar in worker_bars:
            if bar:
                bar.close()
        time_bar.close()

    def _res_listener(self, res_queue: mp.Queue):
        agent_class: BaseAgent = get_agent(self.config.agent.agent)
        while True:
            try:
                res: Results = res_queue.get()
                if res is None:
                    break
                agent_class.res_process(res, self.config)
            except Exception as e:
                logging.error(f"Result Listener Error: {e}")
                break

    def run(self):
        max_workers = self.config.exp.max_workers
        max_inflight = self.config.exp.max_inflight

        msg_listener_thread = threading.Thread(
            target=self._msg_listener, args=(self.msg_queue, len(self.data_items))
        )
        msg_listener_thread.start()

        res_listener_thread = threading.Thread(
            target=self._res_listener, args=(self.res_queue,)
        )
        res_listener_thread.start()

        processes: List[mp.Process] = []
        for i in range(max_workers):
            worker_inst: BaseTaskWorker = self.worker_class(
                self.config, self.msg_queue, self.task_queue, self.res_queue
            )
            p = mp.Process(target=worker_inst.run, name=f"Worker-{i}")
            p.start()
            processes.append(p)

        try:
            for idx, item in enumerate(self.data_items):
                while self.task_queue.qsize() >= max_inflight:
                    sleep(0.1)
                self.task_queue.put(item)
            for _ in range(max_workers):
                self.task_queue.put(None)

        except KeyboardInterrupt:
            logging.warning("Interrupted by user, killing workers...")
            for p in processes:
                p.terminate()
        finally:
            for p in processes:
                p.join()

            # 通知监听线程关闭
            self.msg_queue.put(None)
            msg_listener_thread.join()
            self.res_queue.put(None)
            res_listener_thread.join()


class BaseTask(ABC):
    """
    Base task class.
    """

    def __init__(self, config: MyConfig, taskWorker: BaseTaskWorker):
        super().__init__()
        init_logging(config)
        mp.set_start_method("spawn", force=True)
        self.config = config
        self.taskWorker: BaseTaskWorker = taskWorker

    @split_line(title="Load Dataset")
    def load_data(self):
        self.data_path = self.config.exp.data_path
        file_name = self.config.exp.data_file_name
        full_path = os.path.join(self.data_path, file_name)
        full_path_without_ext = os.path.splitext(full_path)[0]
        if os.path.exists(full_path_without_ext + ".pkl"):
            logging.info(
                f"Processed dataset exists. Loaded data from {full_path_without_ext + '.pkl'}"
            )
            with open(full_path_without_ext + ".pkl", "rb") as f:
                self.data: DataSet = pickle.load(f)
        else:
            logging.info(
                f"Processed dataset does not exist. Loaded data from {full_path} and saved to {full_path_without_ext + '.pkl'}"
            )
            self.data: DataSet = self._load_data(full_path)
            with open(full_path_without_ext + ".pkl", "wb") as f:
                pickle.dump(self.data, f)

        self._resume_process()
        # NOTE DEBUG
        self._debug(False)

        self.data.length = len(self.data.items)

    @split_line(title="Execution")
    def execute(self):
        max_workers = self.config.exp.max_workers
        max_inflight = self.config.exp.max_inflight
        max_inflight = max(max_inflight, max_workers)

        taskRunner = TaskRunner(self.config, self.data.items, self.taskWorker)
        taskRunner.run()

    @split_line(title="Evaluation")
    def evaluate(self):
        agent_class: BaseAgent = get_agent(self.config.agent.agent)
        agent_class.evaluate_all(self.config)

    @abstractmethod
    def _load_data(self, full_path: str) -> DataSet:
        """
        Load data for the task.
        """
        pass

    def _resume_process(self):
        qids = set()
        results_path = os.path.join(self.config.exp.output_path, "results.csv")
        if os.path.exists(results_path):
            # 读取csv文件
            with open(results_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 记录已有questionID
                    qids.add(row["questionID"])

        # 过滤掉已有的questionID
        new_items: List[DataItem] = []
        for item in self.data.items:
            new_qaList = []
            for qa in item.qaList:
                if qa.questionID not in qids:
                    new_qaList.append(qa)
            if new_qaList:
                item.qaList = new_qaList
                new_items.append(item)
        self.data.items = new_items

    def _debug(self, debug: bool = True):
        if debug:
            # NOTE DEBUG 取一部分数据
            self.data.items = self.data.items[:3]

            # NOTE DEBUG 只取前7个history
            for data_item in self.data.items:
                data_item.history.historyItems = data_item.history.historyItems[:3]

            # NOTE DEBUG 每个item只取前5个QA
            for data_item in self.data.items:
                data_item.qaList = data_item.qaList[:5]

                # NOTE DEBUG 只看部分QA
                # tempList = []
                # for qa in data_item.qaList:
                #     if qa.questionID == "conv-44_q56":
                #         tempList.append(qa)
                # data_item.qaList = tempList

            # NOTE DEBUG 只看一下dccbc061任务
            # temp_data_items = []
            # for data_item in self.data.items:
            #     if data_item.tid == "dccbc061":
            #         temp_data_items.append(data_item)
            # if temp_data_items:
            #     self.data.items = temp_data_items

