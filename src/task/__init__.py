from .basetask import BaseTask
from .longmemeval import LongMemEvalTask
from .locomo import LocomoTask

task_registry = {
    "longmemeval": LongMemEvalTask,
    "locomo": LocomoTask,
}

__all__ = ["LongMemEvalTask", "LocomoTask"]

def get_task(task_name: str) -> BaseTask:
    task = task_registry.get(task_name)
    if not task:
        raise ValueError(f"Unknown task: {task_name}")
    return task