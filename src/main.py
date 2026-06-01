import os
from dotenv import load_dotenv

load_dotenv()

import sys
from task import get_task
from task import BaseTask
from components.config import MyConfig, load_config


if __name__ == "__main__":
    
    config: MyConfig = load_config(sys.argv)
    
    task: BaseTask = get_task(config.exp.task)(config)
    task.load_data()
    task.execute()
    task.evaluate()
