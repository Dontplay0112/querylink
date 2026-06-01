from functools import wraps
import os
from tqdm import tqdm
    
# def example(arg1, arg2):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             # ...
#             result = func(*args, **kwargs)
#             # ...
#             return result
#         return wrapper
#     return decorator

def split_line(title: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            terminal_width = os.get_terminal_size().columns
            centered_title = title.center(terminal_width, '=')
            tqdm.write(f"\033[38;2;57;197;187m{centered_title}\033[0m")
            result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator