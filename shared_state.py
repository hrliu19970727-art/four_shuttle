# shared_state.py
import threading

# 初始化线程锁对象
state_lock = threading.Lock()

# 全局共享状态
shared_state = {
    "shuttles": [],        # 穿梭车状态列表
    "release_tasks": [],   # 放货任务队列
    "pick_tasks": [],      # 取货任务队列
    "time": 0,             # 仿真时间
    "done": False,         # 仿真结束标志
    "slots": None          # 货架状态，在main.py中初始化
}