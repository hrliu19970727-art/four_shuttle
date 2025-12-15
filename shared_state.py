# shared_state.py
import threading

# 使用 RLock 防止死锁
state_lock = threading.RLock()

# 全局共享状态
shared_state = {
    "shuttles": [],  # 小车列表
    "release_tasks": [],  # 放货队列
    "pick_tasks": [],  # 取货队列
    "time": 0,  # 仿真时间
    "done": False,  # 结束标志
    "slots": None,  # 货位状态矩阵
    "simulation_started": False,

    # 统计
    "completed_tasks": 0,
    "failed_tasks": 0,
    "completed_release_tasks": 0,  # 新增：入库完成数
    "completed_pick_tasks": 0  # 新增：出库完成数
}