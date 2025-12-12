# shared_state.py
import threading

# 初始化线程锁对象
# 【关键修复】使用 RLock (可重入锁) 代替 Lock
# 这允许同一个线程（例如 SimPy 线程）在已持有锁的情况下再次获取锁，
# 从而避免在 execute_task_operation 调用 log_task_completion 时发生死锁。
state_lock = threading.RLock()

# 全局共享状态
shared_state = {
    "shuttles": [],  # 穿梭车状态列表
    "release_tasks": [],  # 放货任务队列
    "pick_tasks": [],  # 取货任务队列
    "time": 0,  # 仿真时间
    "done": False,  # 仿真结束标志
    "slots": None,  # 货架状态，在main.py中初始化

    # 统计信息
    "completed_tasks": 0,
    "failed_tasks": 0,
    "simulation_started": False
}
