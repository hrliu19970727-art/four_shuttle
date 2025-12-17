# shared_state.py
import threading

state_lock = threading.RLock()

shared_state = {
    "shuttles": [],
    "release_tasks": [],
    "pick_tasks": [],
    "time": 0,
    "done": False,
    "slots": None,
    "simulation_started": False,

    # 任务统计
    "completed_tasks": 0,
    "failed_tasks": 0,
    "completed_release_tasks": 0,
    "completed_pick_tasks": 0,

    # 算法性能统计 (新增)
    "total_planning_time": 0.0,  # 总耗时(秒)
    "planning_count": 0,  # 规划次数
    "total_path_length": 0,  # 总路径步数
    "path_count": 0  # 成功路径数
}