# scheduling.py (简化版)
import random


class Task:
    def __init__(self, task_id, position, task_type):
        self.id = task_id
        self.position = position
        self.task_type = task_type

    def __str__(self):
        return f"Task({self.id}: {self.position} {self.task_type})"

    def __repr__(self):
        return self.__str__()


class SimpleScheduler:
    """简化调度器 - 直接分配任务"""

    def __init__(self):
        self.task_queue = []

    def add_task(self, task):
        """添加任务到队列"""
        self.task_queue.append(task)

    def get_next_task(self):
        """获取下一个任务"""
        if self.task_queue:
            return self.task_queue.pop(0)
        return None

    def get_all_tasks(self):
        """获取所有任务"""
        return self.task_queue.copy()

    def clear_tasks(self):
        """清空任务队列"""
        self.task_queue.clear()