# scheduling.py
class Task:
    def __init__(self, task_id, position, task_type):
        self.id = task_id
        self.position = position
        self.task_type = task_type  # 'release' or 'pick'

    def __str__(self):
        return f"Task({self.id}: {self.position} {self.task_type})"

    def __repr__(self):
        return self.__str__()