import random
import copy
from scheduling import Task
from map import warehouse
from config import ROWS, COLS, TYPE_STORAGE


class BatchScheduler:
    def __init__(self, wave_size):
        """
        静态波次调度器
        :param wave_size: 本次波次需要生成的任务总数
        """
        self.total_tasks = wave_size
        self.static_tasks = []  # 存储生成的静态任务列表
        self._prepare_static_wave()

    def _prepare_static_wave(self):
        """
        生成静态作业波次。
        不依赖 shared_state，而是基于地图结构随机生成一套可复用的任务。
        """
        print(f"[Scheduler] 正在初始化 {self.total_tasks} 个静态任务...")

        # 1. 扫描地图，区分出所有的存储位
        # 我们在生成任务时，临时模拟一个仓库状态（例如 30% 有货）
        # 这样可以同时生成“取货”和“放货”任务
        empties = []
        filled = []

        for r in range(ROWS):
            for c in range(COLS):
                # 只在存储类型的格子上生成任务
                if warehouse.grid_type[r][c] == TYPE_STORAGE:
                    # 假设初始有 30% 的概率是有货的 (用于生成 Pick 任务)
                    # 剩下的 70% 是空的 (用于生成 Release 任务)
                    if random.random() < 0.3:
                        filled.append((r, c))
                    else:
                        empties.append((r, c))

        # 打乱顺序，模拟真实的随机分布
        random.shuffle(empties)
        random.shuffle(filled)

        # 2. 生成具体任务
        for i in range(self.total_tasks):
            # 逻辑：如果模拟的“有货池”里还有货，就以 50% 概率生成取货任务
            # 否则生成入库任务，保持出入库的动态平衡
            is_release = False

            # 尝试生成出库 (Pick)
            if filled and (not empties or random.random() < 0.5):
                pos = filled.pop(0)  # 取出一个有货位置作为目标
                # Task 参数: (id, position, type)
                t = Task(i, pos, "pick")
                self.static_tasks.append(t)

                # 逻辑上这个位置被取走了，变为空位（供后续生成入库用，如果需要更长波次）
                # 这里简单起见，不再回收到 empties，避免单波次内重复操作同一位置

            # 尝试生成入库 (Release)
            elif empties:
                pos = empties.pop(0)  # 取出一个空位作为目标
                t = Task(i, pos, "release")
                self.static_tasks.append(t)

        # 统计生成结果
        pick_cnt = sum(1 for t in self.static_tasks if t.task_type == 'pick')
        release_cnt = sum(1 for t in self.static_tasks if t.task_type == 'release')
        print(f"[Scheduler] 任务清单生成完毕: 总计 {len(self.static_tasks)} (取货: {pick_cnt}, 放货: {release_cnt})")

    def get_tasks(self):
        """
        获取任务列表的深拷贝副本。
        每次仿真开始前调用此方法，可以获得一份全新的、未被修改的任务列表。
        """
        return copy.deepcopy(self.static_tasks)


# 简单的测试代码，直接运行此文件可查看生成效果
if __name__ == "__main__":
    scheduler = BatchScheduler(wave_size=100)
    tasks = scheduler.get_tasks()
    for t in tasks:
        print(f"Task ID: {t.id}, Type: {t.task_type}, Pos: {t.position}")