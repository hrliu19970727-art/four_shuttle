## config.py
1、配置地图尺寸：20*46  
2、设置screen像素大小  
3、配置巷道的行列：区分主巷道和货位  
4、定义颜色  
   主巷道 - 浅蓝  "alley": (173, 216, 230),  
   货位巷道 - 青色  "aisle": (0, 128, 128),   
   占用货位 - 金色 "occupied": (255, 215, 0),  
   穿梭车 - 四种颜色 "shuttle_colors": [
        (255, 0, 0),   # 红色穿梭车1
        (0, 255, 0),   # 绿色穿梭车2
        (0, 0, 255),   # 蓝色穿梭车3
        (255, 255, 0), # 黄色穿梭车4
    提升机：淡黄色 "elevator_color": (255, 255, 0)
    目标点颜色 "target_color": (0, 0, 0)

5、日志文件  
## map.py
1、定义类（仓库地图） class WarehouseMap:
  def __init__(self):  初始化地图-指定某（行列）为主巷道、货位巷道、提升机、障碍物、目标点位置
  def _initialize_map(self):
  def add_obstacle(self, row, col):"""标记已占用货位"""
  def add_elevator(self, row, col):"""标记提升机位置"""
  
def add_target(self, row, col):"""标记目标点位置"""
  
  def add_shuttle(self, shuttle_id, row, col):"""标记穿梭车位置"""
  def clear_position(self, row, col): """清除标记，恢复巷道和货位"""
  def draw(self, screen):  绘制地图
  def get_color(self, val):·获取颜色·

显示地图-显示仓库地图
   0：主巷道 alley
   1：货位巷道 aisle
   2：障碍物(占用) occupied
   3、4、5、6：穿梭车
   7：提升机 elevator
   8：目标点 target

## visualization.py
def run_visualization(warehouse, shuttles_state):启动可视化
def draw_info_panel(screen, shuttles): 绘制信息面板


1、初始化共享状态（穿梭车位置、状态，货位状态，任务列表，时间，任务完成情况）
初始化仿真窗口和时钟。
进入一个主循环，不断处理用户事件（例如关闭窗口）。
每次循环中，清空屏幕，重新绘制仓库状态和信息面板，并更新屏幕显示。
检查任务是否完成，如果完成则等待 2 秒钟后退出循环。
退出循环后，保存任务日志，并关闭 pygame 以释放资源。

enumerate() 函数用于将可迭代对象（如列表、元组、字符串）组合为索引-元素对。
![img.png](img.png)
           
## main.py
1、初始化共享状态（穿梭车位置、状态，货位状态，任务列表，时间，任务完成情况）
2、初始化线程锁
3、初始化日志
4、生成随机任务 def generate_tasks(n=24):
5、记录任务完成日志 def log_task_completion(task_type, position, start_time, end_time):
6、移动小车并更新地图 def move_shuttle(env, warehouse, shuttle_id, path):
7、控制小车逻辑 def shuttle_controller(env, shuttle_id):
8、启动仿真 def run_simulation():
   env = simpy.Environment(): 创建一个 simpy.Environment 对象并赋值给 env。
   tasks = generate_tasks(24): 调用 generate_tasks 函数生成 24 个随机任务，并将这些任务存储在 tasks 列表中。
   scheduler = GeneticScheduler(population_size=50, generations=100): 创建一个 GeneticScheduler 类的对象并赋值给 scheduler，
      同时设置遗传算法的种群大小和迭代次数。
   scheduler.initialize_population(tasks): 调用 GeneticScheduler 类的 initialize_population 方法，将之前生成的任务列表传递给调度器，
      用于初始化遗传算法的种群。这个方法可能用于计算初始调度方案，并将其作为种群的组成部分。