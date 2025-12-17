# benchmark.py
from batch_scheduler import BatchScheduler
from simulation_core import run_comparison_sim
import pandas as pd


def main():
    # 1. 初始化波次：生成 50 个任务
    # 这些任务会被保存下来，分别传给不同的算法，确保测试条件一致
    scheduler = BatchScheduler(wave_size=50)

    results = []

    # 2. 定义要对比的算法
    # 确保 simulation_core 中实现了对应逻辑
    algorithms = ["BFS", "A_STAR"]
    # 如果您配置好了 IDQN，也可以加上 "IDQN"

    print("\n>>> 开始多算法对比测试 <<<")

    for algo in algorithms:
        # 获取任务副本 (深拷贝，防止被修改)
        tasks = scheduler.get_tasks()

        # 运行仿真
        # 该函数会重置环境、运行仿真并返回统计字典
        res = run_comparison_sim(algo, tasks)
        results.append(res)
        print(f"✅ {algo} 完成: 耗时 {res['Time_Sim']:.2f}s, 完成任务 {res['Completed']}")

    # 3. 输出报表
    df = pd.DataFrame(results)

    print("\n" + "=" * 40)
    print("       ALGORITHM BENCHMARK REPORT       ")
    print("=" * 40)
    print(df.to_string(index=False))
    print("=" * 40)

    # 保存结果
    df.to_csv("benchmark_result.csv", index=False)
    print("结果已保存至 benchmark_result.csv")


if __name__ == "__main__":
    main()