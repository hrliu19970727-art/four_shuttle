# BatchScheduler/benchmark.py
from batch_scheduler import BatchScheduler
from simulation_core import run_comparison_sim  # 这里引用的是修改后的 main.py
import pandas as pd


def main():
    # 1. 生成 50 个固定任务
    scheduler = BatchScheduler(wave_size=10)

    results = []

    # 2. 定义对比组
    # STANDARD_A*: 只找最短路，不管拥堵和时间，遇到障碍就死等
    # IMPROVED_A*: 考虑时空路况，会绕路，会预测
    algorithms = ["STANDARD_A*", "IMPROVED_A*"]

    print("\n>>> 开始算法对比测试 <<<")

    for algo in algorithms:
        # 获取完全相同的任务副本
        tasks = scheduler.get_tasks()

        # 运行仿真
        res = run_comparison_sim(algo, tasks)
        results.append(res)
        print(f"✅ {algo} 测试结束")

    # 3. 输出报表
    df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("       ALGORITHM BENCHMARK REPORT       ")
    print("=" * 50)
    print(df.to_string(index=False))
    print("=" * 50)

    df.to_csv("benchmark_comparison.csv", index=False)


if __name__ == "__main__":
    main()