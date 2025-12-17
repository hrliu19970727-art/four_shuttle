# benchmark.py
from batch_scheduler import BatchScheduler
from simulation_core import run_comparison_sim
import pandas as pd


def main():
    # 1. 初始化波次
    # 建议将任务量适当调大 (例如 100)，让 DQN 有机会展现“越学越聪明”的效果
    # 如果任务太少(如20个)，DQN 还没探索完随机策略，仿真就结束了，可能跑不过 FIFO
    WAVE_SIZE = 50
    scheduler = BatchScheduler(wave_size=WAVE_SIZE)

    results = []

    # 2. 定义对比组
    # 格式: (传给仿真的参数名, 报表中显示的名称)
    scenarios = [
        ("IMPROVED_A*", "Without DQN (FIFO)"),  # 对照组：无脑排队
        ("IDQN", "With DQN (Smart)")  # 实验组：智能挑选
    ]

    print(f"\n>>> 开始算法效能对比测试 (任务量: {WAVE_SIZE}) <<<")

    for algo_key, display_name in scenarios:
        # 获取完全相同的任务副本，确保公平
        tasks = scheduler.get_tasks()

        # 运行仿真
        print(f"\n------------------------------------------------")
        print(f"正在运行场景: {display_name}")
        print(f"------------------------------------------------")

        # 这里的 res 包含了该次仿真的统计数据
        res = run_comparison_sim(algo_key, tasks)

        # 修改算法名称为易读格式
        res["Algorithm"] = display_name
        results.append(res)

        print(f"✅ 测试结束: 总耗时 {res['Time_Sim']:.2f}s | 平均路径 {res['Avg_Path_Len']:.1f}步")

    # 3. 输出最终对比报表
    df = pd.DataFrame(results)

    # 计算提升率 (Time_Sim 越小越好)
    if len(df) == 2:
        t_fifo = df.loc[df['Algorithm'] == "Without DQN (FIFO)", 'Time_Sim'].values[0]
        t_dqn = df.loc[df['Algorithm'] == "With DQN (Smart)", 'Time_Sim'].values[0]
        improvement = ((t_fifo - t_dqn) / t_fifo) * 100
        print(f"\n[结论] 加入 DQN 后，总作业时间缩短了 {improvement:.2f}%")

    print("\n" + "=" * 60)
    print("            DQN ABLATION STUDY REPORT            ")
    print("=" * 60)
    # 调整列显示顺序
    cols = ["Algorithm", "Completed", "Time_Sim", "Avg_Path_Len", "Avg_Plan_Time_ms"]
    print(df[cols].to_string(index=False))
    print("=" * 60)

    # 保存结果
    df.to_csv("benchmark_dqn_vs_fifo.csv", index=False)
    print("详细数据已保存至 benchmark_dqn_vs_fifo.csv")


if __name__ == "__main__":
    main()