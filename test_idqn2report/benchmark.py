# benchmark.py
import os

# --- 修复 OpenMP 冲突 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from batch_scheduler import BatchScheduler
from simulation_core import run_comparison_sim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


def plot_results(df):
    """
    根据仿真结果绘制对比图表并保存
    """
    # --- 1. 配置中文字体 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_dir, '..', 'fonts', 'simhei.ttf')

    my_font = None
    if os.path.exists(font_path):
        try:
            my_font = fm.FontProperties(fname=font_path)
            print(f"成功加载字体: {font_path}")
        except:
            print("字体加载失败，将使用默认字体")
    else:
        print(f"未找到字体文件: {font_path}，尝试使用系统默认")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # --- 2. 准备数据 ---
    algorithms = df['Algorithm'].tolist()
    time_sim = df['Time_Sim'].tolist()
    path_len = df['Avg_Path_Len'].tolist()
    plan_time = df['Avg_Plan_Time_ms'].tolist()
    completed = df['Completed'].tolist()

    x = np.arange(len(algorithms))
    width = 0.5

    # --- 3. 创建画布 (2x2 子图) ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('四向穿梭车调度算法效能对比 (DQN vs FIFO)', fontsize=16, fontproperties=my_font)

    colors = ['#FF9999', '#66B2FF']

    # 子图1: 总耗时
    bars1 = axs[0, 0].bar(x, time_sim, width, color=colors)
    axs[0, 0].set_title('总作业耗时 (秒)', fontproperties=my_font)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(algorithms, fontproperties=my_font)
    for bar in bars1:
        height = bar.get_height()
        axs[0, 0].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # 子图2: 平均路径
    bars2 = axs[0, 1].bar(x, path_len, width, color=colors)
    axs[0, 1].set_title('平均任务路径长度 (步)', fontproperties=my_font)
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(algorithms, fontproperties=my_font)
    for bar in bars2:
        height = bar.get_height()
        axs[0, 1].annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # 子图3: 规划耗时
    bars3 = axs[1, 0].bar(x, plan_time, width, color=colors)
    axs[1, 0].set_title('单次规划耗时 (ms)', fontproperties=my_font)
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(algorithms, fontproperties=my_font)
    for bar in bars3:
        height = bar.get_height()
        axs[1, 0].annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # 子图4: 完成数
    bars4 = axs[1, 1].bar(x, completed, width, color=colors)
    axs[1, 1].set_title('任务完成数量', fontproperties=my_font)
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(algorithms, fontproperties=my_font)
    for bar in bars4:
        height = bar.get_height()
        axs[1, 1].annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = 'benchmark_charts.png'
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 图表已生成并保存至: {save_path}")

    try:
        plt.show()
    except:
        pass


def main():
    # 任务量设为 50，确保能快速跑完出结果
    WAVE_SIZE = 50
    scheduler = BatchScheduler(wave_size=WAVE_SIZE)

    results = []

    scenarios = [
        ("IMPROVED_A*", "Without DQN (FIFO)"),
        ("IDQN", "With DQN (Smart)")
    ]

    print(f"\n>>> 开始算法效能对比测试 (任务量: {WAVE_SIZE}) <<<")

    for algo_key, display_name in scenarios:
        tasks = scheduler.get_tasks()

        print(f"\n------------------------------------------------")
        print(f"正在运行场景: {display_name}")
        print(f"------------------------------------------------")

        res = run_comparison_sim(algo_key, tasks)

        res["Algorithm"] = display_name
        results.append(res)

        print(f"✅ 测试结束: 总耗时 {res['Time_Sim']:.2f}s")

    df = pd.DataFrame(results)

    if len(df) == 2:
        t_fifo = df.loc[df['Algorithm'] == "Without DQN (FIFO)", 'Time_Sim'].values[0]
        t_dqn = df.loc[df['Algorithm'] == "With DQN (Smart)", 'Time_Sim'].values[0]
        improvement = ((t_fifo - t_dqn) / t_fifo) * 100
        print(f"\n[结论] 加入 DQN 后，总作业时间缩短了 {improvement:.2f}%")

    print("\n" + "=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    df.to_csv("benchmark_dqn_vs_fifo.csv", index=False)

    print("\n>>> 正在生成对比图表...")
    plot_results(df)


if __name__ == "__main__":
    main()