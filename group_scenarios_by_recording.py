#!/usr/bin/env python3
"""
将 EthicalTrajectoryPlanning/scenarios/ 中的 XML 场景文件按照来源
recording (vehicle_tracks_00k.csv) 分组到子文件夹 track_000/ ~ track_007/ 中。

本脚本复现 interaction_to_cr.py 中的转换循环逻辑:
  - 依次读取每个 vehicle_tracks CSV 的时间范围
  - 计算每个 CSV 对应的 segment 数量
  - 根据 segment 累加确定 scenario ID → recording k 的映射
  - 将对应的 XML 文件拷贝到子文件夹中

原始 XML 文件不会被移动或删除。

用法:
    conda activate traj_env
    python group_scenarios_by_recording.py
"""

import os
import sys
import glob
import shutil
import pandas as pd


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ─── 路径配置 ───
    data_dir = os.path.join(
        base_dir, "dataset_convert",
        "INTERACTION-Dataset-DR-v1_1",
        "recorded_trackfiles",
        "DR_USA_Intersection_EP0"
    )
    scenarios_dir = os.path.join(base_dir, "EthicalTrajectoryPlanning", "scenarios")

    # ─── 转换参数（与 interaction_to_cr.py / config.yaml 一致）───
    dt = 0.1
    scenario_time_steps = 150     # num_time_steps_scenario
    location = "USA_Intersection-1"

    # ─── 读取 CSV 并计算每个 recording 的 segment 数 ───
    csv_files = sorted(glob.glob(os.path.join(data_dir, "vehicle_tracks_*.csv")))
    if not csv_files:
        print(f"错误: 未在 {data_dir} 找到 vehicle_tracks CSV 文件")
        sys.exit(1)

    print(f"找到 {len(csv_files)} 个 recording 文件")
    print(f"场景目录: {scenarios_dir}")
    print()

    # 计算每个 recording 的 segment 数量（复现转换器逻辑）
    segments_per_recording = []
    for k, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path, header=0)
        # 与转换器相同的时间戳转换: raw_ms → CR time step
        df["cr_step"] = (df["timestamp_ms"] / 1000.0 // dt).astype(int)
        t_min = df.cr_step.min()
        t_max = df.cr_step.max()
        num_segments = int((t_max - t_min) / scenario_time_steps)
        segments_per_recording.append(num_segments)
        print(f"  k={k}: {os.path.basename(csv_path)}, "
              f"CR步 [{t_min}, {t_max}], segments={num_segments}")

    print(f"\n  总 segments: {sum(segments_per_recording)}")

    # ─── 构建 scenario_id → recording_k 映射 ───
    # 转换器中 id_config_scenario 依次递增（跨越所有 recording），
    # 每个 segment 消耗一个 ID（包括因无车辆而跳过的 segment）
    mapping = {}   # {scenario_id: k}
    id_cursor = 1  # 起始 ID

    for k, num_seg in enumerate(segments_per_recording):
        for _ in range(num_seg):
            mapping[id_cursor] = k
            id_cursor += 1

    # ─── 验证映射是否与现有文件一致 ───
    existing_ids = set()
    for f in os.listdir(scenarios_dir):
        if (f.startswith(f"{location}_") and f.endswith("_T-1.xml")
                and os.path.isfile(os.path.join(scenarios_dir, f))):
            try:
                sid = int(f.replace(f"{location}_", "").replace("_T-1.xml", ""))
                existing_ids.add(sid)
            except ValueError:
                pass

    mapped_ids = set(mapping.keys())
    unmapped = existing_ids - mapped_ids
    missing = mapped_ids - existing_ids

    print(f"\n验证:")
    print(f"  现有 XML 文件: {len(existing_ids)}")
    print(f"  映射 ID 总数: {len(mapped_ids)}")
    print(f"  缺口(有ID无文件): {sorted(missing)} ({len(missing)} 个)")

    if unmapped:
        print(f"  ⚠ 存在未映射的文件: {sorted(unmapped)}")

    # ─── 创建子文件夹并拷贝文件 ───
    print(f"\n创建子文件夹并拷贝文件...")
    total_copied = 0

    for k in range(len(csv_files)):
        folder_name = f"track_{k:03d}"
        folder_path = os.path.join(scenarios_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # 该 recording 对应的所有 scenario IDs
        sids = sorted(sid for sid, rec_k in mapping.items() if rec_k == k)
        id_min, id_max = sids[0], sids[-1]

        copied = 0
        for sid in sids:
            xml_name = f"{location}_{sid}_T-1.xml"
            src = os.path.join(scenarios_dir, xml_name)
            dst = os.path.join(folder_path, xml_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied += 1

        total_copied += copied
        print(f"  {folder_name}/: {copied} 个文件 "
              f"(ID 范围: {id_min}-{id_max}, 共 {len(sids)} 个 segment)")

    print(f"\n完成! 共拷贝 {total_copied} 个文件到 {len(csv_files)} 个子文件夹")


if __name__ == "__main__":
    main()
