"""
inference_prob_map.py
--------------------
对所有样本做推理，将 Transformer 输出的未来三个时刻 (T+1, T+2, T+3) 的
**原始概率图** (sigmoid 输出, float32, 值域 [0,1]) 保存为 .npy 文件。

输出目录结构：
    BEVPredProb/                 ← 父文件夹
        0_103000/                ← 每个场景一个子文件夹
            T1.npy               (288, 288) float32  T+1 概率图
            T2.npy               (288, 288) float32  T+2 概率图
            T3.npy               (288, 288) float32  T+3 概率图
        0_104000/
            T1.npy
            T2.npy
            T3.npy
        ...
"""

import torch
import numpy as np
import os
from Main import MainModel
from data_loader_v4 import TrainDataset

# ================= 配置区 =================
MODEL_PATH = "model_turbo_epoch_80.pth"
OUTPUT_ROOT = "./BEVPredProb"          # 父文件夹
T_KEYS = ["T1", "T2", "T3"]           # 三个时刻的文件名前缀
MAX_SAMPLES = None  # 设为 None 处理全部样本
# ==========================================


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建输出根目录
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 1. 加载数据
    data_path = "./BEVData_01"
    if not os.path.exists(data_path):
        print(f"Data directory not found: {data_path}")
        return

    total_files = len(os.listdir(data_path))
    if MAX_SAMPLES is None:
        process_indices = list(range(total_files))
    else:
        process_indices = list(range(min(MAX_SAMPLES, total_files)))
    print(f">>> Will process {len(process_indices)} samples")

    dataset = TrainDataset(process_indices)

    # 2. 加载模型
    print(f"Loading model: {MODEL_PATH} ...")
    model = MainModel(
        d_head=4, d_model=64, d_hid=128, in_feature=6, img_size=72,
        crop_h=72, crop_w=72, whole_scale=72 * 72, d_out=16 * 64
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found: {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 3. 推理并保存概率图
    print("=== Start inference: saving probability maps ===")
    with torch.no_grad():
        for i in range(len(dataset)):
            # 获取原始文件名 (例如 0_103000.npy)
            raw_filename = dataset.train_file_list[i]
            base_name = os.path.splitext(raw_filename)[0]  # "0_103000"

            # 获取数据
            x, mask, label = dataset[i]
            x = x.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)

            map_input = label[:, :2, :, :]  # 道路层作为输入

            # 模型推理 → output shape: (1, 5, 288, 288)
            output, _ = model(x, mask, map_input)
            pred_np = output[0].cpu().numpy()  # (5, 288, 288)

            # 为当前场景创建子文件夹，保存三个时刻的概率图
            # channel 2 → T+1, channel 3 → T+2, channel 4 → T+3
            scene_dir = os.path.join(OUTPUT_ROOT, base_name)
            os.makedirs(scene_dir, exist_ok=True)

            for t_idx, t_key in enumerate(T_KEYS):
                prob_map = pred_np[2 + t_idx].astype(np.float32)  # (288, 288)
                save_path = os.path.join(scene_dir, f"{t_key}.npy")
                np.save(save_path, prob_map)

            if i % 50 == 0:
                print(f"  [{i + 1}/{len(dataset)}] saved: {base_name}")

    print(f"\nDone! Probability maps saved.")
    n_scenes = len([d for d in os.listdir(OUTPUT_ROOT)
                    if os.path.isdir(os.path.join(OUTPUT_ROOT, d))])
    print(f"Output structure:")
    print(f"  {OUTPUT_ROOT}/  ({n_scenes} scenes, each containing T1.npy T2.npy T3.npy, (288, 288) float32)")


if __name__ == "__main__":
    main()
