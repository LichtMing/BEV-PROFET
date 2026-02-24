import torch
import numpy as np
import os
from Main import MainModel
from data_loader_v4 import TrainDataset

# ================= 配置区 =================
MODEL_PATH = "model_turbo_epoch_80.pth"
SAVE_DIR = "./All_Predictions_Dataset"
MAX_SAMPLES = None  # 想跑全部就改成 None
VEHICLE_THRESHOLD = 0.3  # 车辆二值化阈值，与 inference.py 保持一致
# ==========================================


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 1. 加载数据
    data_path = "./BEVData_01"
    if not os.path.exists(data_path):
        print(f"Data directory not found: {data_path}")
        return

    total_files = len(os.listdir(data_path))

    if MAX_SAMPLES is None:
        process_indices = range(total_files)
        print(f">>> Processing all {total_files} samples...")
    else:
        process_indices = range(min(MAX_SAMPLES, total_files))
        print(f">>> Processing first {len(process_indices)} samples...")

    dataset = TrainDataset(list(process_indices))

    # 2. 加载模型
    print(f"Loading model: {MODEL_PATH} ...")
    model = MainModel(
        d_head=4, d_model=64, d_hid=128, in_feature=6, img_size=72,
        crop_h=72, crop_w=72, whole_scale=72*72, d_out=16*64
    ).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print(f"Model file not found: {MODEL_PATH}")
        return

    print("=== Start saving predictions as dataset ===")

    with torch.no_grad():
        for i in range(len(dataset)):
            # 1. 获取原始文件名 (例如 0_31000)
            raw_filename = dataset.train_file_list[i]
            base_name = os.path.splitext(raw_filename)[0]

            # 创建专属子文件夹: ./All_Predictions_Dataset/0_31000/
            sample_dir = os.path.join(SAVE_DIR, base_name)
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)

            # 2. 获取数据并预测
            x, mask, label = dataset[i]

            x = x.unsqueeze(0).to(device)
            mask = mask.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            map_input = label[:, :2, :, :]
            mask_input = mask[:, -1:, :, :].repeat(1, 4, 1, 1)

            output, _ = model(x, mask, map_input)

            pred_np = output[0].cpu().numpy()   # (5, 288, 288)
            gt_np = label[0].cpu().numpy()       # (5, H, W)
            map_layers = gt_np[:2]               # (2, H, W) 道路层

            # 3. 逐帧保存预测与 GT
            for frame_idx, t_name in enumerate(["T+1", "T+2", "T+3"]):
                channel_idx = 2 + frame_idx

                # 预测结果：应用阈值二值化，与 inference.py 的 draw_style 一致
                pred_vehicle = (pred_np[channel_idx] > VEHICLE_THRESHOLD).astype(np.uint8)

                # GT 
                gt_vehicle = gt_np[channel_idx]

                # 保存为 .npy，文件名与原 inference.py 的 png 命名对齐
                np.save(os.path.join(sample_dir, f"pred_{t_name}.npy"), pred_vehicle)
                np.save(os.path.join(sample_dir, f"gt_{t_name}.npy"), gt_vehicle)

            # 4. 额外保存道路地图层（每个 sample 只需一份）
            np.save(os.path.join(sample_dir, "map.npy"), map_layers)

            if i % 10 == 0:
                print(f"[{i+1}/{len(dataset)}] Saved: {base_name} -> {sample_dir}")

    print(f"\nDone! Dataset saved to: {SAVE_DIR}")
    print(f"Structure:")
    print(f"  {SAVE_DIR}/")
    print(f"    <sample_id>/")
    print(f"      pred_T+1.npy  - prediction (uint8, H x W)")
    print(f"      pred_T+2.npy")
    print(f"      pred_T+3.npy")
    print(f"      gt_T+1.npy    - ground truth (float32, H x W)")
    print(f"      gt_T+2.npy")
    print(f"      gt_T+3.npy")
    print(f"      map.npy       - road map (float32, 2 x H x W)")


if __name__ == "__main__":
    main()
