import torch
import numpy as np
import cv2
import os
import sys
from Main import MainModel
from data_loader_v4 import TrainDataset

# ================= 配置区 =================
MODEL_PATH = "model_turbo_epoch_80.pth" 
SAVE_DIR = "./All_Predictions_Result"
MAX_SAMPLES = 10  # 想跑全部就改成 None
# ==========================================

# 配色方案
COLORS = {
    "background": [255, 255, 255], 
    "drivable":   [220, 220, 220], 
    "road_line":  [192, 192, 192], 
    "vehicle":    [225, 105, 65]   
}

def draw_style(map_data, vehicle_data):
    h, w = vehicle_data.shape
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    canvas[map_data[0] == 1] = COLORS["drivable"]
    canvas[map_data[1] == 1] = COLORS["road_line"]
    
    # 阈值 + 腐蚀 (防粘连)
    mask = (vehicle_data > 0.3).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask_eroded = cv2.erode(mask, kernel, iterations=1)
    
    canvas[mask_eroded == 1] = COLORS["vehicle"]
    return canvas

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 正在使用: {device} 进行预测...")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    # 1. 加载数据
    data_path = "./BEVData_01"
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件夹 {data_path}")
        return

    total_files = len(os.listdir(data_path))
    
    if MAX_SAMPLES is None:
        process_indices = range(total_files)
        print(f">>> 准备处理所有 {total_files} 份数据...")
    else:
        process_indices = range(min(MAX_SAMPLES, total_files))
        print(f">>> 准备处理前 {len(process_indices)} 份数据...")

    dataset = TrainDataset(list(process_indices))

    # 2. 加载模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = MainModel(
        d_head=4, d_model=64, d_hid=128, in_feature=6, img_size=72, 
        crop_h=72, crop_w=72, whole_scale=72*72, d_out=16*64
    ).to(device)
    
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
    else:
        print("❌ 找不到模型文件！")
        return

    print("=== 开始整理相册 (二级目录模式) ===")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            # 1. 获取原始文件名 (例如 0_31000)
            raw_filename = dataset.train_file_list[i]
            base_name = os.path.splitext(raw_filename)[0]
            
            # === 关键修改：创建专属子文件夹 ===
            # 路径变成: ./All_Predictions_Result/0_31000/
            sample_dir = os.path.join(SAVE_DIR, base_name)
            
            # 如果这个子文件夹不存在，就创建一个
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
            
            pred_np = output[0].cpu().numpy()
            gt_np = label[0].cpu().numpy()
            map_layers = gt_np[:2]
            
            # 3. 保存图片到子文件夹
            for frame_idx, t_name in enumerate(["T+1", "T+2", "T+3"]):
                channel_idx = 2 + frame_idx
                
                img_gt = draw_style(map_layers, gt_np[channel_idx])
                img_pred = draw_style(map_layers, pred_np[channel_idx])
                combined = np.hstack((img_gt, img_pred))
                
                cv2.putText(combined, f"GT {t_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                cv2.putText(combined, f"Pred {t_name}", (288+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                
                # 文件名直接叫 "T+1.png"，因为已经在专属文件夹里了
                # 最终路径: .../All_Predictions_Result/0_31000/T+1.png
                filename = f"{t_name}.png"
                save_path = os.path.join(sample_dir, filename)
                
                cv2.imwrite(save_path, combined)
            
            if i % 10 == 0:
                print(f"已归档: {base_name} -> {sample_dir}")

    print(f"\n✅ 全部整理完毕！请打开 {SAVE_DIR} 查看您的二级相册！")

if __name__ == "__main__":
    main()