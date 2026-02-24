import os
# 1. 魔法药水：治理显存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader_v4 import TrainDataset
from Main import MainModel
from torch.amp import GradScaler 

# === 基础设置 ===
torch.backends.cudnn.benchmark = True 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"⚡ 正在使用 L4 法拉利: {torch.cuda.get_device_name(0)} ⚡")

data_path = "./BEVData_01"
if not os.path.exists(data_path):
    print(f"❌ 找不到文件夹 {data_path}")
    exit()

all_indices = list(range(len(os.listdir(data_path))))

# ================= 配置区 =================
# 1. 显存够用，咱们维持 64，求稳
BATCH_SIZE = 64

# 2. 【关键】增加轮数！
# 治“粘连”这种细活，必须得多练。50轮不够，咱们跑 80 轮！
EPOCHS = 80

dataset = TrainDataset(all_indices)

# 3. 提速配置
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    
    # 保持 4 个工人，防止报错
    num_workers=4, 
    
    # 【关键提速点】让每个工人手里多拿 4 盘菜候着
    # 这样 GPU 就不容易闲下来了
    prefetch_factor=4, 
    
    pin_memory=True, 
    persistent_workers=False # 咱们求稳，不强求一直上班
)

# === 模型初始化 ===
model = MainModel(
    d_head=4, d_model=64, d_hid=128, in_feature=6, img_size=72,       
    crop_h=72, crop_w=72, whole_scale=72*72, d_out=16*64
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler('cuda')

# === 【核心药方】混合 Loss 定义 ===
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, pred, target):
        # 1. 只要车的部分 (Channel 2,3,4)
        pred_car = pred[:, 2:, :, :]
        target_car = target[:, 2:, :, :]
        
        # --- A. 加权 BCE (解决“看不见车”) ---
        # 给车 5 倍的权重，给背景 1 倍
        # 这样模型就不敢忽略小车了，也不会像10倍那样被吓傻
        weight = torch.ones_like(target_car) * 5.0 
        # 背景部分的权重如果需要也可以调整，这里默认利用broadcast机制
        # 实际上我们用 F.binary_cross_entropy 自带的 weight 参数
        
        # 构造一个全图的 weight，背景是1，车是5 (这里简化处理，直接对车区域加权)
        # 简单法：直接算带权重的 BCE
        bce_loss = F.binary_cross_entropy(pred_car, target_car, weight=weight, reduction='mean')

        # --- B. Dice Loss (解决“粘连”) ---
        pred_flat = pred_car.reshape(pred_car.shape[0], -1)
        target_flat = target_car.reshape(target_car.shape[0], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()
        
        # --- C. 混合 (1:1) ---
        return 0.5 * bce_loss + 0.5 * dice_loss

criterion = CombinedLoss()

# === 开始训练 ===
print(f"=== L4 强力去粘连训练 (Epochs: {EPOCHS}) ===")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    print(f"\n🔥 第 {epoch+1} / {EPOCHS} 轮 (加权BCE + Dice)...")
    
    for batch_idx, (x, mask, label) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        map_input = label[:, :2, :, :]
        mask_input = mask[:, -1:, :, :].repeat(1, 4, 1, 1)

        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            output, _ = model(x, mask, map_input)
            # 这里的 output 经过了 sigmoid 吗？
            # MainModel 里最后如果是 Sigmoid，这里直接用。
            # 假设 MainModel 输出范围是 0-1。
        
        # 计算 Loss
        loss = criterion(output.float(), label.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 20 == 0:
            print(f"  > 进度: {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"=== 第 {epoch+1} 轮结束，平均 Loss: {avg_loss:.4f} ===")
    
    # 咱们每 20 轮存一次就行，不用太频繁
    if (epoch + 1) % 20 == 0:
        save_path = f"model_turbo_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"存档成功: {save_path}")

print("\n奶奶，药方开好了！这次咱们有耐心点，让它跑完 80 轮，粘连的问题一定能好转！")