import os
import random
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader_v4 import TrainDataset
from Main import MainModel

# === 基础设置 ===
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True        # Ampere TF32 tensor core
torch.backends.cudnn.allow_tf32 = True               # cuDNN TF32
torch.set_float32_matmul_precision('high')            # 允许 TF32 matmul
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(0)}")

# bf16 支持检测（Ampere+ 原生支持，免 GradScaler）
USE_BF16 = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16
print(f"AMP dtype: {'bfloat16' if USE_BF16 else 'float16'}"
      f"{' (no GradScaler needed)' if USE_BF16 else ''}")

data_path = "./merging/BEVData"
if not os.path.exists(data_path):
    print(f"ERROR: cannot find folder {data_path}")
    exit()

all_indices = list(range(len(os.listdir(data_path))))
random.shuffle(all_indices)

# ================= 8:1:1 训练/验证/测试划分 =================
n_total = len(all_indices)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
n_test = n_total - n_train - n_val

train_indices = all_indices[:n_train]
val_indices = all_indices[n_train:n_train + n_val]
test_indices = all_indices[n_train + n_val:]

print(f"Total: {n_total} | Train: {len(train_indices)} | Val: {len(val_indices)} | Test: {len(test_indices)}")

# ================= 配置区 =================
BATCH_SIZE = 64
EPOCHS = 150
VAL_EVERY = 5   # 每 N 个 epoch 验证一次（加速训练）

train_dataset = TrainDataset(train_indices)
val_dataset = TrainDataset(val_indices)
test_dataset = TrainDataset(test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True
)

# === 模型初始化 ===
model = MainModel(
    d_head=4, d_model=64, d_hid=128, in_feature=6, img_size=72,
    crop_h=72, crop_w=72, whole_scale=72*72, d_out=16*64
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
)

# === 梯度裁剪 (防止 Transformer 梯度爆炸) ===
GRAD_CLIP = 1.0

# === BEV-V2X Loss 函数 (论文 Eq.9) ===
# loss = λ_V * Σ_t [IoU_t(P_veh, O_veh) + BCE_t(P_veh, O_veh)]
#      + λ_M * Σ_t [IoU_t(P_map, O_map) + BCE_t(P_map, O_map)]
# λ_V = 1.0 (车辆主任务), λ_M = 0.03 (地图辅助任务)
class BEVV2XLoss(nn.Module):
    def __init__(self, lambda_v=1.0, lambda_m=0.03, veh_pos_weight=20.0, smooth=1e-5):
        super().__init__()
        self.lambda_v = lambda_v
        self.lambda_m = lambda_m
        self.smooth = smooth
        self.register_buffer('veh_pw', torch.tensor([veh_pos_weight]))

    @staticmethod
    def soft_iou_loss(pred, target, smooth=1e-5):
        """可微 IoU Loss: 1 - (|A∩B| + ε) / (|A∪B| + ε)"""
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
        iou = (intersection + smooth) / (union + smooth)
        return (1 - iou).mean()

    @staticmethod
    def hard_iou(pred, target, smooth=1e-5):
        """Hard IoU (用于评估, >0.5 阈值)"""
        pred_bin = (pred > 0.5).float()
        intersection = (pred_bin * target).sum(dim=(1, 2))
        union = pred_bin.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
        return ((intersection + smooth) / (union + smooth)).mean()

    def forward(self, pred, target):
        # 1. 地图辅助 Loss (ch 0-1: drivable, road_line)
        pred_map = pred[:, :2, :, :]
        target_map = target[:, :2, :, :]
        map_loss = self.soft_iou_loss(pred_map, target_map, self.smooth) \
                 + F.binary_cross_entropy(pred_map, target_map, reduction='mean')

        # 2. 车辆主 Loss (ch 2-4: T+1, T+2, T+3) — 逐帧计算，pos_weight 对抗稀疏性
        veh_loss = 0.0
        for t in range(3):
            pred_veh_t = pred[:, 2 + t:3 + t, :, :]
            target_veh_t = target[:, 2 + t:3 + t, :, :]
            veh_loss += self.soft_iou_loss(pred_veh_t, target_veh_t, self.smooth)
            veh_loss += F.binary_cross_entropy(
                pred_veh_t, target_veh_t,
                pos_weight=self.veh_pw, reduction='mean')

        return self.lambda_m * map_loss + self.lambda_v * veh_loss


criterion = BEVV2XLoss(lambda_v=1.0, lambda_m=0.03, veh_pos_weight=20.0)

# === L1 正则化 (论文 λ_R = 1e-6) ===
# 使用梯度方式注入，比 loss 项方式性能高得多（避免每 batch 建计算图）
LAMBDA_R = 1e-6


def apply_l1_gradient(model, lr_lambda):
    """在 backward 后、step 前直接修改梯度: grad += λ * sign(w)"""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.add_(lr_lambda * torch.sign(p))


# ================= 训练 =================
print(f"\n=== BEV-V2X Training ({EPOCHS} epochs) ===")
print(f"Loss: IoU+BCE (vehicle λ=1.0) + IoU+BCE (map λ=0.03)")
print(f"L1 reg: λ_R={LAMBDA_R} | Grad clip: {GRAD_CLIP} | LR scheduler: ReduceLROnPlateau(factor=0.5, patience=10)")

best_val_loss = float('inf')
best_epoch = 0

for epoch in range(EPOCHS):
    # ---- 训练阶段 ----
    model.train()
    total_train_loss = 0

    for batch_idx, (x, mask, label) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        map_input = label[:, :2, :, :]
        mask_input = mask[:, -1:, :, :].repeat(1, 4, 1, 1)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
            output, _ = model(x, mask, map_input)

        base_loss = criterion(output.float(), label.float())
        loss = base_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        apply_l1_gradient(model, LAMBDA_R)
        optimizer.step()

        total_train_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  [Train] Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)

    # ---- 验证阶段 (每隔 VAL_EVERY 个 epoch) ----
    do_val = ((epoch + 1) % VAL_EVERY == 0) or (epoch == 0)
    if do_val:
        model.eval()
        total_val_loss = 0

        with torch.inference_mode():
            for x, mask, label in val_loader:
                x = x.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                map_input = label[:, :2, :, :]
                mask_input = mask[:, -1:, :, :].repeat(1, 4, 1, 1)

                with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
                    output, _ = model(x, mask, map_input)

                base_loss = criterion(output.float(), label.float())
                total_val_loss += base_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"=== Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ===")

        # 保存最佳模型 (基于验证集 loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, "merging_model_best.pth")
            print(f"  >>> Best model saved (val_loss: {best_val_loss:.4f})")

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  LR: {current_lr:.2e}")
    else:
        print(f"=== Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | (val skipped) ===")

    # 定期 checkpoint
    if (epoch + 1) % 20 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, f"merging_model_epoch_{epoch+1}.pth")
        print(f"  Checkpoint saved: merging_model_epoch_{epoch+1}.pth")

print(f"\n=== Training complete. Best model: epoch {best_epoch} (val_loss: {best_val_loss:.4f}) ===")

# ================= 测试集评估 =================
print("\n=== Test Set Evaluation ===")

# 加载最佳模型
if os.path.exists("merging_model_best.pth"):
    checkpoint = torch.load("merging_model_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")
else:
    print("WARNING: best model not found, using current model weights")

model.eval()

test_iou_all = []
test_bce_all = []

with torch.no_grad():
    for x, mask, label in test_loader:
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        map_input = label[:, :2, :, :]
        mask_input = mask[:, -1:, :, :].repeat(1, 4, 1, 1)

        with torch.amp.autocast('cuda', dtype=AMP_DTYPE):
            output, _ = model(x, mask, map_input)

        # 逐帧 hard IoU (>0.5 阈值) 和 BCE
        for t in range(3):
            pred_veh = output[:, 2 + t, :, :]
            label_veh = label[:, 2 + t, :, :]

            iou_val = BEVV2XLoss.hard_iou(pred_veh, label_veh)
            bce_val = F.binary_cross_entropy(pred_veh, label_veh, reduction='mean')

            test_iou_all.append(iou_val.item())
            test_bce_all.append(bce_val.item())

# 汇总
avg_iou = sum(test_iou_all) / len(test_iou_all) * 100
iou_t1 = sum(test_iou_all[0::3]) / (len(test_iou_all) // 3) * 100
iou_t2 = sum(test_iou_all[1::3]) / (len(test_iou_all) // 3) * 100
iou_t3 = sum(test_iou_all[2::3]) / (len(test_iou_all) // 3) * 100
avg_bce = sum(test_bce_all) / len(test_bce_all)

print(f"Test IoU  (avg):  {avg_iou:.1f}%")
print(f"Test IoU  (T+1):  {iou_t1:.1f}%")
print(f"Test IoU  (T+2):  {iou_t2:.1f}%")
print(f"Test IoU  (T+3):  {iou_t3:.1f}%")
print(f"Test BCE  (avg):  {avg_bce:.4f}")
