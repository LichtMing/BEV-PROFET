import os
# Optional: reduce CUDA memory fragmentation
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_loader_v4 import TrainDataset
from Main import MainModel
from torch.cuda.amp import GradScaler

# === Basic Setup ===
torch.backends.cudnn.benchmark = True 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(0)}")

data_path = "./merging/BEVData"
if not os.path.exists(data_path):
    print(f"ERROR: data path not found: {data_path}")
    exit()

all_indices = list(range(len(os.listdir(data_path))))

# ================= Configuration =================
BATCH_SIZE = 64

EPOCHS = 150

dataset = TrainDataset(all_indices)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=False
)

# === Model Initialization ===
model = MainModel(
    d_head=4, d_model=64, d_hid=128, in_feature=6, img_size=72,       
    crop_h=72, crop_w=72, whole_scale=72*72, d_out=16*64
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()

# === Combined Loss: Weighted BCE + Dice ===
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, pred, target):
        # Vehicle channels only (ch 2,3,4)
        pred_car = pred[:, 2:, :, :]
        target_car = target[:, 2:, :, :]

        # Weighted BCE: vehicle class weight = 5.0
        weight = torch.ones_like(target_car) * 5.0
        bce_loss = F.binary_cross_entropy(pred_car, target_car, weight=weight, reduction='mean')

        # Dice loss
        pred_flat = pred_car.reshape(pred_car.shape[0], -1)
        target_flat = target_car.reshape(target_car.shape[0], -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return 0.5 * bce_loss + 0.5 * dice_loss

criterion = CombinedLoss()

# === Training Loop ===
print(f"=== Training started (Epochs: {EPOCHS}) ===")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    print(f"\nEpoch {epoch+1} / {EPOCHS}")

    for batch_idx, (x, mask, label) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        map_input = label[:, :2, :, :]
        mask_input = mask[:, -1:, :, :].repeat(1, 4, 1, 1)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            output, _ = model(x, mask, map_input)

        loss = criterion(output.float(), label.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"=== Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f} ===")

    if (epoch + 1) % 20 == 0:
        save_path = f"merging_model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved: {save_path}")

print("\nTraining complete.")