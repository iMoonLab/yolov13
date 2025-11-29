import torch
from ultralytics import YOLO

# 1. 用你刚刚写的 yolov13_pgm.yaml 构建模型（不加载 .pt）
model = YOLO(r"D:\Hyper_U\ultralytics\cfg\models\v13\yolov13n_pmap_test.yaml", task="detect")

# 2. 拿到底层的 nn.Module
net = model.model  # 这是 DetectionModel，但最后一层已经不是 Detect 了，而是 PGMHead

# 3. 造一张假图（或真实图像都行）
x = torch.randn(1, 3, 1024, 1024)

# 4. 前向一次
with torch.no_grad():
    out = net(x)   # out 应该是 (B, 3, H, W)

print("out shape:", out.shape)  # (1, 3, Hf, Wf)
Hf, Wf = out.shape[2], out.shape[3]
stride_h = 1024 / Hf
stride_w = 1024 / Wf
print("feature size:", Hf, Wf)
print("stride:", stride_h, stride_w)
