import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image


# ==========================================
# 1. 数据增强：构造对比学习的正样本对
# ==========================================
class SimCLRTransform:
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ==========================================
# 2. 核心修改：支持全类别数据的混合加载器
# ==========================================
class UnifiedFabricDataset(Dataset):
    def __init__(self, root_dataset_dir, transform=None):
        self.root_dataset_dir = root_dataset_dir
        self.transform = transform
        self.image_paths = []

        print(f"--- 正在扫描全类别正常样本: {root_dataset_dir} ---")
        # 遍历数据集根目录下的所有子文件夹 (例如 fabric_tiaosha, fabric_youwu 等)
        for class_name in os.listdir(root_dataset_dir):
            class_dir = os.path.join(root_dataset_dir, class_name)

            # 确保它是个文件夹
            if os.path.isdir(class_dir):
                good_dir = os.path.join(class_dir, 'train', 'good')

                # 如果这个类别有 train/good 文件夹，就把里面的图片全部加入训练列表
                if os.path.exists(good_dir):
                    class_img_count = 0
                    for f in os.listdir(good_dir):
                        if f.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            self.image_paths.append(os.path.join(good_dir, f))
                            class_img_count += 1
                    print(f"[{class_name}] 找到 {class_img_count} 张正常图片")

        print(f"--- 扫描完毕，总共收集到 {len(self.image_paths)} 张正常图片参与 SimCLR 预训练 ---")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


# ==========================================
# 3. 轻量级网络设计 + 投影头
# ==========================================
class SimCLR_MobileNetV3(nn.Module):
    def __init__(self, out_dim=128):
        super(SimCLR_MobileNetV3, self).__init__()
        backbone = models.mobilenet_v3_large(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.projection_head = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        z = self.projection_head(h)
        return h, z

    # ==========================================


# 4. 对比学习损失函数
# ==========================================
def info_nce_loss(z_i, z_j, temperature=0.5):
    device = z_i.device
    batch_size = z_i.size(0)

    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T) / temperature
    similarity_matrix.fill_diagonal_(-1e9)

    labels = torch.cat([torch.arange(batch_size) + batch_size,
                        torch.arange(batch_size)], dim=0).to(device)

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


# ==========================================
# 5. 主训练循环
# ==========================================
def main():
    # --- 关键修改：将路径指向你的数据集根目录 ---
    dataset_root = r'E:\Dataset\mvtec'
    batch_size = 16
    epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行环境: {device}")

    # 使用新的 UnifiedFabricDataset 加载全类别数据
    dataset = UnifiedFabricDataset(root_dataset_dir=dataset_root, transform=SimCLRTransform(size=224))

    # 防止图片太少导致 batch 报错，做个保护
    if len(dataset) < batch_size:
        batch_size = len(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = SimCLR_MobileNetV3(out_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for (x_i, x_j) in pbar:
            x_i, x_j = x_i.to(device), x_j.to(device)

            _, z_i = model(x_i)
            _, z_j = model(x_j)

            loss = info_nce_loss(z_i, z_j, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': loss.item()})

        print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {total_loss / len(dataloader):.4f}")

    os.makedirs('code/src/datasets/weights', exist_ok=True)
    # --- 关键修改：更名为全类别权重 ---
    save_path = 'code/src/datasets/weights/mobilenet_v3_simclr_all_classes.pth'
    torch.save(model.backbone.state_dict(), save_path)
    print(f"--- 全类别 SimCLR 预训练完成，权重已保存至 {save_path} ---")


if __name__ == '__main__':
    main()