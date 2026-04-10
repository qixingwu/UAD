import argparse
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from einops import rearrange

import  mvtec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./curtain_results_dual")
    # 默认加载你第一阶段 SimCLR 跑出来的权重
    parser.add_argument("--weight_path", type=str, default="./weights/mobilenet_v3_simclr_all_classes.pth")
    return parser.parse_args()


# ==========================================
# 模块 1：轻量级网络与权重加载 (满足任务书效率要求)
# ==========================================
def load_simclr_model(weight_path, device):
    """加载经过 SimCLR 预训练的 MobileNetV3 骨干网络"""
    backbone = models.mobilenet_v3_large(pretrained=False)
    # 去除分类头，保留特征提取部分
    model = nn.Sequential(*list(backbone.children())[:-1])

    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"--- 成功加载 SimCLR 预训练权重: {weight_path} ---")
    else:
        print(f"--- 警告: 未找到 {weight_path}，使用未预训练权重 ---")

    model.to(device)
    model.eval()

    # 注册 Forward Hook 提取中间层特征
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    # 提取 MobileNetV3 第 6 层的特征 (包含丰富的局部纹理语义)
    model[0][6].register_forward_hook(hook)

    return model, outputs


# ==========================================
# 模块 2：构建外部特征原型库 (满足任务书正常样本建模)
# ==========================================
def build_prototype_bank(model, train_dataloader, device, outputs):
    """利用正常样本构建 Memory Bank"""
    print("--- 正在构建正常纹理特征原型库 (Memory Bank) ---")
    normal_features = []

    with torch.no_grad():
        for (x, _, _) in tqdm(train_dataloader, '| 提取正常特征 | train |'):
            x = x.to(device)
            _ = model(x)

            feat = outputs[-1]
            # 3x3 局部平均池化，增加感受野 (Patch-level)
            m = torch.nn.AvgPool2d(3, 1, 1).to(device)
            feat = m(feat)

            # 展平并存入列表
            f = rearrange(feat, 'b c h w -> (b h w) c')
            normal_features.append(f)
            outputs.clear()

    # 拼接所有正常特征
    all_normal_features = torch.cat(normal_features, dim=0)
    # 为保证 60 FPS 推理，随机下采样保留 2000 个原型节点
    if all_normal_features.size(0) > 2000:
        indices = torch.randperm(all_normal_features.size(0))[:2000]
        prototypes = all_normal_features[indices]
    else:
        prototypes = all_normal_features

    print(f"--- Memory Bank 构建完成，原型数量: {prototypes.shape[0]} ---")
    return prototypes


# ==========================================
# 模块 3：内外双重对比打分 (满足任务书对比学习与缺陷判定)
# ==========================================
def calc_dual_score_gpu(features, prototypes, K=400, lambda_weight=0.5):
    """融合内部同质性对比与外部 Memory Bank 对比（修复归一化 Bug 版）"""
    B, C, H, W = features.shape
    f = rearrange(features, 'b c h w -> b (h w) c')  # [1, N, C]

    # 自适应 K 值保护
    N = f.shape[1]
    actual_K = min(K, N - 1)

    # --- 1. 内部同质性打分 (Zero-shot 自对比) ---
    dist_int = torch.cdist(f, f, p=2)
    topk_dist_int, _ = torch.topk(dist_int, k=actual_K + 1, dim=-1, largest=False)
    score_int = torch.mean(topk_dist_int[:, :, 1:], dim=-1)  # [1, N]

    # --- 2. 外部原型打分 (Memory Bank 对比) ---
    dist_ext = torch.cdist(f, prototypes.unsqueeze(0), p=2)  # [1, N, M]
    score_ext, _ = torch.min(dist_ext, dim=-1)  # [1, N]

    # --- 3. 核心修复：直接使用原始物理距离相加，绝对不能做除法归一化！ ---
    # 由于同属于一个特征空间，直接根据 lambda_weight 组合即可
    total_score = score_int + lambda_weight * score_ext

    return total_score.view(B, H, W)


def interpolate_scoremap(imgID, heatMap, cut, imgshape):
    """插值函数，修补上一版本发现的 Numpy/Tensor 转换问题"""
    if isinstance(heatMap, np.ndarray):
        heatMap = torch.from_numpy(heatMap)

    current_map = heatMap[imgID, :, :]
    blank = torch.ones_like(current_map) * current_map.min()

    # 切除卷积 Padding 带来的边缘伪影
    h, w = current_map.shape
    blank[cut:h - cut, cut:w - cut] = current_map[cut:h - cut, cut:w - cut]

    return F.interpolate(blank.unsqueeze(0).unsqueeze(0), size=imgshape, mode='bilinear', align_corners=False)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, save_path, class_name, vis_num, cut_pixel):
    for t_idx in range(vis_num):
        if t_idx >= len(test_imgs):
            break
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        heat = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy().copy()

        test_pred = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy()
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img = test_pred_img[cut_pixel:test_img.shape[0] - cut_pixel, cut_pixel:test_img.shape[0] - cut_pixel,
                        :]
        test_pred_img[test_pred == 0] = 0

        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 4))
        fig_img.subplots_adjust(left=0, right=1, bottom=0, top=1)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(test_img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(test_gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax_img[2].imshow(heat, cmap='viridis')
        ax_img[2].title.set_text('HeatMap (Dual Contrast)')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted Anomalous Area')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前运行环境: {device}')

    # 加载带有 SimCLR 权重的模型
    model, outputs = load_simclr_model(args.weight_path, device)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    for class_name in mvtec.CLASS_NAMES:
        print(f"\n========== 开始处理类别: {class_name} ==========")
        if class_name in ['fabric_tiaosha', 'fabric_fangyan']:
            dynamic_lambda = 0.01  # 结构型瑕疵，大幅降低外部干扰，依赖内部自对比
        else:
            dynamic_lambda = 1.0  # 覆盖型/颜色型瑕疵，依赖外部 Memory Bank
        # 1. 准备训练集 (用于构建原型) 和测试集
        train_dataset = mvtec.MVTecDataset(root_path='E:/', class_name=class_name, is_train=True)
        test_dataset = mvtec.MVTecDataset(root_path='E:/', class_name=class_name, is_train=False)

        train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

        # 2. 核心：构建正常特征原型库 (Memory Bank)
        prototypes = build_prototype_bank(model, train_dataloader, device, outputs)

        gt_list = []
        gt_mask_list = []
        test_imgs = []
        score_map_list = []
        scores = []
        cut_surrounding = 32

        # 3. 开启推理测试流程
        for (x, y, mask) in tqdm(test_dataloader, '| 异常特征提取与打分 | test |'):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask[:, :, cut_surrounding:x.shape[2] - cut_surrounding,
                                cut_surrounding:x.shape[2] - cut_surrounding].cpu().detach().numpy().astype(int))

            with torch.no_grad():
                x = x.to(device)
                _ = model(x)

            feat = outputs[-1]
            m = torch.nn.AvgPool2d(3, 1, 1).to(device)
            refined_features = m(feat)

            # --- 调用双重对比打分函数 ---
            # lambda_weight=0.5 表示外部特征权重，对于“油污”可调高该值
            heatMap_gpu = calc_dual_score_gpu(refined_features, prototypes, K=400, lambda_weight=0.5)
            outputs.clear()

            # 处理多张图片的结果 (尽管当前 batch_size=1)
            for imgID in range(x.shape[0]):
                cut2 = 3
                newHeat = interpolate_scoremap(imgID, heatMap_gpu, cut2, x.shape[2])

                # 转为 numpy 进行高斯平滑
                newHeat_np = newHeat.squeeze().cpu().detach().numpy()
                newHeat_np = gaussian_filter(newHeat_np, sigma=4)

                # 转回 tensor
                newHeat = torch.from_numpy(newHeat_np.astype(np.float32)).clone().unsqueeze(0).unsqueeze(0)

                score_map_list.append(newHeat[:, :, cut_surrounding:x.shape[2] - cut_surrounding,
                                      cut_surrounding:x.shape[2] - cut_surrounding])
                current_map = score_map_list[-1].squeeze().flatten()
                topk_num = 10
                if current_map.shape[0] > topk_num:
                    topk_values, _ = torch.topk(current_map, k=topk_num)
                    image_score = topk_values.mean().item()
                else:
                    image_score = current_map.max().item()

                scores.append(image_score)

        # ==========================================
        # 结果计算与评估
        # ==========================================
        # Image-level ROC AUC
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        print(f'[{class_name}] Image ROCAUC: {roc_auc:.3f}')
        fig_img_rocauc.plot(fpr, tpr, label=f'{class_name} ROCAUC: {roc_auc:.3f}')

        # Pixel-level ROC AUC
        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        print(f'[{class_name}] Pixel ROCAUC: {per_pixel_rocauc:.3f}')
        fig_pixel_rocauc.plot(fpr, tpr, label=f'{class_name} ROCAUC: {per_pixel_rocauc:.3f}')

        # 获取最佳阈值
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # 可视化并保存
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, 5,
                             cut_surrounding)

    fig_img_rocauc.title.set_text('Image ROCAUC')
    fig_img_rocauc.legend()
    fig_pixel_rocauc.title.set_text('Pixel ROCAUC')
    fig_pixel_rocauc.legend()
    fig.tight_layout()
    os.makedirs(args.save_path, exist_ok=True)
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)
    print(f"\n--- 评估全部完成！结果已保存至 {args.save_path} ---")


if __name__ == '__main__':
    main()