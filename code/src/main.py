import argparse
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2
import datasets.mvtec as mvtec
from einops import rearrange
from sklearn.neighbors import NearestNeighbors


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./result")
    return parser.parse_args()


def main():
    args = parse_args()
    print('pwd=', os.getcwd())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = wide_resnet50_2(pretrained=True, progress=True)
    model.to(device)
    model.eval()
    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer2[3].register_forward_hook(hook)

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]
    fig_pixel_rocauc = ax[1]

    total_roc_auc = []
    total_pixel_roc_auc = []

    for class_name in mvtec.CLASS_NAMES:
        test_dataset = mvtec.MVTecDataset(
        root_path=r'E:/',
        class_name=class_name,
        is_train=False
        )
        test_dataloader = DataLoader(test_dataset, batch_size=1, pin_memory=True)

        gt_list = []
        gt_mask_list = []
        test_imgs = []
        score_map_list = []
        scores = []
        cut_surrounding = 32

        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test |'):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask[:, :, cut_surrounding:x.shape[2] - cut_surrounding,
                                cut_surrounding:x.shape[2] - cut_surrounding].cpu().detach().numpy().astype(int))
            # 找到循环内部的这一段，修改如下：
            features = get_feature(model, x, device, outputs)  # features 已经在 GPU 上
            m = torch.nn.AvgPool2d(3, 1, 1).to(device)

            # 直接调用 GPU 版计算函数
            # 这里的 features[0] 形状是 [1, 512, 40, 40]
            refined_features = m(features[0])
            heatMap_gpu = calc_score_gpu(refined_features, K=400)

            # 将结果转回 CPU 供后续可视化使用
            heatMap2 = heatMap_gpu.cpu().detach().numpy()

            for imgID in range(x.shape[0]):
                cut2 = 3
                newHeat = interpolate_scoremap(imgID, heatMap2, cut2, x.shape[2])
                newHeat = gaussian_filter(newHeat.squeeze().cpu().detach().numpy(), sigma=4)
                newHeat = torch.from_numpy(newHeat.astype(np.float32)).clone().unsqueeze(0).unsqueeze(0)

                score_map_list.append(newHeat[:, :, cut_surrounding:x.shape[2]-cut_surrounding,
                                      cut_surrounding:x.shape[2] - cut_surrounding])
                scores.append(score_map_list[-1].max().item())

        ##################################################
        # calculate image-level ROC AUC score
        fpr, tpr, _ = roc_curve(gt_list, scores)
        roc_auc = roc_auc_score(gt_list, scores)
        total_roc_auc.append(roc_auc)
        print('%s ROCAUC: %.3f' % (class_name, roc_auc))
        fig_img_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, roc_auc))

        # calculate per-pixel level ROCAUC
        flatten_gt_mask_list = np.concatenate(gt_mask_list).ravel()
        flatten_score_map_list = np.concatenate(score_map_list).ravel()

        fpr, tpr, _ = roc_curve(flatten_gt_mask_list, flatten_score_map_list)
        per_pixel_rocauc = roc_auc_score(flatten_gt_mask_list, flatten_score_map_list)
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))

        # get optimal threshold
        precision, recall, thresholds = precision_recall_curve(flatten_gt_mask_list, flatten_score_map_list)
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        # visualize localization result
        visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold, args.save_path, class_name, 5,
                             cut_surrounding)

        fig.tight_layout()
        fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


def interpolate_scoremap(imgID, heatMap, cut, imgshape):
    # --- 关键修复：确保 heatMap 是 Tensor ---
    if isinstance(heatMap, np.ndarray):
        heatMap = torch.from_numpy(heatMap)

    # 获取单张图的分数图
    current_map = heatMap[imgID, :, :]

    # 创建底板，使用当前图的最小值填充（论文逻辑：消除边缘影响） [cite: 165]
    blank = torch.ones_like(current_map) * current_map.min()

    # 只有中间部分保留计算出的分数，边缘设为最小值（Mask 掉 Padding 噪声） [cite: 166]
    h, w = current_map.shape
    blank[cut:h - cut, cut:w - cut] = current_map[cut:h - cut, cut:w - cut]

    # 进行双线性插值缩放到原图大小 [cite: 169]
    return F.interpolate(blank.unsqueeze(0).unsqueeze(0), size=imgshape, mode='bilinear', align_corners=False)


def get_feature(model, img, device, outputs):
    with torch.no_grad():
        _ = model(img.to(device))

    layer2_feature = outputs[-1]

    outputs.clear()
    return [layer2_feature]


def calc_score_gpu(features, K=400):
    """
    使用 PyTorch CUDA 加速计算异常得分
    features: [1, 512, 40, 40] -> 来自 Layer2 的特征
    K: 400 (论文建议值)
    """
    # 1. 形状变换: [1, 512, 40, 40] -> [1, 1600, 512]
    B, C, H, W = features.shape
    f = rearrange(features, 'b c h w -> b (h w) c')

    # 2. 计算两两之间的欧氏距离矩阵 [1, 1600, 1600]
    # torch.cdist 在 GPU 上极其快速
    dist_matrix = torch.cdist(f, f, p=2)

    # 3. 寻找最近的 K 个邻居
    # 因为 gallery 是它自己，所以最近的第一个点距离必然是 0 (它自己)，我们取 K+1 个点
    # topk 默认取最大，设置 largest=False 取最小
    topk_dist, _ = torch.topk(dist_matrix, k=K + 1, dim=-1, largest=False)

    # 4. 计算平均距离 (剔除第一个距离为 0 的点)
    # 结果形状: [1, 1600]
    score_map = torch.mean(topk_dist[:, :, 1:], dim=-1)

    # 5. 还原回空间维度 [1, 40, 40]
    return score_map.view(B, H, W)


def visualize_loc_result(test_imgs, gt_mask_list, score_map_list, threshold,
                         save_path, class_name, vis_num,cut_pixel):
    for t_idx in range(vis_num):
        test_img = test_imgs[t_idx]
        test_img = denormalization(test_img)
        test_gt = gt_mask_list[t_idx].transpose(1, 2, 0).squeeze()
        heat = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy().copy()
        test_pred = score_map_list[t_idx].flatten(0, 2).cpu().detach().numpy()
        test_pred[test_pred <= threshold] = 0
        test_pred[test_pred > threshold] = 1
        test_pred_img = test_img.copy()
        test_pred_img =test_pred_img[cut_pixel:test_img.shape[0]-cut_pixel,cut_pixel:test_img.shape[0]-cut_pixel, :]
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
        ax_img[2].imshow(heat)
        ax_img[2].title.set_text('HeatMap')
        ax_img[3].imshow(test_pred_img)
        ax_img[3].title.set_text('Predicted anomalous image')

        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        fig_img.savefig(os.path.join(save_path, 'images', '%s_%03d.png' % (class_name, t_idx)), dpi=100)
        fig_img.clf()
        plt.close(fig_img)


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


if __name__ == '__main__':
    main()

