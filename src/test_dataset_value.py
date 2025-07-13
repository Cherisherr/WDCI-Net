import os
import math
import cv2
import numpy as np
from skimage import metrics


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    return metrics.structural_similarity(img1, img2)


def test_dataset_quality_avg():
    path1 = r"E:\flickr\left"
    path2 = r"D:\pycharm\11\WDCINet-main\dataset\test\flickr\normal\gt\left"

    file_list = os.listdir(path1)
    total_psnr = 0.0
    total_ssim = 0.0
    valid_count = 0

    for file_name in file_list:
        img_path1 = os.path.join(path1, file_name)
        img_path2 = os.path.join(path2, file_name)

        if not os.path.exists(img_path2):
            print(f"Ground truth missing for {file_name}, skipping.")
            continue

        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None or img2 is None:
            print(f"Error reading {file_name}, skipping.")
            continue

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        psnr_val = psnr(img1, img2)
        ssim_val = ssim(img1, img2)

        print(f"{file_name} - PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}")

        total_psnr += psnr_val
        total_ssim += ssim_val
        valid_count += 1

    if valid_count > 0:
        avg_psnr = total_psnr / valid_count
        avg_ssim = total_ssim / valid_count
        print(f"\n平均 PSNR: {avg_psnr:.4f}")
        print(f"平均 SSIM: {avg_ssim:.4f}")
    else:
        print("没有有效的图像对进行评估。")
