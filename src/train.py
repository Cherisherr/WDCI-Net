from torch.utils.data import DataLoader
from getmetrics import psnr_ssim
from module.Net import Net
from losses import LossFre, LossSpa,TVLoss
from datasets import Dataset
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import torch
import time
from util import set_random_seed
import pywt
import numpy as np
import network


epoch_losses = [1]
# train
low_left = r'D:\pycharm\11\WDCINet-main\dataset\train_true\low\left'
low_right = r'D:\pycharm\11\WDCINet-main\dataset\train_true\low\right'
gt_left = r'D:\pycharm\11\WDCINet-main\dataset\train_true\gt\left'
gt_right = r'D:\pycharm\11\WDCINet-main\dataset\train_true\gt\right'
seed = 12345
train_batch_size = 20
crop_train = [128, 128]
scheduler_list = [250, 500, 750, 1000]
lr = 0.0002
epochs = 1000
# val
val_low_left = r'D:\pycharm\11\WDCINet-main\dataset\val_true\low\left'
val_low_right = r'D:\pycharm\11\WDCINet-main\dataset\val_true\low\right'
val_gt_left = r'D:\pycharm\11\WDCINet-main\dataset\val_true\gt\left'
val_gt_right = r'D:\pycharm\11\WDCINet-main\dataset\val_true\gt\right'
val_batch_size = 1
crop_val = [400, 400]
val_gap = 1
set_random_seed(seed)
# datasets
train_dataset = Dataset(low_left, low_right, gt_left, gt_right,mode='train', crop=crop_train, random_resize=None)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1, pin_memory=True)
val_dataset = Dataset( val_low_left, val_low_right,val_gt_left, val_gt_right, mode='val', crop=crop_val)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1, pin_memory=True)
ssimepochs = [1]
best_ssim_epoch = 0
best_psnr_epoch = 0
max_ssim_val = 0
max_psnr_val = 0
model = Net().cuda()
freloss = LossFre().cuda()
spaloss = LossSpa().cuda()
tvloss = TVLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, scheduler_list, 0.5)




save_dir = "../model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  
def main():
    start_epoch = 1
    start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train(epoch)
        if (epoch) % val_gap == 0:
            val(epoch)
        elapsed_time = time.time() - start
        h, rem = divmod(elapsed_time, 3600)
        m, rem = divmod(rem, 60)
        s = rem
        avg_time_per_epoch = elapsed_time / epoch
        remaining_epochs = epochs - epoch
        remaining_time = remaining_epochs * avg_time_per_epoch

        hours, rem = divmod(remaining_time, 3600)
        minutes, _ = divmod(rem, 60)
        print(
            'Epoch [{}/{}], Time Elapsed: {:.2f} hours, {:.2f} minutes, {:.2f} seconds, Expected Time Remaining: {} hours and {} minutes'.format(
                epoch, epochs, int(h), int(m), int(s), int(hours), int(minutes)))




def train(epoch):
    model.train()
    max = len(train_dataloader)
    for i, (low_l, low_r, gt_l, gt_r) in enumerate(train_dataloader):
        [low_l, low_r, gt_l, gt_r] = [x.cuda() for x in [low_l, low_r, gt_l, gt_r]]
        optimizer.zero_grad()
        gtl = gt_l * 255.0
        gtr = gt_r * 255.0
        gt_l_cpu = gtl.cpu()
        gt_ll = gt_l_cpu.numpy()
        gt_r_cpu = gtr.cpu()
        gt_rr = gt_r_cpu.numpy()

        cA_r1, (cH_rL2, cV_rL2, cD_rL2) = pywt.dwt2(gt_ll[:, 0, :, :], "haar")  # Red Channel
        cA_g1, (cH_gL2, cV_gL2, cD_gL2) = pywt.dwt2(gt_ll[:, 1, :, :], "haar")  # Green Channel
        cA_b1, (cH_bL2, cV_bL2, cD_bL2) = pywt.dwt2(gt_ll[:, 2, :, :], "haar")  # Blue Channel


        rgb_image = np.stack((cA_r1, cA_g1,  cA_b1), axis=1)




        cA_r2, (cH_rL4, cV_rL4, cD_rL4) = pywt.dwt2(rgb_image[:, 0, :, :], "haar")  # Red Channel
        cA_g2, (cH_gL4, cV_gL4, cD_gL4) = pywt.dwt2(rgb_image[:, 1, :, :], "haar")  # Green Channel
        cA_b2, (cH_bL4, cV_bL4, cD_bL4) = pywt.dwt2(rgb_image[:, 2, :, :], "haar")  # Blue Channel


        rgb_image = np.stack((cA_r2, cA_g2, cA_b2), axis=1)



        cA_r3, (cH_rL8, cV_rL8, cD_rL8) = pywt.dwt2(rgb_image[:, 0, :, :], "haar")  # Red Channel
        cA_g3, (cH_gL8, cV_gL8, cD_gL8) = pywt.dwt2(rgb_image[:, 1, :, :], "haar")  # Green Channel
        cA_b3, (cH_bL8, cV_bL8, cD_bL8) = pywt.dwt2(rgb_image[:, 2, :, :], "haar")  # Blue Channel


        rgb_image = np.stack((cA_r3, cA_g3, cA_b3), axis=1)
        rgb_image = rgb_image / np.max(rgb_image)









        cAr1, (cH_rR2, cV_rR2, cD_rR2) = pywt.dwt2(gt_rr[:, 0, :, :], "haar")  # Red Channel
        cAg1, (cH_gR2, cV_gR2, cD_gR2) = pywt.dwt2(gt_rr[:, 1, :, :], "haar")  # Green Channel
        cAb1, (cH_bR2, cV_bR2, cD_bR2) = pywt.dwt2(gt_rr[:, 2, :, :], "haar")  # Blue Channel



        rgbimage = np.stack((cAr1, cAg1, cAb1), axis=1)

        cAr2, (cH_rR4, cV_rR4, cD_rR4) = pywt.dwt2(rgbimage[:, 0, :, :], "haar")  # Red Channel
        cAg2, (cH_gR4, cV_gR4, cD_gR4) = pywt.dwt2(rgbimage[:, 1, :, :], "haar")  # Green Channel
        cAb2, (cH_bR4, cV_bR4, cD_bR4) = pywt.dwt2(rgbimage[:, 2, :, :], "haar")  # Blue Channel


        rgbimage = np.stack((cAr2, cAg2, cAb2), axis=1)



        cAr3, (cH_rR8, cV_rR8, cD_rR8) = pywt.dwt2(rgbimage[:, 0, :, :], "haar")  # Red Channel
        cAg3, (cH_gR8, cV_gR8, cD_gR8) = pywt.dwt2(rgbimage[:, 1, :, :], "haar")  # Green Channel
        cAb3, (cH_bR8, cV_bR8, cD_bR8) = pywt.dwt2(rgbimage[:, 2, :, :], "haar")  # Blue Channel



        rgbimage = np.stack((cAr3, cAg3, cAb3), axis=1)
        rgbimage = rgbimage / np.max(rgbimage)





        rgb_imagel = torch.from_numpy(rgb_image).float()
        rgbimager = torch.from_numpy(rgbimage).float()
        rgb_imagel= rgb_imagel.cuda()
        rgbimager = rgbimager.cuda()






        pre_l, pre_r,GL8,GR8= model(low_l, low_r)

        VGG = network.VGG19(init_weights=r'D:\pycharm\11\pre_trained_VGG19_model\vgg19.pth',feature_mode=True)
        VGG.cuda()
        l2_loss = torch.nn.MSELoss()

        GL8_1 = VGG(GL8)
        GR8_1 = VGG(GR8)
        rgb_imagel_1 = VGG(rgb_imagel)
        rgbimager_1 = VGG(rgbimager)

        fre_loss = freloss(pre_l, gt_l) + freloss(pre_r, gt_r)
        spa_loss = spaloss(pre_l, gt_l) + spaloss(pre_r, gt_r)
        midfre_loss = freloss(GL8,rgb_imagel)+ freloss(GR8,rgbimager)
        midspa_loss = spaloss(GL8,rgb_imagel)+ spaloss(GR8,rgbimager)
        vgg_loss = 0.001*(l2_loss(GL8_1,rgb_imagel_1)+l2_loss(GR8_1,rgbimager_1))

        loss =  fre_loss + spa_loss +midfre_loss +midspa_loss +vgg_loss    #改动
        loss.backward()
        optimizer.step()
        if (i + 1) % 1 == 0:
            print(
                'Training: Epoch[{:0>4}/{:0>4}] Iteration[{:0>4}/{:0>4}]  loss: {:.4f} fre_loss: {:.4f} spa_loss: {:.4f} midfre_loss: {:.4f} midspa_loss: {:.4f} vgg_loss: {:.4f} '.format(
                    epoch,
                    epochs, i + 1, max, loss, fre_loss, spa_loss,midfre_loss,midspa_loss,vgg_loss))        #改动

    torch.save(model.state_dict(), "../model/current_epoch_200.pth")

    scheduler.step()










def val(epoch):
    a_total = 0
    b_total = 0
    sum = 0
    for i, (low_l, low_r, gt_l, gt_r) in enumerate(val_dataloader):
        with torch.no_grad():
            model.eval()
            torch.cuda.empty_cache()
            [low_l, low_r, gt_l, gt_r] = [x.cuda() for x in [low_l, low_r, gt_l, gt_r]]
            val_l, val_r,GL8,GR8= model(low_l, low_r)     #改动
        val_l = val_l.squeeze(0)
        val_r = val_r.squeeze(0)
        gt_l = gt_l.squeeze(0)
        gt_r = gt_r.squeeze(0)
        psnr_l, ssim_l = psnr_ssim(val_l, gt_l)
        psnr_r, ssim_r = psnr_ssim(val_r, gt_r)
        a_total = a_total + (psnr_l + psnr_r) / 2
        b_total = b_total + (ssim_l + ssim_r) / 2
        i += 1
        sum = i
        print("第", i, "张:", "psnr:", (psnr_l + psnr_r) / 2, "  ssim:", (ssim_l + ssim_r) / 2)
    print("total: psnr", a_total / sum, " ssim:", b_total / sum)
    ssimepochs.append(1 - b_total / sum)
    global max_ssim_val, max_psnr_val, best_ssim_epoch, best_psnr_epoch
    if a_total / sum > max_psnr_val:
        max_psnr_val = a_total / sum
        best_psnr_epoch = epoch
        torch.save(model.state_dict(), "../model/best_psnr_epoch_200.pth")
    if b_total / sum > max_ssim_val:
        max_ssim_val = b_total / sum
        best_ssim_epoch = epoch
        torch.save(model.state_dict(), "../model/best_ssim_epoch_200.pth")
    plt.figure()
    plt.plot(ssimepochs, color='b')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.title('SSIM during Validation')
    plt.savefig(f'ssim_during_val.png')
    plt.close()
    torch.cuda.empty_cache()
    print("best ssim epoch: ", best_ssim_epoch, " ssim:", max_ssim_val, "best psnr epoch: ", best_psnr_epoch, " psnr:", max_psnr_val)






if __name__ == '__main__':
    main()
