# Environment
Python 3.8.19

Pytroch 2.0.0

Cuda 12.6

# Our Dataset


# Directory structure description
├── README.md // Help document

├── Modules

```
│   ├── DTEM.py

│   ├── HFCIM.py

│   ├── HFE_Block.py

│   ├── highfre.py

│   ├── highfre3.py

│   ├── IAM.py

│   ├── lowfre.py

│   ├── SKFF.py

│   ├── wavelet.py

│   ├── Net.py  // Overall network architecture
```
├── src
```
│   ├── datasets.py // dataloader

│   ├── getmetrics.py  // calculate PSNR SSIM

│   ├── losses.py   // loss

│   ├── network.py   // VGG loss

│   ├── test.py   // test

│   ├── test_dataset_value.py // Calculate the PSNR and SSIM of the predicted images

│   ├── train.py  // train

│   ├── util.py  // save image and set random seed
```
# Training process
Run train.py train:
```
low_left = r'Low light left view path'
low_right = r'Low light right view path'
gt_left = r'ground truth left view path'
gt_right = r'ground truth right view path'
```
val:
```
val_low_left = r'Low light left view path'
val_low_right = r'Low light right view path'
val_gt_left = r'ground truth left view path'
val_gt_right = r'ground truth right view path'
```
# Testing process
```
Run test.py

parser.add_argument('--low_l', type=str, default=r"Low light left view path")
parser.add_argument('--low_r', type=str, default=r"Low light right view path")
parser.add_argument('--sava_left', type=str, default=r"save left view path")
parser.add_argument('--save_right', type=str, default=r"save right view path")
parser.add_argument('--snapshots_pth', type=str, default="../models/111.pth")

Run test_dataset_value.py to calculate SSIM and PSNR

path1 = r"ground truth path"
path2 = r"pre path"
```
