import argparse
import os
import time
from datetime import datetime
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from network import AHDRNet
from Dataset import HDR_Dataset
import utils


def set_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/epoch3300_2301.pt', help='训练好的权重的路径')
    parser.add_argument('--dataset_path', type=str, default='D:/MyDataset/HDR/', help='HDR数据集主路径')
    parser.add_argument('--save_path', type=str, default='./save_hdr_image/', help='保存HDR图片的路径')

    args = parser.parse_args()
    print('=-' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=-' * 30)

    return args


def fusion(args):
    # 保存路径
    test_time = datetime.now().strftime("%m-%d_%H-%M")
    save_path = os.path.join(args.save_path, test_time + '/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 获取计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-当前计算设备：{}".format(torch.cuda.get_device_name(0)))

    # 构建网络
    model = AHDRNet().to(device)
    print('-网络构建完成')

    # 读取权重
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('--权重载入完成...')

    # 读取数据
    test_dataset = HDR_Dataset(
        dataset_path=args.dataset_path,
        is_Training=False
    )
    test_dataset.transform = A.Compose(
            [
                ToTensorV2(p=1.0),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
                "image3": "image",
            },
        )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False
    )

    for i, sample in enumerate(test_loader):
        time_start = time.time()  # 记录开始时间

        X1 = sample['X1'].to(device)
        X2 = sample['X2'].to(device)
        X3 = sample['X3'].to(device)
        H = sample['HDR'].to(device)

        X = torch.cat([X1, X2, X3], dim=0)
        I = X[:, :3, :, :]

        with torch.no_grad():
            H_merge = model(X1, X2, X3)
            HDR_Mu = utils.Gamma_Correction(H_merge)

        HDR = H_merge.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        HDR_Mu = HDR_Mu.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        save_name = 'result_{}.hdr'.format(i + 1)
        save_name_Mu = 'result_with_Mu_law{}.hdr'.format(i + 1)
        cv2.imwrite(os.path.join(save_path, save_name), HDR)
        cv2.imwrite(os.path.join(save_path, save_name_Mu), HDR_Mu)
        save_image(I, os.path.join(save_path, 'Input_Image_{}.tif'.format(i + 1)))

        time_end = time.time()
        print('输出路径：' + os.path.join(save_path, save_name) + '-融合耗时：{}'.format(time_end - time_start))


if __name__ == '__main__':
    args = set_test_args()
    fusion(args)
