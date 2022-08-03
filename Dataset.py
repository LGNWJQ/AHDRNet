import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import os
import cv2
import imageio
imageio.plugins.freeimage.download()


class HDR_Dataset(Dataset):
    def __init__(self, dataset_path, is_Training=True, patch_size=256):
        self.dataset_path = os.path.join(dataset_path, 'Training/') if is_Training else os.path.join(dataset_path, 'Test/EXTRA/')
        data_list = os.listdir(self.dataset_path)
        # 通过检查是否有hdr文件来过滤无效文件or文件夹
        self.data_list = [x for x in data_list if os.path.exists(os.path.join(self.dataset_path, x, 'HDRImg.hdr'))]
        self.ToFloat32 = A.ToFloat(max_value=65535.0)
        self.train_transform = A.Compose(
            [
                A.RandomCrop(width=patch_size, height=patch_size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                ToTensorV2(p=1.0),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
                "image3": "image",
            },
        )
        self.test_transform = A.Compose(
            [
                A.CenterCrop(height=512, width=512),
                ToTensorV2(p=1.0),
            ],
            additional_targets={
                "image1": "image",
                "image2": "image",
                "image3": "image",
            },
        )
        self.transform = self.train_transform if is_Training else self.test_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_root = os.path.join(self.dataset_path, self.data_list[idx] + '/')
        file_list = sorted(os.listdir(data_root))

        # 子文件路径
        image_path1 = os.path.join(data_root, file_list[0])
        image_path2 = os.path.join(data_root, file_list[1])
        image_path3 = os.path.join(data_root, file_list[2])
        label_path = os.path.join(data_root, file_list[3])
        txt_path = os.path.join(data_root, file_list[4])

        # 读取输入的TIFF图像
        image1 = cv2.imread(image_path1, cv2.IMREAD_UNCHANGED)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = self.ToFloat32(image=image1)['image']

        image2 = cv2.imread(image_path2, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = self.ToFloat32(image=image2)['image']

        image3 = cv2.imread(image_path3, cv2.IMREAD_UNCHANGED)
        image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
        image3 = self.ToFloat32(image=image3)['image']

        # 读取曝光时间
        expoTimes = np.power(2, np.loadtxt(txt_path))
        expoTimes = torch.from_numpy(expoTimes)

        # 读取HDR图像
        hdr_image = imageio.imread(label_path, format='HDR-FI')
        hdr_image = np.array(hdr_image)
        hdr_image = hdr_image[:, :, (2, 1, 0)]

        # 数据增强
        augmentations = self.transform(
            image=hdr_image,
            image1=image1,
            image2=image2,
            image3=image3,
        )
        image1 = augmentations['image1']
        image2 = augmentations['image2']
        image3 = augmentations['image3']
        hdr_image = augmentations['image']

        # 数据处理
        H_image1 = Gamma_Correction(image1, gamma=2.2)
        H_image2 = Gamma_Correction(image2, gamma=2.2)
        H_image3 = Gamma_Correction(image3, gamma=2.2)

        X1 = torch.cat([image1, H_image1], dim=0)
        X2 = torch.cat([image2, H_image2], dim=0)
        X3 = torch.cat([image3, H_image3], dim=0)

        sample = {
            'X1': X1,
            'X2': X2,
            'X3': X3,
            'HDR': hdr_image
        }

        return sample


def Gamma_Correction(image_tensor, gamma):
    return torch.pow(image_tensor, 1.0/gamma)





from args_file import set_args
from matplotlib import pyplot as plt
import pylab
from torchvision.utils import save_image
from torch.utils.data import DataLoader

def visualize(image):
    # Divide all values by 65535 so we can display the image using matplotlib
    # image = image / 65535
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    pylab.show()


if __name__ == '__main__':
    args = set_args()

    HDR_set = HDR_Dataset(
        dataset_path=args.dataset_path,
        is_Training=False
    )

    loader = DataLoader(
        dataset=HDR_set,
        batch_size=4,
        num_workers=2,
        shuffle=False
    )

    for i, sample in enumerate(loader):
        X1 = sample['X1']
        X2 = sample['X2']
        X3 = sample['X3']
        H = sample['HDR']
        #     # hdr输出案例
        #     # hdr = hdr.permute(1, 2, 0).numpy()
        #     # hdr = hdr[:, :, (2, 1, 0)]
        #     # cv2.imwrite('test.hdr', hdr)

        for j in range(H.shape[0]):
            hdr = H[j, :, :, :]
            hdr = hdr.permute(1, 2, 0).numpy()
            # hdr = hdr[:, :, (2, 1, 0)]
            cv2.imwrite('./hdrimage3/test{}{}.hdr'.format(i, j), hdr)






    # for i in range(HDR_set.__len__()):
    #     H_image1, H_image2, H_image3 = HDR_set.__getitem__(i)
    #     print('=-'*30)
    #     H_image1 = H_image1.unsqueeze(0)
    #     H_image2 = H_image2.unsqueeze(0)
    #     H_image3 = H_image3.unsqueeze(0)
    #
    #
    #     # ldr1 = law['X1']
    #     # ldr2 = law['X1']
    #     # ldr3 = law['X1']
    #     # hdr = law['HDR']
    #
    #     # ldr1 = sample['ldr1'] # .unsqueeze(0)
    #     # ldr2 = sample['ldr2']
    #     # ldr3 = sample['ldr3']
    #     # hdr = sample['hdr']
    #
    #     # print(type(ldr1))
    #     # print(type(ldr2))
    #     # print(type(ldr3))
    #     # print(type(hdr))
    #     # # <class 'numpy.ndarray'>
    #
    #     # print(ldr1.shape)
    #     # print(ldr2.shape)
    #     # print(ldr3.shape)
    #     # print(hdr.shape)
    #     # (1000, 1500, 3)
    #
    #     # print(type(ldr1[1, 1, 1])) # <class 'numpy.uint16'>
    #     # print(type(ldr2[1, 1, 1])) # <class 'numpy.uint16'>
    #     # print(type(ldr3[1, 1, 1])) # <class 'numpy.uint16'>
    #     # print(type(hdr[1, 1, 1]))  # <class 'numpy.float32'>
    #     # <class 'numpy.float32'>
    #
    #     # print(ldr1.dtype)
    #     # print(ldr2.dtype)
    #     # print(ldr3.dtype)
    #     # print(hdr.dtype)
    #
    #     # print(ldr1.min(), ldr1.max())  # 1158 65535
    #     # print(ldr2.min(), ldr2.max())  # 3905 65535
    #     # print(ldr3.min(), ldr3.max())  # 8883 65535
    #     # print(hdr.min(), hdr.max())    # 0.00079345703 1.0
    #
    #     # compare = torch.cat([ldr1, ldr2, ldr3, hdr], dim=0)
    #     # save_image(compare, './image/image{}.png'.format(i))
    #
    #     # hdr输出案例
    #     # hdr = hdr.permute(1, 2, 0).numpy()
    #     # hdr = hdr[:, :, (2, 1, 0)]
    #     # cv2.imwrite('test.hdr', hdr)
    #
    #     compare_LH = torch.cat([H_image1, H_image2, H_image3], dim=0)
    #     save_image(compare_LH, 'D:/MyAHDRNet/AHDRNet/test_image_list/gamma/image{}.png'.format(i))
    #     print(H_image2.shape, H_image2.max(), H_image2.min())





















