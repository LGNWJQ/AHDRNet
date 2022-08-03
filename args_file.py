import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_writer', type=str, default="WJQ", help="Name of code writer")

    # 数据相关参数
    parser.add_argument('--dataset_path', type=str, default='D:/MyDataset/HDR/', help='HDR数据集主路径')
    parser.add_argument('--batch_size', type=int, default=4, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=2, help='读取数据集的cpu线程数量')
    parser.add_argument('--patch_size', type=int, default=256, help='训练时裁剪的图像尺寸')

    # 训练相关参数
    parser.add_argument('--seed', type=int, default=2022, help='随机种子，用于保证实验可复现')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='学习率')
    parser.add_argument('--epochs', type=int, default=500, help='训练周期数')

    parser.add_argument('--warm_start_path', type=str, default=None, help='继续训练的权重的路径')
    parser.add_argument('--save_frequency', type=int, default=10, help='权重保存的频率，以epoch为单位')



    # 显示参数
    args = parser.parse_args()
    print('=-' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=-' * 30)

    return args


if __name__ == '__main__':
    set_args()
    print(2 ** 16)