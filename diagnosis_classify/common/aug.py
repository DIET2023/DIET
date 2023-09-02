import numpy as np
import random
import skimage
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import imgaug.augmenters as iaa


class DataAug(object):
    __doc__ = """
        数据增强
    """
    random_order = None
    aug_mode = None
    aug_someopt = lambda aug: iaa.Sometimes(0.5, aug)
    aug_someopt_list = []

    def __init__(self, aug_mode, random_order=True, *args, **kwargs):
        self.random_order = random_order
        self.aug_mode = {mode: "" for mode in aug_mode}
        self.aug_opt = self.add_img_aug_opt()
        super(DataAug, self).__init__()

    @classmethod
    def add_img_aug_opt(cls):
        """
        :description: 添加数据增强操作
        :return:
        """
        for key in cls.aug_mode.keys():
            cls.aug_someopt_list.append(cls.img_aug_transform(key))
        cls.aug_someopt_list.append(iaa.Identity())
        return iaa.Sequential(cls.aug_someopt_list, random_order=cls.random_order)

    @classmethod
    def img_aug_transform(cls, mode):
        """
        Description: imgaug图像增强库
        :param mode:
        :return:
        """
        if mode == "Multiply":
            opt = iaa.Multiply((0.8, 1.2), per_channel=0.2)
        elif mode == "MultiplyAndAddToBrightness":
            opt = iaa.MultiplyAndAddToBrightness(mul=(0.3, 2.0), add=(-60, 60))
        elif mode == "MultiplyHueAndSaturation":
            opt = iaa.MultiplyHueAndSaturation((0.9, 1.2), per_channel=True)
        elif mode == "AddToHueAndSaturation":
            opt = iaa.AddToHueAndSaturation((-15, 15), per_channel=True)
        elif mode == "GaussianNoise":
            opt = iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
        elif mode == "ChangeColorTemperature":
            opt = iaa.ChangeColorTemperature((1300, 9000))
        elif mode == "ContrastNormalization":
            opt = iaa.ContrastNormalization((0.75, 1.5)),
        elif mode == "Affine":
            opt = iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        elif mode == "Crop":
            opt = iaa.Crop(percent=(0, 0.1))
        elif mode == "CropAndPad":
            opt = iaa.CropAndPad(percent=(-0.3, 0.1))
        elif mode == "Fliplr":
            opt = iaa.Fliplr(0.5)
        elif mode == "GammaContrast":
            opt = iaa.GammaContrast((0.3, 2.0), per_channel=True)

        return cls.aug_someopt(opt)

    @classmethod
    def torch_transforms(cls, shape, mode="train"):
        """
        • 数据中心化
        • 数据标准化
        • 缩放
        • 裁剪
        • 旋转
        • 翻转
        • 填充
        • 噪声添加
        • 灰度变换
        • 线性变换
        • 仿射变换
        • 亮度、饱和度及对比度变换
        """
        transform_opt ={
            "train":
                T.Compose([
                    T.Lambda(lambda x: TF.resize(x, shape)),
                    T.RandomCrop(shape),
                    T.RandomRotation(90),
                    T.RandomHorizontalFlip(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            "test":
                T.Compose([
                    T.Lambda(lambda x: TF.resize(x, shape)),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
            "val":
                T.Compose([
                    T.Lambda(lambda x: TF.resize(x, shape)),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        }
        return transform_opt[mode]

    @classmethod
    def random_noise(cls, image, mode='gaussian', seed=None, clip=True, **kwargs):
        # 功能：为浮点型图片添加各种随机噪声
        # 参数：
        # image：输入图片（将会被转换成浮点型），ndarray型
        # mode： 可选择，str型，表示要添加的噪声类型
        #  gaussian：高斯噪声
        #  localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
        #  poisson：泊松再生
        #  salt：盐噪声，随机将像素值变成1
        #  pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
        #  s&p：椒盐噪声
        #  speckle：均匀噪声（均值mean方差variance），out=image+n*image
        # seed： 可选的，int型，如果选择的话，在生成噪声前会先设置随机种子以避免伪随机
        # clip： 可选的，bool型，如果是True，在添加均值，泊松以及高斯噪声后，会将图片的数据裁剪到合适范围内。如果谁False，则输出矩阵的值可能会超出[-1,1]
        # mean： 可选的，float型，高斯噪声和均值噪声中的mean参数，默认值=0
        # var：  可选的，float型，高斯噪声和均值噪声中的方差，默认值=0.01（注：不是标准差）
        # local_vars：可选的，ndarry型，用于定义每个像素点的局部方差，在localvar中使用
        # amount： 可选的，float型，是椒盐噪声所占比例，默认值=0.05
        # salt_vs_pepper：可选的，float型，椒盐噪声中椒盐比例，值越大表示盐噪声越多，默认值=0.5，即椒盐等量
        # --------
        # 返回值：ndarry型，且值在[0,1]或者[-1,1]之间，取决于是否是有符号数
        return skimage.util.random_noise(image, mode=mode, seed=seed, clip=True)

    @classmethod
    def rotate(cls, image, anger, anger_factor=0.7):
        """旋转"""
        imgInfo = image.shape
        height = imgInfo[0]
        width = imgInfo[1]
        matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), anger, anger_factor)
        dst = cv2.warpAffine(image, matRotate, (height, width))
        return dst

    @classmethod
    def move(cls, image, move_x, move_y):
        """平移"""
        rows, cols, channels = image.shape
        M = np.float32([[1, 0, move_x], [0, 1, move_y]])
        dst = cv2.warpAffine(image, M, (cols, rows))
        return dst

    @classmethod
    def flip(cls, image, flipcode=0):
        """翻转"""
        return cv2.flip(image, flipcode)

    @classmethod
    def sp_noise(cls, image, prob):
        """
        :param image:
        :param prob:
        :return:
        """
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    @classmethod
    def gasuss_noise(cls, image, mean=0, var=0.001):
        """
        :param image:
        :param mean:
        :param var:
        :return:
        """
        image = np.array(image / 255, dtype=float)
        noise = np.random.normal(mean, var ** 0.5, image.shape)
        out = image + noise
        low_clip = -1. if out.min() else 0.
        out = np.clip(out, low_clip, 1.0)
        out = np.uint8(out * 255)
        return out

    @classmethod
    def area_mask(cls, image, area, color=(128, 128, 128)):
        """
        :param image:
        :param area: np.ndarray
        :param color: 灰度填充
        :return:
        """
        image = cv2.fillPoly(image, [area], color)
        return image

    @classmethod
    def box_mask(cls, image, point, h_w=[5, 5], color=(128, 128, 128)):
        area = [point, [point[0]+h_w[0], point[1]], [point[0]+h_w[0], point[1]+h_w[1]], [point[0], point[1]+h_w[1]]]
        area = np.array(area)
        image = cv2.fillPoly(image, [area], color)
        return image

    def __call__(self, *args, **kwargs):
        return self.aug_opt
