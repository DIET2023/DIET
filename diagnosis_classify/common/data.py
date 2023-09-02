"""
Created by yizhi.chen.
"""

import logging
import os
import json
import sys
from tqdm import tqdm
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
#import imageio
import numpy as np
#import imgaug as ia
#import imgaug.augmenters as iaa
#from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import ImageFile, Image
import cv2
import threading
import collections
#from .aug import DataAug

ImageFile.LOAD_TRUNCATED_IMAGES = True
cldas_sum = collections.deque()


class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def read_jsonl(path):
    coll = []
    for line in open(path):
        coll.append(json.loads(line))
    return coll


def load_dataset_info_jsonl(paths, mode=None):
    infos = []
    for path in paths:
        jsons = read_jsonl(path)
        for unit in jsons:
            infos.append(unit)
    random.shuffle(infos)
    return infos


def check_train_test(train_path, val_path, mode="train"):
    train_data = load_dataset_info_jsonl(train_path, mode)
    val_data = load_dataset_info_jsonl(val_path, mode)

    train_image = [os.path.join(iterm["image_path"]["volume"], iterm["image_path"]["key"]) for iterm in train_data if
                   iterm["image_path"]["volume"]]
    test_image = [os.path.join(iterm["image_path"]["volume"], iterm["image_path"]["key"]) for iterm in val_data if
                  iterm["image_path"]["volume"]]
    min_size = len(test_image)
    overlap = [test_image[i] for i in range(min_size) if test_image[i] in train_image]
    with open("overlap.txt", "w") as file:
        for iterm in overlap:
            file.write(iterm + "\n")
    if len(overlap) == 0:
        print("训练集{0} 数量{1}\n测试集{2}数量{3}\n无重叠数据".format(train_path, len(train_image), val_path, len(test_image)))
    else:
        print("训练集{0}\n测试集{1}\n有{2}重叠数据, 请仔细检查生成的重叠数据文件".format(train_path, val_path, len(overlap)))
        exit()


def _center_padding(img):
    height, width, _ = img.shape
    if width > height:
        img = np.pad(img, (((width - height) // 2, width - height - (width - height) // 2), (0, 0), (0, 0)), 'constant',
                     constant_values=((0, 0), (0, 0), (0, 0)))
    else:
        img = np.pad(img, ((0, 0), ((height - width) // 2, height - width - (height - width) // 2), (0, 0)), 'constant',
                     constant_values=((0, 0), (0, 0), (0, 0)))
    return img


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


def aug_data(image, anger=90, anger_factor=0.85, move_x=50, move_y=50, flipcode=0):
    """图片数据增强
    :param image:
    :return:
    """

    def rotate(image, anger, anger_factor=0.7):
        """旋转"""
        imgInfo = image.shape
        height = imgInfo[0]
        width = imgInfo[1]
        matRotate = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), anger, anger_factor)
        dst = cv2.warpAffine(image, matRotate, (height, width))
        return dst

    def move(image, move_x, move_y):
        """平移"""
        rows, cols, channels = image.shape
        M = np.float32([[1, 0, move_x], [0, 1, move_y]])
        dst = cv2.warpAffine(image, M, (cols, rows))
        return dst

    def flip(image, flipcode=0):
        """翻转"""
        return cv2.flip(image, flipcode)

    rotate_image = rotate(image, anger, anger_factor)
    move_image = move(image, move_x, move_y)
    flip_image = flip(image, flipcode)
    return [move_image, flip_image, rotate_image]


def read_image_with_protocol(path, shape=None, test=False):
    # # Lots of accidents during image reading, so catch it.
    # 【修改】 ############################################################################
    # image_data = Image.open(path)
    # image_data = np.array(image_data)
    # 1-4-1 实验测试发现许多图片读取，故因此修改图片读取的方式改为opencv: 2021.06.18:17:56(wliu)

    image_data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

    # resize: 512 操作 ##################################################
    if not test:
        image_data = cv2.resize(image_data, (int(shape[0]), int(shape[1])))
    # ##################################################################

    assert image_data.dtype == np.uint8
    assert image_data.ndim == 3, str(image_data.shape)

    if image_data.shape[2] == 4:
        # RGBA -> RGB
        image_data = image_data[:, :, :3]

    assert image_data.shape[2] == 3, "%s has shape: %s" % str(image_data.shape)
    return image_data


def onthot_label(label_info, label2id):
    num_classes = len(label2id)
    label = torch.zeros(num_classes)
    for l in label_info:
        label[label2id[l]] = 1.0
    return label


def read_image_label(iterm, label2id, shape):
    if iterm["image_path"]["volume"]:
        image_path = os.path.join(iterm["image_path"]["volume"], iterm["image_path"]["key"])
        image_data = read_image_with_protocol(image_path, shape)
        label = iterm["label"]
        label = onthot_label(label, label2id)
        return image_data, label


def get_into_memory(path, label_list, mode="train", threads_num=5, config=None):
    image_memory, label_memory, except_iterm = [], [], []
    infos = load_dataset_info_jsonl(path, mode)
    label2id = {key: i for i, key in enumerate(label_list)}
    infos_size = len(infos)
    print("输入数据总量:{0}".format(infos_size))
    print("输入数据示例①：", infos[0])
    print("输入类别标签:", label2id)
    bar = tqdm(total=len(infos))
    # #多线程处理#
    # for index in range(0, infos_size, threads_num):
    #     threads = [MyThread(read_image_labe, (infos[index+i], label2id, config["SHAPE"]), read_image_labe.__name__)
    #                        for i in range(threads_num)]
    #     for t in threads:
    #         t.setDaemon(True)
    #         t.start()
    #     for t in threads:
    #         t.join()
    #     for i in range(threads_num):
    #         image_memory.append(threads[i].get_result()[0])
    #         label_memory.append(threads[i].get_result()[1])
    #
    #     bar.update(threads_num)
    # if index>= 1000: break

    # #单线程处理#
    for index, iterm in enumerate(infos):
        # try:
        if iterm["image_path"]["volume"]:
            image_path = os.path.join(iterm["image_path"]["volume"], iterm["image_path"]["key"])
            image_data = read_image_with_protocol(image_path, shape=config["SHAPE"])
            image_memory.append(image_data)

            label = iterm["label"]
            label = onthot_label(label, label2id)
            label_memory.append(label)
            # 数据增强##########################################################################
            if config["AUG_DATA"]:
                anger = np.random.randint(--180, 180)
                anger_factor = np.random.randint(85, 120) / 100
                move_x = np.random.randint(-int(image_data.shape[0] * 0.1), int(image_data.shape[0] * 0.1))
                move_y = np.random.randint(-int(image_data.shape[0] * 0.1), int(image_data.shape[0] * 0.1))
                flipcode = np.random.choice([0, 1])
                aug_image = aug_data(image_data, anger=anger, anger_factor=anger_factor, move_x=move_x, move_y=move_y,
                                     flipcode=flipcode)
                for son_index, son_iterm in enumerate(aug_image):
                    image_memory.append(son_iterm)
                    label_memory.append(label)
                    # print(iterm["label"], label)
                    cv2.imwrite("/mnt/coyote/wliu/package/disease_analysis/MDC-5DC-1-5-7/eval/{0}_{1}".format(
                        iterm["image_path"]["key"], son_index),
                        cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                if index > 1000: exit()  # 调试数据
            # ##################################################################################
            bar.update(1)
            bar.set_description(f"Into memory {index}")
        # except Exception as e:
        #     except_iterm.append(iterm)

    print("读取到内存数：", len(image_memory), len(label_memory), "异常数据量:", len(except_iterm))
    return [image_memory, label_memory]


class SegmentDataset(torch.utils.data.Dataset):
    __doc__ = """
        修改为批量添加到运行内存中
    """

    def __init__(self, infos, mode):
        """
        Args:
            infos: List of dict. Necessary keys: image_path. Optional Keys: label_path
        """
        assert mode in ["train", "val", "test"], mode
        self.tfms = tfms[mode]
        self.ia_tfms = ia_tfms[mode]
        self.infos = infos

    def __getitem__(self, idx):
        image, label = self.infos[0][idx], self.infos[1][idx]
        # Img Aug.
        # image = self.ia_tfms(image=image)
        # Torch Aug.
        image = image.transpose((2, 0, 1)).astype(np.float32) / 255
        image = torch.from_numpy(image).contiguous()
        image = self.tfms(image)
        return image, label

    def __len__(self):
        return len(self.infos[0])

class PredImages(torch.utils.data.Dataset):
    def __init__(self, image_list: list, label_num: int, config=None):
        self.tfms = tfms['test']
        self.image_list = image_list
        self.num_classes = label_num
        self.config = config

    def get_raw_data(self, idx):
        assert 0 <= idx and idx < len(self.image_list), f"idx:{idx} out of range"

        image = cv2.cvtColor(self.image_list[idx], cv2.COLOR_BGR2RGB)
        assert image.dtype == np.uint8, f"{image.dtype=}"
        assert image.ndim == 3, str(image.shape)
        assert image.shape[2] == 3, f"image channel is {image.shape[2]} not 3"

        label = torch.zeros(self.num_classes)
        return image, label

    def __getitem__(self, idx):
        try:
            image, label = self.get_raw_data(idx)

            image = image.transpose((2, 0, 1)).astype(np.float32) / 255
            image = torch.from_numpy(image).contiguous()
            image = self.tfms(image)
            return image, label

        except Exception as e:
            logging.exception(sys.exc_info())
            logging.error("Exception Error：", self.info[idx])
            return torch.from_numpy(np.array([0])).contiguous(), ""

    def __len__(self):
        return len(self.image_list)


class PredDataset(torch.utils.data.Dataset):
    def __init__(self, infos, label_list, mode, config=None):
        """
        Args:
            infos: List of dict. Necessary keys: image_path. Optional Keys: label_path
        """
        assert mode in ["train", "val", "test"], mode
        self.ia_tfms = ia_tfms[mode]
        self.tfms = tfms[mode]
        self.info = infos
        self.num_classes = len(label_list)
        self.label2id = {key: i for i, key in enumerate(label_list)}
        self.config = config
        print("类别标签:", self.label2id)

    def get_raw_data(self, idx, test=False):

        info = self.info[idx]
        image_path = os.path.join(info['image_path']["volume"], info['image_path']["key"])
        image = read_image_with_protocol(image_path, shape=self.config["SHAPE"], test=test)
        assert image.dtype == np.uint8
        assert image.ndim == 3, str(image.shape)

        # RGBA -> RGB
        if image.shape[2] == 4:
            image = image[:, :, :3]

        assert image.shape[2] == 3, "%s has shape: %s" % (info["image_path"], str(image.shape))

        label = torch.zeros(self.num_classes)

        for l in info["label"]:
            label[self.label2id[l]] = 1.0

        return image, label

    def __getitem__(self, idx):
        try:
            image, label = self.get_raw_data(idx, test=False)

            # #数据增强
            # image = DataAug.box_mask(image, [np.random.randint(int(image.shape[1]*0.1),image.shape[1]), np.random.randint(int(image.shape[0]*0.1),image.shape[0])],
            #     [np.random.randint(image.shape[1]*0.1, image.shape[1]*0.15), np.random.randint(image.shape[0]*0.1, image.shape[0]*0.15)])
            # image = self.ia_tfms(image=image)
            # 验证输入网络的原始图片输入 #
            # cv2.imwrite("/mnt/coyote/wliu/package/disease_analysis/MDC-6DC-1-7-3/__cache__/aug/{0}_{1}".format(idx, self.info[idx]['image_path']["key"]),
            #             cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # ##########################
            image = image.transpose((2, 0, 1)).astype(np.float32) / 255
            image = torch.from_numpy(image).contiguous()
            image = self.tfms(image)
            return image, label

        except Exception as e:
            print("异常信息：", self.info[idx])
            return torch.from_numpy(np.array([0])).contiguous(), ""

    def __len__(self):
        return len(self.info)


#sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#ia_tfms = {
#    "train": iaa.Sequential([
#        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),    # 高斯噪点
#        # sometimes(iaa.MultiplyAndAddToBrightness(mul=(0.3, 2.0), add=(-100, 100))),  # 亮度变化
#        # sometimes(iaa.MultiplyHueAndSaturation((0.9, 1.2), per_channel=True)),
#        # sometimes(iaa.AddToHueAndSaturation((-20, 20), per_channel=True)),           # 饱和度变化
#        # sometimes(iaa.ChangeColorTemperature((1000, 1020))),                         # 色温--->变化原理待查看
#        # iaa.GammaContrast((0.8, 1.2), per_channel=True),                               # 对比度
#        sometimes(
#            iaa.Affine(scale={"x": (0.85, 1.2), "y": (0.85, 1.2)}, translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
#                       rotate=(-180, 180), shear=(-10, 10))),
#        sometimes(iaa.CropAndPad(percent=(-0.3, 0.1))),
#        iaa.Resize((512, 512)),
#        iaa.Identity(),
#    ], random_order=False),
#    "val": iaa.Identity(),
#    "test": iaa.Identity(),
#}

tfms = {
    'train': T.Compose([
        # T.Lambda(lambda x: TF.resize(x, shape_size)),
        # T.RandomCrop(shape_size),
        T.RandomRotation(90),
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': T.Compose([
        # T.Lambda(lambda x: TF.resize(x, shape_size)),
        # T.CenterCrop(shape_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': T.Compose([
        # T.Lambda(lambda x: TF.resize(x, shape_size)),
        # T.CenterCrop(shape_size),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
}
    
