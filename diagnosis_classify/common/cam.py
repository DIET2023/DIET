import torch as t
import torchvision.models.vgg
import torchvision.models
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF

tfms = {
    'test': T.Compose([
        T.Lambda(lambda x: TF.resize(x, 224)),
        T.CenterCrop(224),
    ])
}


class FeatureExtractor():
    """
    1. 提取目标层特征
    2. register 目标层梯度
    """

    def __init__(self, model, target_layers):
        self.model = model
        # self.model_features = model.features
        self.target_layers = target_layers
        self.gradients = list()

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_activations = list()
        self.gradients = list()
        #        for name, module in self.model_features._modules.items():  # 遍历的方式遍历网络的每一层
        #            x = module(x)  # input 会经过遍历的每一层
        #            if name in self.target_layers:  # 设个条件，如果到了你指定的层， 则继续
        #                x.register_hook(self.save_gradient)  # 利用hook来记录目标层的梯度
        #                target_activations += [x]  # 这里只取得目标层的features
        # assert isinstance(self.model, torch.nn.DataParallel)
        x = self.model.cnn_model.conv1(x)
        x = self.model.cnn_model.bn1(x)
        x = self.model.cnn_model.relu(x)
        x = self.model.cnn_model.maxpool(x)

        x = self.model.cnn_model.layer1(x)
        x = self.model.cnn_model.layer2(x)
        x = self.model.cnn_model.layer3(x)
        # x.register_hook(self.save_gradient)
        # target_activations += [x]
        x = self.model.cnn_model.layer4(x)
        x.register_hook(self.save_gradient)
        target_activations += [x]

        x = self.model.cnn_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.cnn_model.fc(x)
        return target_activations, x,


class GradCam():
    """
    GradCam主要执行
    1.提取特征（调用FeatureExtractor)
    2.反向传播求目标层梯度
    3.实现目标层的CAM图
    """

    def __init__(self, model, target_layer_names):
        self.model = model
        self.extractor = FeatureExtractor(self.model, target_layer_names)
        self.ori_image_path_list = []
        self.save_image_path_list = []

    def forward(self, input):
        return self.model(input)

    def __call__(self, input):
        features, output = self.extractor(input)  # 这里的feature 对应的就是目标层的输出， output是图像经过分类网络的输出
        cams = []
        # print("type(output):", type(output), output)
        assert isinstance(output, torch.Tensor)
        for cls_id in range(output.shape[-1]):
            # one_hot = output.max()  # 取1000个类中最大的值
            one_hot = output[0, cls_id]

            # self.model.features.zero_grad()  # 梯度清零
            # self.model.classifier.zero_grad()  # 梯度清零
            self.model.zero_grad()  # 梯度清零
            one_hot.backward(retain_graph=True)  # 反向传播之后，为了取得目标层梯度
            # print('self.extractor.get_gradients() :\n{}'.format(type(self.extractor.get_gradients()[-1])))
            grad_val = self.extractor.get_gradients()[-1].data.cpu().numpy()
            # 调用函数get_gradients(),  得到目标层求得的梯

            target = features[-1]
            # features 目前是list 要把里面relu层的输出取出来, 也就是我们要的目标层 shape(1, 512, 14, 14)
            target = target.data.cpu().numpy()[0, :]  # (1, 512, 14, 14) > (512, 14, 14)
            weights = np.mean(grad_val, axis=(2, 3))[0, :]  # array shape (512, ) 求出relu梯度的 512层 每层权重

            cam = np.zeros(target.shape[1:])  # 做一个空白map，待会将值填上
            # (14, 14)  shape(512, 14, 14)tuple  索引[1:] 也就是从14开始开始

            # for loop的方式将平均后的权重乘上目标层的每个feature map， 并且加到刚刚生成的空白map上
            for i, w in enumerate(weights):
                cam += w * target[i, :, :]
                # w * target[i, :, :]
                # target[i, :, :] = array:shape(14, 14)
                # w = 512个的权重均值 shape(512, )
                # 每个均值分别乘上target的feature map
                # 在放到空白的14*14上（cam)
                # 最终 14*14的空白map 会被填满
            cam = cv2.resize(cam, (input.shape[2], input.shape[3]))  # 将14*14的featuremap 放大回224*224
            cams.append(cam)
        total_max = max([np.max(cam) for cam in cams])
        total_min = min([np.min(cam) for cam in cams])
        for i in range(len(cams)):
            #            cam = cam - np.min(cam)
            #            cam = cam / np.max(cam)
            cams[i] = (cams[i] - total_min) / (total_max - total_min + 1e-6)

        return cams

    def show_cam_on_image_new(self, img, mask, img_shape):
        """
        img: resize之后的图像
        msk: 经过cam之后的图形
        img_shape: 原始图像shape（需要还原到原始图形进行显示）
        注:
        """
        # if type(img) == torch.Tensor:
        if isinstance(img, torch.Tensor):  # 由tensor转成numpy格式
            img: torch.Tensor
            img = img.numpy()
        if img.shape[0] == 3:  # 将通道数放后面
            img: np.array
            # image = image.transpose((2, 0, 1)).astype(np.float32) / 255
            img = np.transpose(img, (1, 2, 0))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)  # 利用色彩空间转换将heatmap凸显
        heatmap = np.float32(heatmap)  # 归一化
        cam_color = (heatmap + img) * 0.5  # 将heatmap 叠加到原图
        new_mask = np.transpose(np.array([mask] * 3), (1, 2, 0))
        cam_alpha = img * new_mask
        cam_color = cv2.resize(cam_color, (img_shape[1], img_shape[0])).astype("uint8")
        cam_alpha = cv2.resize(cam_alpha, (img_shape[1], img_shape[0])).astype("uint8")
        # print('cam: \n{}'.format(cam.shape))
        return cam_color, cam_alpha

    def concatenate_ori_image_mask_list(self, img, mask_list, ori_img):
        """
        img: resize之后的图像
        mask_list: 经过cam之后的图形（每一类对应一个cam图）
        ori_img: 原始图像（需要还原到原始图形进行显示）
        return: 返回水平拼接之后的图像
        注: img是由cv2读取的原始图像经过resize到输入到神经网络模型的图像的shape,不需要进行归一化
        ori_img是由cv2读取的原始图像
        """
        cam_color_imgs = []
        cam_alpha_imgs = []
        for mask in mask_list:
            color_cam, alpha_cam = self.show_cam_on_image_new(img, mask, ori_img.shape)
            # cam_image_path = os.path.join(res_path, 'cam_'+image_name)
            # ori_image_path = os.path.join(res_path, 'ori_'+image_name)
            cam_color_imgs.append(color_cam)
            cam_alpha_imgs.append(alpha_cam)
        color_hstack_image = np.concatenate([ori_img] + cam_color_imgs, axis=1)  # 水平拼接
        alpha_hstack_image = np.concatenate([ori_img] + cam_alpha_imgs, axis=1)  # 水平拼接
        hstack_image = np.concatenate([color_hstack_image, alpha_hstack_image], axis=0)

        return hstack_image

    def list_to_str(self, output_list):
        """
        将list中的元素转换成字符类型数据
        """
        res = None
        for i in output_list:
            if res == None:
                res = str(i) + '_'
            else:
                res = res + str(i) + '_'
        return res

    def save_mapping_table(self, save_path):
        """
        存储原路径与保存路径的映射表
        """
        save_path = os.path.join(save_path, 'map_table')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path_map_tab = os.path.join(save_path, 'map_table.csv')

        res_df = pd.DataFrame()
        res_df.insert(loc=0, value=self.save_image_path_list, column='save image path')
        res_df.insert(loc=0, value=self.ori_image_path_list, column='ori image path')
        res_df.to_csv(save_path_map_tab, index=False)



