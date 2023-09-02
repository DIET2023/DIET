

import os
import json
from pathlib import Path
from pprint import pprint

import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torchvision.transforms.functional as TF
import numpy as np
import imageio
import torch
from torch import nn
from torch.utils.data import DataLoader

import copy
import pandas as pd

from common.data import load_dataset_info_jsonl, PredDataset
from common.model import ClassificationModel
from common.metrics import dice, recall, precision, fbeta, fa
from common.utils import read_tsv, write_tsv, get_maskMerged_image
from getConfig import config
import logging

EVAL_DIR = os.path.join('./', config["EVAL_EPOCH"])
if not os.path.exists(EVAL_DIR):
    os.mkdir(EVAL_DIR)

LABEL_LIST = [i.strip() for i in open(config["LABEL_LIST"])]
LABEL2ID = {key: i for i, key in enumerate(LABEL_LIST)}
ID2LABEL = {i: key for i, key in enumerate(LABEL_LIST)}

NUM_CLASSES = len(LABEL_LIST)
print(f"标签数据集：{LABEL2ID}--> {ID2LABEL} ")


class FeatureExtractor():
    """
    1. 提取目标层特征
    2. register 目标层梯度
    """

    def __init__(self, model, target_layers=None):
        self.model = model
        # self.model_features = model.features
        self.target_layers = target_layers
        self.gradients = list()

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        # target_activations = list()
        # self.gradients = list()

        x = self.model.cnn_model.conv1(x)
        x = self.model.cnn_model.bn1(x)
        x = self.model.cnn_model.relu(x)
        x = self.model.cnn_model.maxpool(x)

        x = self.model.cnn_model.layer1(x)
        x = self.model.cnn_model.layer2(x)
        x = self.model.cnn_model.layer3(x)

        x = self.model.cnn_model.layer4(x)
        # x.register_hook(self.save_gradient)
        # target_activations += [x]
        feature_vector = x
        x = self.model.cnn_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.model.cnn_model.fc(x)
        return feature_vector, x


def predict(dataset_info, iteration, device):
    eval_dir = Path(EVAL_DIR)
    ckpoint = "model/model_%08d.ckpt" % iteration
    model = ClassificationModel.load_from_checkpoint(ckpoint, num_classes=NUM_CLASSES)
    model = model.to(device)
    model.eval()
    # #######################################################################################
    # 抽取网络模型中的部分层: wliu: 2021.07.09
    # extract_model = FeatureExtractor(model)
    # image_path = []          # 保存图片文件路径
    # image_feature = []       # 保存图片feature
    # #######################################################################################

    logging.info("Loading Finish %s" % ckpoint)
    error_file = open("error_file.txt", "w", encoding="utf-8")

    for dataset_name, path in dataset_info:
        outputdir = eval_dir / str(iteration) / dataset_name / "prediction"

        if os.path.exists(outputdir):
            logging.info("%s exists. Jump!" % outputdir)
            continue
        outputdir.mkdir(parents=True)

        data_infos = load_dataset_info_jsonl([path])
        dataset = PredDataset(data_infos, label_list=LABEL_LIST, mode="test", config=config)

        batch_size = 1
        testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                                                 num_workers=16)
        preds_list = []
        output_data_infos = []
        for i, (images, labels) in tqdm(enumerate(testloader), total=len(testloader)):
            try:
                images,  = images.to(device), labels.to(device)
                with torch.no_grad():
                    preds = model(images).sigmoid()
                    # feature_vectors, _ = extract_model(images)

                # for pred, feature_vector in zip(preds, feature_vectors):
                for pred in preds:
                    preds_list.append({"predictions": pred.cpu().numpy().tolist()})
                    output_data_infos.append(data_infos[i])

                    # ######################################################################################
                    # image_path.append(data_infos[i])
                    # image_feature.append(feature_vector.cpu().numpy())
                    # ######################################################################################
            except Exception as e:
                logging.error(f"I = {i}, labels = {labels}, data_info = {data_infos[i]}")
                error_file.write(f"{data_infos[i]}\n")

        # print("图片路径:", image_path)
        # print("图片feature:", image_feature)
        #
        # image_feature_concate = np.array([image_path, image_feature])
        # np.save("data/image_avg_feature_map_train.npy", image_feature_concate)
        # exit()

        res = []
        len_preds_list = len(preds_list)

        assert len(output_data_infos) == len(preds_list), f"{len(data_infos)} != {len(preds_list)}"
        for i in range(len_preds_list):
            res.append((output_data_infos[i], preds_list[i]))

        logging.debug(f"[{dataset_name}] predict result save at {outputdir}")
        write_tsv(res, outputdir / "prediction.tsv")

def calc_single_curve(labels, scores):
    assert len(labels) == len(scores), f"{len(labels)} != {len(scores)}"

    def _dict_merge(d1, d2):
        return dict(list(d1.items()) + list(d2.items()))

    def _calc_metric(_tp, _fp, _tn, _fn):
        eps = 1e-8
        total = _tp + _fp + _tn + _fn
        gt_p, inf_p = _tp + _fn, _tp + _fp
        gt_n, inf_n = _tn + _fp, _tn + _fn
        ret = {'TP': _tp, 'FP': _fp, "TN": _tn, "FN": _fn}
        ret['accuracy'] = (_tp + _tn) / (total + eps)
        ret['recall'] = _tp / (gt_p + eps)
        ret['precision'] = _tp / (inf_p + eps)
        ret['FPR'] = _fp / (gt_n + eps)
        ret['TNR'] = _tn / (gt_n + eps)
        ret['f1_score'] = 2.0 / (1 / (ret['recall'] + eps) + 1 / (ret['precision'] + eps))
        pe = 1.0 * (gt_p * inf_p + gt_n * inf_n) / (total ** 2 + eps)
        ret['kappa'] = (ret['accuracy'] - pe) / (1 - pe + eps)
        return ret

    N = len(labels)
    curve_result = []
    ls_pair = sorted([(scores[i], labels[i]) for i in range(N)], key=lambda p: p[0])

    thr = 1.0
    start_i = N - 1
    total_p = len([i for i in range(N) if labels[i] > 0])
    total_n = N - total_p
    tp, fp = 0, 0
    tn = total_n
    fn = total_p
    i = N - 1
    while(i >= -1):
        if i >= 0 and ls_pair[i][0] >= thr:
            tp += int(ls_pair[i][1] > 0)
            fn -= int(ls_pair[i][1] > 0)
            fp += int(ls_pair[i][1] <= 0)
            tn -= int(ls_pair[i][1] <= 0)
            i -= 1
        else:
            curve_result.append(_dict_merge({'thr': thr}, _calc_metric(tp, fp, tn, fn)))
            if i >= 0:
                thr = ls_pair[i][0]
            else:
                break
    return curve_result

def summary_curve(curve_list):
    def _key_max_i(find_key, curve=curve_list, rev=False):
        max_i = 0
        for i in range(1, len(curve)):
            if not rev:
                if curve[i][find_key] > curve[max_i][find_key]:
                    max_i = i
            else:
                if curve[i][find_key] < curve[max_i][find_key]:
                    max_i = i
        return max_i

    def _key_thr_i(find_thr, curve=curve_list):
        thr_i_pairs = sorted([(i, curve[i]['thr']) for i in range(len(curve))], key=lambda p:p[1])
        for i in range(len(thr_i_pairs)):
            if thr_i_pairs[i][1] >= find_thr:
                return thr_i_pairs[i][0]
        return thr_i_pairs[-1][0]

    def calc_auc(curve=curve_list):
        pass

    def calc_ap(curve=curve_list):
        pass

    ret = {}
    f1_max_i = _key_max_i('f1_score')
    ret['f1.max.score'] = curve_list[f1_max_i]['f1_score']
    ret['f1.max.thr'] = curve_list[f1_max_i]['thr']
    ret['f1.max.recall'] = curve_list[f1_max_i]['recall']
    ret['f1.max.precision'] = curve_list[f1_max_i]['precision']
    kappa_max_i = _key_max_i('kappa')
    ret['kappa.max.score'] = curve_list[kappa_max_i]['kappa']
    ret['kappa.max.thr'] = curve_list[kappa_max_i]['thr']
    acc_max_i = _key_max_i('accuracy')
    ret['acc.max.score'] = curve_list[acc_max_i]['accuracy']
    ret['acc.max.thr'] = curve_list[acc_max_i]['thr']

    thr_050_i = _key_thr_i(0.5)
    ret['f1.thr050.score'] = curve_list[thr_050_i]['f1_score']
    ret['kappa.thr050.score'] = curve_list[thr_050_i]['kappa']
    ret['acc.thr050.score'] = curve_list[thr_050_i]['accuracy']

    return ret

def evaluate_zkwei_ver(dataset_info, iteration, force):
    common_threshold = config["COMMON_THRESHOLD"]
    for dataset_name, path in dataset_info:
        outputdir = Path(EVAL_DIR) / str(iteration) / dataset_name / 'evaluation'
        outputdir.mkdir(parents=True, exist_ok=True)
        output_path = outputdir / "eval.txt"

        pred_path = Path(EVAL_DIR) / str(iteration) / dataset_name / "prediction" / "prediction.tsv"
        preds = read_tsv(pred_path)

        labels = np.zeros([len(preds), NUM_CLASSES])
        scores = np.zeros([len(preds), NUM_CLASSES])

        for i, (data_info, pred) in enumerate(preds):
            for l in data_info["label"]:
                labels[i, LABEL2ID[l]] = 1.0
            scores[i] = pred["predictions"]

        res_curve = {}
        res_summary = {}
        for i in tqdm(range(NUM_CLASSES)):
            curve = calc_single_curve(labels[:, i], scores[:, i])
            curve_summary = summary_curve(curve)
            res_curve[ID2LABEL[i]] = curve
            res_summary[ID2LABEL[i]] = curve_summary
        # ALL
        #res_df.to_csv(outputdir / "res_eval.csv", index=False)

        #pprint(contents)
        #pprint(res_summary)
        output_curve_path = outputdir / 'curve.json'
        output_curve_summary_path = outputdir / 'curve_summary.json'
        with open(output_curve_path, "w") as f:
            f.write(json.dumps(res_curve, indent=2) + "\n")

        with open(output_curve_summary_path, "w") as f:
            f.write(json.dumps(res_summary, indent=2) + "\n")

def evaluate(dataset_info, iteration, force):
    common_threshold = config["COMMON_THRESHOLD"]
    for dataset_name, path in dataset_info:
        outputdir = Path(EVAL_DIR) / str(iteration) / dataset_name / 'evaluation'
        if outputdir.exists():
            logging.info("%s exists. Jump evaluation!" % outputdir)
            continue
        outputdir.mkdir(parents=True)
        output_path = outputdir / "eval.txt"

        pred_path = Path(EVAL_DIR) / str(iteration) / dataset_name / "prediction" / "prediction.tsv"
        preds = read_tsv(pred_path)

        labels = np.zeros([len(preds), NUM_CLASSES])
        scores = np.zeros([len(preds), NUM_CLASSES])

        for i, (data_info, pred) in enumerate(preds):
            for l in data_info["label"]:
                labels[i, LABEL2ID[l]] = 1.0
            scores[i] = pred["predictions"]

        predicted_positive = scores >= common_threshold
        num_predicted_positive = np.sum(predicted_positive, axis=0)
        true_positive = predicted_positive * labels
        true_negtive = (1 - predicted_positive) * (1 - labels)
        num_true_positive = np.sum(true_positive, axis=0)
        num_true_negtive = np.sum(true_negtive, axis=0)
        num_gt_positive = np.sum(labels, axis=0)
        num_gt_negtive = np.sum(1 - labels, axis=0)

        contents = []
        res = []
        for i in range(NUM_CLASSES):
            recall = num_true_positive[i] / num_gt_positive[i]
            precision = num_true_positive[i] / num_predicted_positive[i]
            acc = (num_true_positive[i] + num_true_negtive[i]) / len(preds)
            tnr = num_true_negtive[i] / num_gt_negtive[i]
            f1_score = 2 * recall * precision / (recall + precision)
            info = {
                "category": LABEL_LIST[i],
                "recall": recall,
                "precision": precision,
                "tnr": tnr,
                "f1_score": f1_score,
            }
            contents.append(info)
            res.append(copy.deepcopy([LABEL_LIST[i], recall, precision, tnr, f1_score]))
            print("{}: acc{}".format(i, acc))
            print("{}: info{}".format(i, info))

        # ALL
        recall = np.sum(num_true_positive) / np.sum(num_gt_positive)
        precision = np.sum(num_true_positive) / np.sum(num_predicted_positive)
        tnr = np.sum(num_true_negtive) / np.sum(num_gt_negtive)
        f1_score = 2 * recall * precision / (recall + precision)
        info = {
            "category": "all",
            "recall": recall,
            "precision": precision,
            "tnr": tnr,
            "f1_score": f1_score,
        }
        contents.append(info)

        res.append(copy.deepcopy(['all', recall, precision, tnr, f1_score]))
        res_df = pd.DataFrame(data=res)
        print('res_df: \n{}'.format(res_df))

        res_df.to_csv(outputdir / "res_eval.csv", index=False)

        pprint(contents)
        with open(output_path, "w") as f:
            f.write(json.dumps(contents, indent=2) + "\n")


def _stampText(image, text, line):
    """自适应字体"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    margin = 5
    thickness = 2
    color = (255, 255, 255)

    size = cv2.getTextSize(text, font, font_scale, thickness)

    text_width = size[0][0]
    text_height = size[0][1]
    line_height = text_height + size[1] + margin

    x = image.shape[1] - margin - text_width
    y = margin + size[0][1] + line * line_height
    image = np.asarray(image)

    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    return image


def save_map_table(res_path, ori_path_list, save_path_list, num):
    """

    :param res_path: 映射表保存路径
    :param ori_path: 原始图片路径
    :param save_path: 保存图片路径
    :param num: 映射表条目
    :return:
    """
    map_table_path = os.path.join(res_path, 'map_table_{}.txt'.format(num))
    assert len(ori_path_list) == len(save_path_list), print('图像路径不匹配')
    with open(map_table_path, "w", encoding="utf-8") as file:
        for ori_path, save_path in zip(ori_path_list, save_path_list):
            item = {'ori_path': ori_path, 'save_path': save_path}
            item = str(item).strip('\n').replace("'", "\"")
            file.write(item + '\n')


def visualize(dataset_info, iteration):
    logging.info("Start Visualizing: {}".format(dataset_info))
    FONT = ImageFont.truetype("wqy-microhei", 15)  # sudo apt-get install ttf-wqy-microhei
    common_threshold = config["COMMON_THRESHOLD"]
    ori_path_list = []
    save_path_list = []
    for dataset_name, path in dataset_info:
        input_path = Path(EVAL_DIR) / str(iteration) / dataset_name / 'prediction' / "prediction.tsv"
        outputdir = Path(EVAL_DIR) / str(iteration) / dataset_name / 'visualization'

        if os.path.exists(outputdir):
            logging.info("%s exists. Jump!" % outputdir)
            continue

        outputdir.mkdir(parents=True)
        imagedir = outputdir / "data"
        relative_imagedir = os.path.join('../../', 'data')  # 相对路径

        imagedir.mkdir()
        for i in range(NUM_CLASSES):
            (outputdir / str(i) / "gt_positive").mkdir(parents=True)
            (outputdir / str(i) / "gt_negitive").mkdir(parents=True)

        data_infos = load_dataset_info_jsonl([path])
        dataset = PredDataset(data_infos, label_list=LABEL_LIST, mode="test", config=config)
        predict_infos = read_tsv(input_path)

        for i, (data_info, pred_info) in enumerate(predict_infos):
            data_info["index"] = i

        logging.info("类标签LABEL2ID: {}".format(LABEL2ID))
        logging.info("ID2LABEL: {}".format(ID2LABEL))

        for j in range(NUM_CLASSES):
            predict_infos.sort(key=lambda x: x[1]["predictions"][j])
            interval = 1

            for data_info, pred_info in tqdm(predict_infos[::interval]):
                try:

                    scores = pred_info["predictions"]
                    labels = [0] * NUM_CLASSES

                    for l in data_info["label"]:
                        labels[LABEL2ID[l]] = 1.0

                    input_image_name: str = data_info["image_path"]["key"]
                    if '/' in input_image_name:
                        input_image_name = input_image_name.split('/')[-1]  # 去掉input_image_name中文件分隔符，从而取出最终的名称
                    output_image_path = os.path.join(imagedir, input_image_name)
                    output_relative_image_path = os.path.join(relative_imagedir, input_image_name)

                    if not os.path.exists(output_image_path):
                        # 保存映射表 start
                        ori_path = os.path.join(data_info['image_path']['volume'], data_info['image_path']['key'])
                        ori_path_list.append(ori_path)
                        # 保存映射表 end

                        save_path_list.append(output_image_path)  # 文件保存的地址

                        data_transformed = dataset[data_info["index"]][0].cpu().numpy().transpose([1, 2, 0])
                        data_transformed *= [0.229, 0.224, 0.225]
                        data_transformed += [0.485, 0.456, 0.406]
                        data_transformed *= 255
                        data_transformed = data_transformed.astype(np.uint8)
                        data_transformed_shape = data_transformed.shape

                        data_ori, _ = dataset.get_raw_data(data_info["index"], test=True)
                        data_ori = np.array(data_ori)
                        data_ori_shape = data_ori.shape

                        height = 1024
                        if data_ori_shape[0] > height:
                            scale_ratio = height / data_ori_shape[0]
                            width = int(data_ori_shape[1] * scale_ratio)
                            data_ori = cv2.resize(data_ori, (width, height))
                            black = np.zeros([data_ori.shape[0], int(data_ori.shape[1] * scale_ratio), 3],
                                             dtype=np.uint8)
                            if data_ori.shape[0] < 512 or data_ori.shape[1] < 512:
                                data_ori = cv2.resize(data_ori, (512, 512))
                                black = np.zeros([512, 512, 3], dtype=np.uint8)
                        elif data_ori.shape[0] < 512 or data_ori.shape[1] < 512:
                            data_ori = cv2.resize(data_ori, (512, 512))
                            black = np.zeros([512, 512, 3], dtype=np.uint8)
                        else:
                            black = np.zeros([data_ori.shape[0], data_ori.shape[1], 3], dtype=np.uint8)

                        black_data_transformed = np.zeros([data_ori.shape[0], data_ori.shape[1], 3], dtype=np.uint8)
                        # print(black_data_transformed.shape, "--->", data_transformed.shape)

                        black_data_transformed[:data_transformed_shape[0], :data_transformed_shape[1],
                        :3] = data_transformed
                        image = np.concatenate([black, data_ori, black_data_transformed], axis=1)

                        for k in range(NUM_CLASSES):
                            if scores[k] >= common_threshold and ID2LABEL[k] in data_info["label"]:
                                # 预测标签在标签列表中，标记为蓝色
                                font_color = (0, 255, 0)
                            elif scores[k] >= common_threshold and ID2LABEL[k] not in data_info["label"]:
                                # 预测标签不在标签列表中，标记为红色，说明有错误
                                font_color = (255, 0, 0)
                            elif scores[k] < common_threshold and ID2LABEL[k] in data_info["label"]:
                                # 非预测标签在标签列表中，标记为红色，说明预测有误
                                font_color = (255, 0, 0)
                            else:  # 非预测标签不在标签列表中，标记为蓝色，预测正确
                                font_color = (0, 0, 255)

                            image = cv2.putText(image, LABEL_LIST[k], (10, 30 + k * 70 + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6, (255, 255, 255), 2)
                            image = cv2.putText(image, "gt     : %d" % labels[k], (10, 30 + k * 70 + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            image = cv2.putText(image, "predict: %0.4f" % scores[k], (10, 30 + k * 70 + 50),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, font_color, 2)
                        imageio.imwrite(output_image_path, image)

                    if labels[j]:

                        os.symlink(output_relative_image_path, outputdir / str(j) / "gt_positive" / (
                                "%0.4f_%08d.jpg" % (scores[j], data_info["index"])))
                    else:
                        os.symlink(output_relative_image_path, outputdir / str(j) / "gt_negitive" / (
                                "%0.4f_%08d.jpg" % (scores[j], data_info["index"])))

                except Exception as e:
                    print(data_info["index"], data_info)

        save_map_table(res_path=outputdir, ori_path_list=ori_path_list, save_path_list=save_path_list,
                       num=len(save_path_list))


def main(FLAGS):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #dataset_info = [(config["MODE"], config["TRAIN_FILE"])] if config["MODE"] == "train" else [
    #    (config["MODE"], config["TEST_FILE"])]
    dataset_info = [(ds_name, config["TEST_FILE"][ds_name]) for ds_name in config['TEST_FILE']]
    print(dataset_info)
    # #########################################################################################
    # 原命令行：
    # for iteration in config["VALITION_METRICE"]:
    # 批量验证模型性能
    for i in range(0, 1):
    # 指定轮次下模型
    #     iteration = 16 * 1000
        iteration = FLAGS.iteration + i * 1000
        predict(dataset_info, iteration, device)
        evaluate(dataset_info, iteration, FLAGS.force)
        evaluate_zkwei_ver(dataset_info, iteration, FLAGS.force)

    # #########################################################################################评估一系列的模型指标
    # predict(dataset_info, FLAGS.iteration, device)
    if FLAGS.run == "predict":
        return

    # evaluate(dataset_info, FLAGS.iteration, FLAGS.force)
    if FLAGS.run == "evaluate":
        return

    visualize(dataset_info, FLAGS.iteration)
    if FLAGS.run == "visualize":
        return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", default="evaluate")
    parser.add_argument("--iteration", "-i", type=int, required=False)
    parser.add_argument("--force", "-f", default=False, action="store_true")
    FLAGS = parser.parse_args()
    assert FLAGS.run in ["predict", "evaluate", "visualize"]
    main(FLAGS)
