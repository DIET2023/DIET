#!/usr/bin/env python
# -*- coding: utf-8 -*-



import os
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

# LABEL_LIST = [i.strip() for i in open("data/label.list")]  # 需要给与一个label list文件
# LABEL2ID = {key: i for i, key in enumerate(LABEL_LIST)}  # label到ID的转换
LABEL_LIST = ['ance', 'acne_rosacea_sd', 'others', 'urticaria', 'vitiligo', 'yinxie']  # 需要给与一个label list文件
LABEL2ID = {key: i for i, key in enumerate(LABEL_LIST)}  # label到ID的转换


class RocCurve():
    def __init__(self, res_save_path):

        if not os.path.exists(res_save_path):
            os.mkdir(res_save_path)
        self.res_save_path = res_save_path
        self.LABEL2ID = LABEL2ID
        self.LABEL_LIST = LABEL_LIST

    def write_jsonl(self, jsons, path):
        with open(path, "w") as f:
            for unit in jsons:
                f.write(json.dumps(unit, ensure_ascii=False) + "\n")

    def read_jsonl(self, path):
        coll = []
        for line in open(path):
            coll.append(json.loads(line.strip()))
        return coll

    def read_tsv(self, path):

        # 数据格式：
        # {"image_format": "image", "image_path": {"protocol": "DIR", "volume": "/mnt/coyote/yzchen/data/spider/haodf/20210221/210422/resized_data",
        # "key": "patients_haofei_with18_patients_6450316262_images_0f04ab8b693c303e3c73b35a66c1df4f.jpg"}, "label": ["urticaria"],
        # "phase": "train", "from": {"type": "spider", "doctor": "haofei", "key": "doctor chat", "ori_keyword": "荨麻疹"}, "remark": "荨麻疹"}
        # {"predictions": [0.006921885069459677, 0.9994198083877563, 0.0007900372729636729, 0.032382942736148834, 0.003952509723603725, 0.02513154223561287]}

        coll = []
        for line in open(path, encoding='utf-8'):
            line = line.strip()
            segments = []
            for seg in line.split("\t"):
                segments.append(json.loads(seg))
            coll.append(segments)

        return coll

    def write_tsv(self, tsv, path):
        with open(path, "w") as f:
            for line in tsv:
                f.write("\t".join(json.dumps(seg, ensure_ascii=False) for seg in line) + "\n")

    def read(self, file_path: str, file_type='tsv'):
        if not os.path.exists(file_path):
            print('current file_path\n{}'.format(file_path))
            raise FileNotFoundError

        if file_type == 'tsv':
            preds = self.read_tsv(file_path)

        else:
            print('current file_type\n{}'.format(file_type))
            raise NotImplementedError

        # print('preds\n{}'.format(preds[:1]))

        return preds

    def get_labels_predict_from_tsv(self, preds, class_num):
        # LABEL_LIST = [i.strip() for i in open("data/label.list")]  # 需要给与一个label list文件
        # LABEL2ID = {key: i for i, key in enumerate(LABEL_LIST)}  # label到ID的转换
        # ID2LABEL = {i: key for i, key in enumerate(LABEL_LIST)}

        labels = np.zeros([len(preds), class_num])
        scores = np.zeros([len(preds), class_num])
        for i, (data_info, pred) in enumerate(preds):
            for l in data_info['label']:
                if l == "yinxie":
                    pass
                else:
                    labels[i, self.LABEL2ID[l]] = 1.0
            if class_num > len(pred["predictions"]):  # 无
                pred["predictions"].append(0)
            # #后期可删：暂时对比 1-7-1 & 1-3-2 ############### wliu 2021.07.08
            if len(pred["predictions"]) > class_num:
                pred["predictions"] = pred["predictions"][:4] + [pred["predictions"][-1]]
            scores[i] = pred["predictions"]

        return labels, scores

    def get_labels_predict(self, preds, class_num, file_type='tsv'):
        """
            可以在此函数中增加新类型文件的数据获取
        """
        if file_type == 'tsv':
            labels, scores = self.get_labels_predict_from_tsv(preds=preds, class_num=class_num)
        else:
            raise NotImplementedError

        return labels, scores

    def get_roc_figure(self, predict_file_path: str, class_num):
        """
        默认绘制所有类的roc曲线
        """
        file_type = predict_file_path.split('.')[-1]
        preds = self.read(file_path=predict_file_path, file_type=file_type)
        labels, scores = self.get_labels_predict(preds=preds, class_num=class_num)
        labels: np.array
        scores: np.array
        for i in range(class_num):
            temp_pred = scores[:, i]
            temp_label = labels[:, i]
            print('temp_pred\n{}'.format(temp_pred))
            print('temp_score\n{}'.format(temp_label))
            input('暂停')
            figure_path = os.path.join(self.res_save_path, '{}_roc.png'.format(i))
            self.plotROC(pred=temp_pred, classLabels=temp_label, figure_path=figure_path)

    def get_roc_figure_all_in(self, predict_file_path: str, class_num):
        """
        默认绘制所有类的roc曲线----同一张图
        """
        file_type = predict_file_path.split('.')[-1]
        preds = self.read(file_path=predict_file_path, file_type=file_type)
        labels, scores = self.get_labels_predict(preds=preds, class_num=class_num)
        labels: np.array
        scores: np.array

        # 多张图： start
        all_figure_path = os.path.join(self.res_save_path, 'roc_all.png')
        self.plotROC_all(pred_list=scores, classLabels_list=labels, figure_path=all_figure_path)

    def get_roc_figure_sk(self, predict_file_path: str, class_num):
        """

        :param predict_file_path: prediction文件路径
        :param class_num: 类别数量
        :return:
        """
        file_type = predict_file_path.split('.')[-1]
        title = predict_file_path.split('\\')[-1].split('.')[0]
        preds = self.read(file_path=predict_file_path, file_type=file_type)
        labels, scores = self.get_labels_predict(preds=preds, class_num=class_num)
        labels: np.array
        scores: np.array

        class_num_list = list(range(class_num))
        # roc_curve.plot_roc_sklearn(labels=labels, predict_prob=scores, pos_label_list=class_num_list,
        #                            res_path=res_path)
        roc_curve.plot_roc_sklearn_all_in(labels=labels, predict_prob=scores, pos_label_list=class_num_list,
                                          res_path=res_path, title=title)

    def calAUC(self, prob, labels):
        f = list(zip(prob, labels))
        rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
        rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
        posNum = 0
        negNum = 0
        for i in range(len(labels)):
            if (labels[i] == 1):
                posNum += 1
            else:
                negNum += 1
        auc = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
        return auc

    def get_need_data(self, data: np.array, drop_label='others'):
        """
        去掉对应列的数据
        :param drop_label: 去掉对应列的数据
        :param data: 原始数据标签
        :return:
        """
        if drop_label == None:
            return data
        else:
            drop_id = LABEL2ID[drop_label]
            # print('ori_data\n{}'.format(data[:3, :]))
            new_data = np.delete(data, drop_id, axis=1)
            # print('new_data\n{}'.format(new_data[:3, :]))
            return new_data

    # def get_odd_data(self, ):

    def get_roc_figure_sk_compare(self, predict_file_path_list, class_num, drop_label=None, dtype="roc"):
        # 注意当去掉某一类时，对应的labels和LabelId也要做相应的更改的
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        all_labels_list = []
        all_scores_list = []
        name_list = []
        for predict_file_path in predict_file_path_list[:]:
            file_type = predict_file_path.split('.')[-1]
            name = predict_file_path.split('\\')[-1].split('.')[0]
            preds = self.read(file_path=predict_file_path, file_type=file_type)
            labels, scores = self.get_labels_predict(preds=preds, class_num=class_num)
            labels = self.get_need_data(data=labels, drop_label=drop_label)
            scores = self.get_need_data(data=scores, drop_label=drop_label)
            labels: np.array
            scores: np.array
            all_labels_list.append(labels)
            all_scores_list.append(scores)
            name_list.append(name)

        res_path_roc = os.path.join(res_path, dtype)
        class_num = int(labels.shape[1])

        # ######################################################################################################
        print("混淆矩阵")
        tmp_metric = np.zeros((class_num, class_num + 1))
        data_set = len(all_labels_list[-1])
        counter = 0
        for i in range(data_set):
            real_label, pred_label = np.argmax(all_labels_list[-1][i]), np.argmax(all_scores_list[-1][i])
            if all_scores_list[-1][i][pred_label] >= 0.5 and real_label == pred_label:
                tmp_metric[real_label, pred_label] += 1
                counter += 1
            elif all_scores_list[-1][i][pred_label] < 0.5 and real_label != pred_label:
                tmp_metric[real_label, -1] += 1
            else:
                tmp_metric[real_label, pred_label] += 1

        res_df = pd.DataFrame(data=tmp_metric)
        print('res_df: \n{}'.format(res_df))
        # res_df.to_csv("res_eval.csv", index=False)
        print(np.sum(tmp_metric), f"正确率: {counter / np.sum(tmp_metric)}")
        # #######################################################################################################

        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)
        need_pos_list = list(range(class_num))
        color = ['k', 'm', 'b', 'yellow', 'g', 'c', 'r', 'indigo']
        for label_index in need_pos_list:
            title = LABEL_LIST[label_index]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            plt.figure(figsize=(8, 8))
            for index in range(len(predict_file_path_list)):
                if dtype == "roc":
                    # 原有计算ROC/AUC
                    fpr[index], tpr[index], thresholds = roc_curve(all_labels_list[index][:, label_index],
                                                                   all_scores_list[index][:, label_index], pos_label=1,
                                                                   drop_intermediate=True)

                    # 寻找 0.5点
                    break_thr_05_i = None
                    for i05 in range(thresholds.shape[0] - 1, -1, -1):
                        if thresholds[i05] >= 0.5:
                            break_thr_05_i = i05
                            break

                    roc_auc[index] = auc(fpr[index], tpr[index])
                    false_positive_rate = fpr[index]
                    true_positive_rate = tpr[index]

                    plt.plot(false_positive_rate, true_positive_rate, color[index],
                             label='AUC:{:.3f} thd:{:.2f}, {}'.format(roc_auc[index], thresholds[break_thr_05_i],
                                                                      name_list[index]))

                    plt.plot(false_positive_rate[break_thr_05_i], true_positive_rate[break_thr_05_i],
                             c=color[index], marker="o", markersize=10)
                    plt.scatter(false_positive_rate[break_thr_05_i], true_positive_rate[break_thr_05_i],
                                c=color[index], marker='+')

                    plt.ylabel('TPR')
                    plt.xlabel('FPR')

                # 原有计算ROC/AUC曲线

                elif dtype == "pr":

                    # 计算pr 曲线 #####################################################################################
                    precision, recall, thresholds_pr = precision_recall_curve(all_labels_list[index][:, label_index],
                                                                              all_scores_list[index][:, label_index])

                    # # # 选择最优阈值 ####################################################################################
                    p_r = [[precision[i], recall[i], thresholds_pr[i]] for i in range(thresholds_pr.shape[0])]
                    p_r = np.array(p_r)
                    p_r_fa = 2 / ((1 / p_r[:, 0]) + (1 / p_r[:, 1]))
                    p_r_fa = np.reshape(p_r_fa, (p_r_fa.shape[0], 1))
                    p_r_thd_fa = np.concatenate((p_r, p_r_fa), axis=-1)
                    best_thd = np.argmax(p_r_thd_fa[:, -1])
                    # 选择最优阈值 ####################################################################################

                    # # 计算AP数值 ####################################################################################
                    ap = average_precision_score(all_labels_list[index][:, label_index],
                                                 all_scores_list[index][:, label_index])
                    # ################################################################################################
                    plt.step(recall, precision, color=color[index], alpha=1, where='post',
                             label='ap:{:.3f} p:{:.3f} r:{:.3f}, thd {:.3f} {}'.format(
                                 ap,
                                 precision[best_thd],
                                 recall[best_thd],
                                 thresholds_pr[best_thd],
                                 name_list[index]))
                    plt.scatter(recall[best_thd], precision[best_thd], c=color[index], marker='o')
                    plt.plot(recall[best_thd], precision[best_thd], c=color[index], marker="o", markersize=10)
                    # plt.fill_between(recall, precision, step='post', alpha=0.2, color=color[index])
                    plt.ylabel('Precision')
                    plt.xlabel('Recall')
                    # # # 计算pr 曲线 #####################################################################################                # #

                # Compute micro-average ROC curve and ROC area

                # fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_list[index].ravel(), all_scores_list[index].ravel())
                # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
                #
                # plt.plot(fpr["micro"], tpr["micro"],
                #          label='micro-average ROC curve (area = {0:0.2f})'
                #                ''.format(roc_auc["micro"]),
                #          color=color[index], linestyle=':', linewidth=4)

            # First aggregate all false positive rates
            # from scipy import interp
            # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(predict_file_path_list))]))
            #
            # # Then interpolate all ROC curves at this points
            # mean_tpr = np.zeros_like(all_fpr)
            # for i in range(len(predict_file_path_list)):
            #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # # Finally average it and compute AUC
            # mean_tpr /= len(predict_file_path_list)
            #
            # fpr["macro"] = all_fpr
            # tpr["macro"] = mean_tpr
            # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
            # print(roc_auc["macro"])
            #
            # plt.plot(fpr["macro"], tpr["macro"],
            #          label='macro-average ROC curve(area = {0:0.2f})'
            #                ''.format(roc_auc["macro"]),
            #          color='darkorange', linestyle=':', linewidth=4)

            plt.title(title)
            plt.plot([0, 1], [0, 1], 'r--', label='base_line')
            plt.grid()

            f = plt.gcf()  # 获取当前图像
            figure_path = os.path.join(res_path_roc, '{}_roc_test_{}.png'.format('all', title))
            plt.legend(loc='lower right')
            f.savefig(figure_path)
            plt.show()

    def plot_roc_sklearn(self, labels, predict_prob, res_path, pos_label_list=None):
        if pos_label_list is None:
            pos_label_list = [0]  # 默认第一个类别

        from sklearn.metrics import roc_curve, auc
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        res_path_roc = os.path.join(res_path, 'roc')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        roc_auc = dict()
        for i in pos_label_list:
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predict_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            false_positive_rate = fpr[i]
            true_positive_rate = tpr[i]
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob, pos_label=pos_label)
            # print('false_positive_rate\n{}'.format(false_positive_rate))
            # print('true_positive_rate\n{}'.format(true_positive_rate))
            # print('{}_roc_auc\n{}'.format(i, roc_auc[i]))
            plt.figure()
            plt.title('ROC')
            plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc[i])
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.ylabel('TPR')
            plt.xlabel('FPR')
            f = plt.gcf()  # 获取当前图像
            figure_path = os.path.join(res_path_roc, '{}_{}_roc_test.png'.format(i, roc_auc[i]))
            f.savefig(figure_path)
            f.clear()  # 释放内存

    def plot_roc_sklearn_con(self, labels, predict_prob, res_path, title='', name_list=['']):
        """

        :param labels:
        :param predict_prob:
        :param res_path:
        :param title: 类别标签
        :param name_list: 文件名
        :return:
        """
        pos_label_list = list(range(labels.shape[1]))  # 默认第一个类别
        from sklearn.metrics import roc_curve, auc
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        res_path_roc = os.path.join(res_path, 'roc')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        roc_auc = dict()
        plt.figure(figsize=(8, 8))

        color = ['k', 'm', 'b', 'yellow', 'g', 'c']
        ax = plt.gca()
        for i in pos_label_list:
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predict_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            false_positive_rate = fpr[i]
            true_positive_rate = tpr[i]
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob, pos_label=pos_label)
            # print('false_positive_rate\n{}'.format(false_positive_rate))
            # print('true_positive_rate\n{}'.format(true_positive_rate))
            # print('{}_roc_auc\n{}'.format(i, roc_auc[i]))
            plt.plot(false_positive_rate, true_positive_rate, color[i],
                     label='AUC:{:.3f}, {}'.format(roc_auc[i], name_list[i]))
            plt.ylabel('TPR')
            plt.xlabel('FPR')

        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))
        plt.title(title)
        plt.plot([0, 1], [0, 1], 'r--', label='base_line')
        plt.grid()
        f = plt.gcf()  # 获取当前图像
        figure_path = os.path.join(res_path_roc, '{}_roc_test_{}.png'.format('all', title))
        # print('figure_path\n{}'.format(figure_path))
        plt.legend(loc='lower right')
        f.savefig(figure_path)
        plt.show()

    def plot_pr_sklearn_all_in(self, labels, predict_prob, res_path, pos_label_list=None, title='', label_list=LABEL_LIST):
        if pos_label_list is None:
            pos_label_list = [0]  # 默认第一个类别

        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        res_path_roc = os.path.join(res_path, 'pr')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        pr_ap = dict()
        plt.figure(figsize=(8, 8))

        color = ['k', 'm', 'b', 'yellow', 'g', 'c', 'r', 'o']
        # ax = plt.gca()
        best_thd_list = []
        pr_value_dir = {}
        for i in pos_label_list:
            print(i)
            fpr[i], tpr[i], thresholds_pr = precision_recall_curve(labels[:, i], predict_prob[:, i])
            pr_ap[i] = average_precision_score(labels[:, i], predict_prob[:, i])
            precision = fpr[i]
            recall = tpr[i]
            temp_data_num = sum(labels[:, i])
            # # # 选择最优阈值 ####################################################################################
            p_r = [[precision[idx], recall[idx], thresholds_pr[idx]] for idx in range(thresholds_pr.shape[0])]
            p_r = np.array(p_r)
            p_r_fa = 2 / ((1 / p_r[:, 0]) + (1 / p_r[:, 1])+1e10)
            p_r_fa = np.reshape(p_r_fa, (p_r_fa.shape[0], 1))
            p_r_thd_fa = np.concatenate((p_r, p_r_fa), axis=-1)
            best_thd_index = np.argmax(p_r_thd_fa[:, -1])
            best_thd_list.append(thresholds_pr[best_thd_index])
            pr_value_dir[label_list[i]] = pr_ap[i]
            ###################################################
            plt.plot(recall, precision, color[i%len(color)],
                     label='AP:{:.3f}, {}-{}'.format(pr_ap[i], label_list[i], temp_data_num))
            # plt.fill_between(recall, precision, step='post', alpha=0.2, color=color[index])
            plt.ylabel('Precision')
            plt.xlabel('Recall')

        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))
        plt.title(title)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--', label='base_line')
        plt.grid()
        f = plt.gcf()  # 获取当前图像
        figure_path = os.path.join(res_path_roc, '{}_pr_test_{}.png'.format('all', title))
        print('figure_path\n{}'.format(figure_path))
        f.savefig(figure_path)
        # plt.show()
        return pr_value_dir, best_thd_list

    def plot_pr_sklearn_all_in_single(self, labels, predict_prob, res_path, pos_label_list=None, title='', label_list=LABEL_LIST):
        if pos_label_list is None:
            pos_label_list = [0]  # 默认第一个类别

        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        res_path_roc = os.path.join(res_path, 'pr')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        pr_ap = dict()

        color = ['k', 'm', 'b', 'yellow', 'g', 'c', 'r', 'o']
        # ax = plt.gca()
        best_thd_list = []
        for i in pos_label_list:
            # print(i)
            # debug_name = 'bao4-wen1-yang2-qiu1-zhen3-bing4'
            # if label_list[i] != debug_name:
            #     continue
            plt.figure(figsize=(8, 8))
            fpr[i], tpr[i], thresholds_pr = precision_recall_curve(labels[:, i], predict_prob[:, i])
            # print('pr_labels[:, i] is \n{}'.format(labels[:, i]))
            # print('pr_predict_prob[:, i] is \n{}'.format(predict_prob[:, i]))
            pr_ap[i] = average_precision_score(labels[:, i], predict_prob[:, i])
            precision = fpr[i]
            recall = tpr[i]
            temp_data_num = sum(labels[:, i])
            # # # 选择最优阈值 ####################################################################################
            p_r = [[precision[idx], recall[idx], thresholds_pr[idx]] for idx in range(thresholds_pr.shape[0])]
            p_r = np.array(p_r)
            p_r_fa = 2 / ((1 / p_r[:, 0]) + (1 / p_r[:, 1])+1e10)
            p_r_fa = np.reshape(p_r_fa, (p_r_fa.shape[0], 1))
            p_r_thd_fa = np.concatenate((p_r, p_r_fa), axis=-1)
            best_thd_index = np.argmax(p_r_thd_fa[:, -1])
            best_thd_list.append(thresholds_pr[best_thd_index])
            ###################################################
            plt.plot(recall, precision, color[i%len(color)],
                     label='AP:{:.3f}, {}-{}'.format(pr_ap[i], label_list[i], temp_data_num))
            # plt.fill_between(recall, precision, step='post', alpha=0.2, color=color[index])
            plt.ylabel('Precision')
            plt.xlabel('Recall')

        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))
            plt.title(title)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--', label='base_line')
            plt.grid()
            f = plt.gcf()  # 获取当前图像
            figure_path = os.path.join(res_path_roc, '{}_pr_test_{}.png'.format('single', label_list[i]))
            print('figure_path\n{}'.format(figure_path))
            f.savefig(figure_path)
            # plt.show()
        return best_thd_list

    def plot_pr_sklearn_all_in_test(self, labels, predict_prob, res_path, pos_label_list=None, title=''):
        if pos_label_list is None:
            pos_label_list = [0]  # 默认第一个类别

        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        res_path_roc = os.path.join(res_path, 'pr')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        pr_ap = dict()
        plt.figure(figsize=(8, 8))

        color = ['k', 'm', 'b', 'yellow', 'g', 'c', 'r', 'o']
        # ax = plt.gca()
        for i in pos_label_list:

            fpr[i], tpr[i], thresholds_pr = precision_recall_curve(labels[:, i], predict_prob[:, i])
            # print(f'labels is {labels}')
            # print(';;;;;;')
            # print(len(labels[:, i]))
            # print(predict_prob[:, i])
            # exit()
            # pr_ap[i] = auc(fpr[i], tpr[i])
            pr_ap[i] = average_precision_score(labels[:, i], predict_prob[:, i])
            precision = fpr[i]
            recall = tpr[i]
            temp_data_num = sum(labels[:, i])
            # # # 选择最优阈值 ####################################################################################
            p_r = [[precision[i], recall[i], thresholds_pr[i]] for i in range(thresholds_pr.shape[0])]
            p_r = np.array(p_r)
            p_r_fa = 2 / ((1 / p_r[:, 0]) + (1 / p_r[:, 1]))
            p_r_fa = np.reshape(p_r_fa, (p_r_fa.shape[0], 1))
            p_r_thd_fa = np.concatenate((p_r, p_r_fa), axis=-1)
            best_thd = np.argmax(p_r_thd_fa[:, -1])
            # ################################################################################################
            plt.step(recall, precision, color=color[i%len(color)], alpha=1, where='post',
                     label='ap:{:.3f} p:{:.3f} r:{:.3f}, thd {:.3f} {}-{}'.format(
                         pr_ap[i],
                         precision[best_thd],
                         recall[best_thd],
                         thresholds_pr[best_thd],
                         LABEL_LIST[i],
                         temp_data_num))
            plt.scatter(recall[best_thd], precision[best_thd], c=color[i], marker='o')
            plt.plot(recall[best_thd], precision[best_thd], c=color[i], marker="o", markersize=10)
            # plt.fill_between(recall, precision, step='post', alpha=0.2, color=color[index])
            plt.ylabel('Precision')
            plt.xlabel('Recall')

        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))
        plt.title(title)
        plt.plot([0, 1], [0, 1], 'r--', label='base_line')
        plt.grid()
        # f = plt.gcf()  # 获取当前图像
        # figure_path = os.path.join(res_path_roc, '{}_roc_test_{}.png'.format('all', title))
        # print('figure_path\n{}'.format(figure_path))
        # plt.legend(loc='lower right')
        # f.savefig(figure_path)
        plt.show()

    def get_best_f1_measure(self, labels, predict_prob, res_path, pos_label_list=None, title='', label_list=LABEL_LIST):
        # 根据recall和precision计算得到f1-measure值
        # for idx,  in false_positive_rate
        # 参考：https://blog.csdn.net/smileyan9/article/details/118599928

        if pos_label_list is None:
            pos_label_list = [0]  # 默认第一个类别

        # from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import precision_recall_curve
        # Compute ROC curve and ROC area for each class
        prec = dict()
        recalls = dict()
        # res_path_roc = os.path.join(res_path, 'roc')
        res_path_roc = os.path.join(res_path, 'f1-score')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        # roc_auc = dict()
        plt.figure(figsize=(8, 8))

        color = ['k', 'm', 'b', 'yellow', 'g', 'c', 'r', 'o']
        # ax = plt.gca()
        best_f1_sorce_list = {}
        best_threshold = {}
        for i in pos_label_list:
            prec[i], recalls[i], thresholds = precision_recall_curve(labels[:, i], predict_prob[:, i])
            # print(f'labels is {labels}')
            # print(';;;;;;')
            # print(len(labels[:, i]))
            # print(predict_prob[:, i])
            # exit()
            # roc_auc[i] = auc(prec[i], recalls[i])
            precisions = prec[i]
            recall_value = recalls[i]
            f1_score_value = (2 * precisions * recall_value) / (precisions + recall_value)
            best_f1_sorce = np.max(f1_score_value[np.isfinite(f1_score_value)])
            best_f1_sorce_list[label_list[i]] = best_f1_sorce
            best_f1_sorce_index = np.argmax(f1_score_value[np.isfinite(f1_score_value)])
            threshold_temp = thresholds[best_f1_sorce_index]
            best_threshold[label_list[i]] = threshold_temp

            # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob, pos_label=pos_label)
            # print('false_positive_rate\n{}'.format(false_positive_rate))
            # print('true_positive_rate\n{}'.format(true_positive_rate))
            # print('{}_roc_auc\n{}'.format(i, roc_auc[i]))
            temp_data_num = sum(labels[:, i])
            print(f'i is {i}')
            plt.plot(precisions, recall_value, color[i%len(color)],
                     label='f1-measure-best:{:.3f}, {}-{}'.format(best_f1_sorce_list[label_list[i]], label_list[i], temp_data_num))
            plt.ylabel('recall_value')
            plt.xlabel('precisions')

        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))
        plt.title(title)
        plt.plot([0, 1], [0, 1], 'r--', label='base_line')
        plt.grid()
        f = plt.gcf()  # 获取当前图像
        figure_path = os.path.join(res_path_roc, '{}_f1_score_test_{}.png'.format('all', title))
        print('figure_path\n{}'.format(figure_path))
        plt.legend(loc='lower right')
        f.savefig(figure_path)
        return best_f1_sorce_list, best_threshold
        # plt.show()


    def plot_roc_sklearn_all_in(self, labels, predict_prob, res_path, pos_label_list=None, title='', label_list=LABEL_LIST):


        if pos_label_list is None:
            pos_label_list = [0]  # 默认第一个类别

        from sklearn.metrics import roc_curve, auc
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        res_path_roc = os.path.join(res_path, 'roc')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        roc_auc = dict()
        plt.figure(figsize=(8, 8))

        color = ['k', 'm', 'b', 'yellow', 'g', 'c', 'r', 'o']
        # ax = plt.gca()
        # best_f1_sorce_list = {}
        auc_dir = {}
        for i in pos_label_list:
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predict_prob[:, i])
            # print(f'labels is {labels}')
            # print(';;;;;;')
            # print(len(labels[:, i]))
            # print(predict_prob[:, i])
            # exit()
            roc_auc[i] = auc(fpr[i], tpr[i])
            false_positive_rate = fpr[i]
            true_positive_rate = tpr[i]
            auc_dir[label_list[i]] = roc_auc[i]
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob, pos_label=pos_label)
            # print('false_positive_rate\n{}'.format(false_positive_rate))
            # print('true_positive_rate\n{}'.format(true_positive_rate))
            # print('{}_roc_auc\n{}'.format(i, roc_auc[i]))
            temp_data_num = sum(labels[:, i])
            print(f'i is {i}')
            plt.plot(false_positive_rate, true_positive_rate, color[i%len(color)],
                     label='AUC:{:.3f}, {}-{}'.format(roc_auc[i], label_list[i], temp_data_num))
            plt.ylabel('TPR')
            plt.xlabel('FPR')

        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))
        plt.title(title)
        plt.plot([0, 1], [0, 1], 'r--', label='base_line')
        plt.grid()
        f = plt.gcf()  # 获取当前图像
        figure_path = os.path.join(res_path_roc, '{}_roc_test_{}.png'.format('all', title))
        print('figure_path\n{}'.format(figure_path))
        plt.legend(loc='lower right')
        f.savefig(figure_path)
        # plt.show()
        return auc_dir



    def plot_roc_sklearn_single_in(self, labels, predict_prob, res_path, pos_label_list=None, title='', label_list=LABEL_LIST):
        if pos_label_list is None:
            pos_label_list = [0]  # 默认第一个类别

        from sklearn.metrics import roc_curve, auc
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        res_path_roc = os.path.join(res_path, 'roc')
        if not os.path.exists(res_path_roc):
            os.mkdir(res_path_roc)

        # print('labels\n{}'.format(labels))
        # print('predict_prob\n{}'.format(predict_prob))
        roc_auc = dict()

        color = ['k', 'm', 'b', 'yellow', 'g', 'c', 'r', 'o']
        ax = plt.gca()
        for i in pos_label_list:
            # debug_name = 'bao4-wen1-yang2-qiu1-zhen3-bing4'
            # if label_list[i] != debug_name:
            #     continue

            plt.figure(figsize=(8, 8))
            # print('pr_labels[:, i] is \n{}'.format(labels[:, i]))
            # print('pr_predict_prob[:, i] is \n{}'.format(predict_prob[:, i]))
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predict_prob[:, i])

            # print(f'labels is {labels}')
            # print(';;;;;;')
            # print(len(labels[:, i]))
            # print(predict_prob[:, i])
            # exit()
            roc_auc[i] = auc(fpr[i], tpr[i])
            false_positive_rate = fpr[i]
            true_positive_rate = tpr[i]
            # false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob, pos_label=pos_label)
            # print('false_positive_rate\n{}'.format(false_positive_rate))
            # print('true_positive_rate\n{}'.format(true_positive_rate))
            # print('{}_roc_auc\n{}'.format(i, roc_auc[i]))
            temp_data_num = sum(labels[:, i])
            print(f'i is {i}')
            plt.plot(false_positive_rate, true_positive_rate, color[i%len(color)],
                     label='AUC:{:.3f}, {}-{}'.format(roc_auc[i], label_list[i], temp_data_num))
            plt.ylabel('TPR')
            plt.xlabel('FPR')

        # ax.xaxis.set_ticks_position('bottom')
        # ax.yaxis.set_ticks_position('left')  # 指定下边的边作为 x 轴   指定左边的边为 y 轴
        # ax.spines['bottom'].set_position(('data', 0))
        # ax.spines['left'].set_position(('data', 0))
            plt.title(title)
            plt.plot([0, 1], [0, 1], 'r--', label='base_line')
            plt.grid()
            f = plt.gcf()  # 获取当前图像
            figure_path = os.path.join(res_path_roc, '{}_roc_test_{}.png'.format('single', label_list[i]))
            print('figure_path\n{}'.format(figure_path))
            plt.legend(loc='lower right')
            f.savefig(figure_path)
        # plt.show()

    def plotROC_all(self, pred_list, classLabels_list, figure_path):
        """

        :description: 画ROC曲线
        :param predStrengths:
        :param classLabels:
        :return:
               路径定义： figure_path = os.path.join(res_path_roc, '{}_{}_roc.png'.format(i, roc_auc))
        """

        def draw_roc(ax, pred, class_labels, color, index):
            cur = (0.0, 0.0)
            numPosClass = np.sum(np.array(class_labels) == 1)
            yStep = 1.0 / numPosClass
            xStep = 1.0 / (len(class_labels) - numPosClass)
            sortedIndicies = np.argsort(-np.array(pred.flatten()))
            ySum = 0.0
            for index in sortedIndicies:
                if class_labels[index] == 1:
                    delY = yStep
                    delX = 0
                else:
                    delY = 0
                    delX = xStep
                    ySum += cur[1]
                ax.plot([cur[0], cur[0] + delX], [cur[1], cur[1] + delY], c=color)
                # ax.lengend('label id-{}'.format(index))
                cur = (cur[0] + delX, cur[1] + delY)
        fig = plt.figure()
        fig.clf()
        f = plt.gcf()  # 获取当前图像
        ax = plt.subplot(111)
        color = ['k', 'r', 'b', 'yellow', 'g', 'c']
        print('length \n{}'.format(len(pred_list)))
        print('length \n{}'.format(len(pred_list)))
        # for index, (pred,label) in enumerate(zip(pred_list, classLabels_list)):
        for index in range(len(color) - 4):
            draw_roc(ax, pred_list[:, index], classLabels_list[:, index], color[index], index)

        ax.plot([0, 1], [0, 1], 'b--')
        ax.axis([0, 1, 0, 1])
        ax.legend(LABEL_LIST[:2])
        plt.xlabel('False Positve Rate')
        plt.ylabel('True Postive Rate')
        plt.title('ROC')
        ax.axis([0, 1, 0, 1])
        f.savefig(figure_path)
        plt.show()

    def plotROC(self, pred, classLabels, figure_path=None):
        """

        :description: 画ROC曲线
        :param predStrengths:
        :param classLabels:
        :return:
               路径定义： figure_path = os.path.join(res_path_roc, '{}_{}_roc.png'.format(i, roc_auc))

        """
        cur = (0.0, 0.0)
        numPosClass = np.sum(np.array(classLabels) == 1)
        print(numPosClass)
        yStep = 1.0 / numPosClass
        xStep = 1.0 / (len(classLabels) - numPosClass)
        sortedIndicies = np.argsort(-np.array(pred.flatten()))
        fig = plt.figure()
        fig.clf()
        ySum = 0.0
        ax = plt.subplot(111)
        for index in sortedIndicies:
            if classLabels[index] == 1:
                delY = yStep
                delX = 0
            else:
                delY = 0
                delX = xStep
                ySum += cur[1]
            ax.plot([cur[0], cur[0] + delX], [cur[1], cur[1] + delY], c='b')
            cur = (cur[0] + delX, cur[1] + delY)
        ax.plot([0, 1], [0, 1], 'b--')
        ax.axis([0, 1, 0, 1])
        plt.xlabel('False Positve Rate')
        plt.ylabel('True Postive Rate')
        plt.title('ROC')
        ax.axis([0, 1, 0, 1])
        # f.savefig(figure_path)
        plt.show()
        # f.clear()  # 释放内存

    def concat_auc_pr(self, res_path):
        pr_path = os.path.join(res_path, "pr")
        auc_path = os.path.join(res_path, "roc")
        pr_auc_path = os.path.join(res_path, "pr_auc")
        if not os.path.exists(pr_auc_path):
            os.makedirs(pr_auc_path)
        list_iter = os.listdir(pr_path)
        for _iter in list_iter:
            auc = os.path.join(auc_path, _iter)
            pr = os.path.join(pr_path, _iter)

            pr_data = cv2.imdecode(np.fromfile(pr, dtype=np.uint8), -1)
            auc_data = cv2.imdecode(np.fromfile(auc, dtype=np.uint8), -1)
            pr_auc = cv2.hconcat([pr_data, auc_data])
            cv2.imencode(".png", pr_auc)[-1].tofile(os.path.join(pr_auc_path, _iter))


if __name__ == '__main__':
    res_path = os.path.join(os.getcwd(), 'data')
    figure_path = os.path.join(res_path, '{}_{}_roc.png'.format(0, 0))
    LABEL_LIST = [i.strip() for i in open("data/label.list")]  # 需要给与一个label list文件
    LABEL2ID = {key: i for i, key in enumerate(LABEL_LIST)}  # label到ID的转换
    class_num = len(LABEL_LIST)
    print('class_num\n{}'.format(class_num))

    roc_curve = RocCurve(res_save_path=res_path)
    roc_curve.LABEL_LIST = LABEL_LIST
    roc_curve.LABEL2ID = LABEL2ID

    file_list = [os.path.join(res_path, file) for file in
                 [
                     'prediction_1_3_2_16000.tsv',
                     'prediction_1_5_8B_16000.tsv',
                     'prediction_1_5_9B_16000.tsv',
                     'prediction_1_5_11_16000.tsv',
                     'prediction_1_5_13_16000.tsv',
                 ]]
    roc_curve.get_roc_figure_sk_compare(predict_file_path_list=file_list, class_num=class_num, dtype="roc")
    roc_curve.get_roc_figure_sk_compare(predict_file_path_list=file_list, class_num=class_num, dtype="pr")
    roc_curve.concat_auc_pr(res_path)





