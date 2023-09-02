import logging
import os
import time
import sys
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
import pickle
from mapping import output_disease_set
from copy import deepcopy
import math

class BaseNormFunc():
    def __init__(self, args, upper_alg) -> None:
        logging.info("Init BaseNormFunc")
        self.args = args
        self.upper_alg = upper_alg
        if 'use_score_fit_qa' in args:
            self.use_score_fit_qa = args['use_score_fit_qa']
        else:
            self.use_score_fit_qa = False
        if self.use_score_fit_qa:
            self.max_to_score = args['max_to_score']
            self.min_to_score = args['min_to_score']
            self.score_cut_off_thr = args['score_cut_off_thr']

    def __call__(self, *args, **kwds):
        raise NotImplemented

    def score_fit_qa(self, output_score_dict):
        logging.debug("run score_fit_qa")
        fin_max_score = max([output_score_dict[lb] for lb in output_score_dict])
        if self.max_to_score is not None:
            max_from = fin_max_score
        else:
            max_from = 1.0
        new_k = (self.max_to_score - self.min_to_score) / (max_from - self.score_cut_off_thr)
        for lb in output_score_dict:
            if output_score_dict[lb] < self.score_cut_off_thr:
                output_score_dict[lb] = 0.0
            else:
                output_score_dict[lb] = (output_score_dict[lb] - self.score_cut_off_thr) * new_k + self.min_to_score
        return output_score_dict


class PlattScalingNormFunc(BaseNormFunc):
    def __init__(self, args, upper_alg) -> None:
        logging.info("Init PlattScalingNormFunc")
        super(PlattScalingNormFunc, self).__init__(args, upper_alg)

        self.num_class = upper_alg.num_class
        self.model_pickle_path = args['norm_lr_pickle']
        self.model_output_names_pair = upper_alg.label_names
        self.model_output_names = [item['LID'] for item in self.model_output_names_pair]
        with open(self.model_pickle_path, 'rb') as fp:
            self.lr_model = pickle.load(fp)
        assert isinstance(self.lr_model, LogisticRegression), f"{type(self.lr_model)=} is not LogisticRegression"
        assert self.lr_model.classes_.shape[0] == self.num_class ,\
            f"{self.lr_model.classes_.shape[0]=} != {self.num_class=}"
        assert self.lr_model.n_features_in_ == self.num_class ,\
            f"{self.lr_model.n_features_in_=} != {self.num_class=}"
        logging.debug(f"{self.use_score_fit_qa=}")

    def __call__(self, label_score_dict):
        score_list = []
        for lid in self.model_output_names:
            if lid not in label_score_dict:
                error_code = f"{lid=} in model_output_names, not in score_dict"
                logging.error(error_code)
                raise ValueError(error_code)
            score_list.append(label_score_dict[lid])
        score_vector = np.asarray([score_list])
        norm_score = self.lr_model.predict_proba(score_vector)
        output_label_dict = {}
        for i, lid in enumerate(self.model_output_names):
            output_label_dict[lid] = float(norm_score[0][i])
        for lid in label_score_dict:
            if lid not in output_label_dict:
                output_label_dict[lid] = 0.0
        
        if self.use_score_fit_qa:
            output_label_dict = self.score_fit_qa(output_label_dict)

        return output_label_dict

class OrgRankToScoreNormFunc(BaseNormFunc):
    def __init__(self, args, upper_alg) -> None:
        logging.info("Init OrgRankToScoreNormFunc")
        super(OrgRankToScoreNormFunc, self).__init__(args, upper_alg)

        self.remain_topk = args['remain_topk']
        self.rank2score = args['rank2score']
        self.num_class = upper_alg.num_class
        assert self.remain_topk <= self.num_class, f"{self.remain_topk=} > {self.num_class=}"
        assert isinstance(self.rank2score, list), f"{type(self.rank2score)=} != list"
        assert self.remain_topk <= len(self.rank2score), f"{self.remain_topk=} > {len(self.rank2score)=}"

    def __call__(self, label_score_dict):
        label_names = list(label_score_dict.keys())
        name_rank = sorted(label_names, key=lambda lb: label_score_dict[lb], reverse=True)

        output_label_dict = {}
        for idx in range(self.remain_topk):
            idx_name = name_rank[idx]
            output_label_dict[idx_name] = self.rank2score[idx]
        for lb in label_score_dict:
            if lb not in output_label_dict:
                output_label_dict[lb] = 0.0

        return output_label_dict

class InfBayesMatrixNormFunc(BaseNormFunc):
    def __init__(self, args, upper_alg) -> None:
        logging.info("Init InfBayesMatrixNormFunc")
        super(InfBayesMatrixNormFunc, self).__init__(args, upper_alg)

        self.matrix_json_path = args['matrix_json_path']
        self.lid2zh = upper_alg.label_zh2LID
        self.label_names = upper_alg.label_names
        with open(self.matrix_json_path,"r") as fp:
            self.matrix_json_zh = json.load(fp)
        self.matrix_json = {}
        for item_inf in self.label_names:
            assert item_inf['zh'] in self.matrix_json_zh, f"{item_inf['zh']=} not in matrix_json_zh"
            self.matrix_json[item_inf['LID']] = {}
            for item_to in self.label_names:
                assert item_to['zh'] in self.matrix_json_zh[item_inf['zh']], f"{item_inf['zh']=}{item_to['zh']=} not in matrix_json_zh"
                self.matrix_json[item_inf['LID']][item_to['LID']] = self.matrix_json_zh[item_inf['zh']][item_to['zh']]
        self.max_to_score = args['max_to_score']
        self.min_to_score = args['min_to_score']
        self.score_cut_off_thr = args['score_cut_off_thr']
        self.softmax_prate = args['softmax_prate']

    def hand_softmax(self, in_dict):
        def _exp(x):
            return math.exp(x * self.softmax_prate)
        _exp_num = {lb: _exp(in_dict[lb]) for lb in in_dict}
        _exp_sum = sum([_exp_num[lb] for lb in _exp_num])
        return {lb: _exp_num[lb] / _exp_sum for lb in _exp_num}

    def __call__(self, label_score_dict):
        label_names = list(label_score_dict.keys())

        softmax_score_dict = self.hand_softmax(label_score_dict)
        output_score_dict = {lb: 0 for lb in label_names}
        for inf_name in label_names:
            assert inf_name in self.matrix_json, f"{inf_name=} not in matrix_json"
            for to_name in label_names:
                assert to_name in self.matrix_json[inf_name], f"{inf_name}|{to_name} not in matrix_json"
                output_score_dict[to_name] += self.matrix_json[inf_name][to_name] * softmax_score_dict[inf_name]

        fin_max_score = max([output_score_dict[lb] for lb in output_score_dict])
        if self.max_to_score is not None:
            mul_rate = 1 / fin_max_score
        else:
            mul_rate = 1.0
        new_k = (self.max_to_score - self.min_to_score) / (1 - self.score_cut_off_thr)
        for lb in output_score_dict:
            output_score_dict[lb] *= mul_rate
            if output_score_dict[lb] < self.score_cut_off_thr:
                output_score_dict[lb] = 0.0
            else:
                output_score_dict[lb] = (output_score_dict[lb] - self.score_cut_off_thr) * new_k + self.min_to_score

        for lb in label_score_dict:
            if lb not in output_score_dict:
                output_score_dict[lb] = 0.0

        return output_score_dict

score_norm_func_dict = {
    "org_rank_to_score": OrgRankToScoreNormFunc,
    "inf_bayes_matrix": InfBayesMatrixNormFunc,
    "platt_scaling_norm": PlattScalingNormFunc
}