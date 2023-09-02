import os
from re import L
import sys
import json
from pathlib import Path
from time import sleep

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from copy import deepcopy
import logging

from mapping import disease_set_LID2zh, output_disease_set
from .common.data import PredImages
from .common.model import ClassificationModel


class DiagnosisClassifyAlg():
    def __init__(self) -> None:
        self.inited = False

    def init(self, config) -> None:
        logging.info("DiagnosisClassifyAlg.init()")
        logging.info(f"{config=}")

        if torch.cuda.is_available():
            #self.device = torch.device(f"cuda:{config['gpu_id']}")
            self.device = torch.device(f"cuda")
        else:
            self.device = 'cpu'
            logging.warning("torch.cuda.is_available = False, use cpu mode")
        self.model_label_list = config['model_label_list']
        self.model_label_num = len(self.model_label_list)
        self.model_thres_dict = config['model_thres_dict']
        self.fin_thres_dict = config['fin_thres_dict']
        self.fin_label_list = config['fin_label_list']
        self.body_part_dict = config['body_part_dict']
        self.body_part_sub2up = {}
        for _up_part in self.body_part_dict:
            for _sub_part in self.body_part_dict[_up_part]:
                self.body_part_sub2up[_sub_part] = _up_part
        self.qa_question_names = config['qa_questions']

        self.model_path = config['model_path']
        logging.info(f"loadding {self.model_path}")
        self.cls_model = ClassificationModel.load_from_checkpoint(self.model_path,
                                                                  num_classes=self.model_label_num)
        logging.info(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")
        self.cls_model = self.cls_model.to(self.device)
        self.cls_model.eval()

        self.supported_post_rule = config['supported_post_rule']
        self.simple_top2_config = config['simple_top2_config']
        self.score_norm_rule = config['score_norm_rule']
        self.score_norm_poly_config = config['score_norm_poly_config']
        for skey in [True, False]:
            _pair = self.score_norm_poly_config[skey]
            assert len(_pair['from']) == len(_pair['to']), f"{skey=}, {len(_pair['from'])=} != {len(_pair['to'])=}"
            for _lb in ['from', 'to']:
                for item in _pair[_lb]:
                    assert isinstance(item, float) or isinstance(item, int) or (isinstance(item, str) and item == 'thr'), f"{item=}"
        self.model_version = config['version']

        self.inited = True

    def predict_skin_img_total(self, lesin_bbox, total_img: np.ndarray):
        assert self.inited, f"DiagnosisClassifyAlg not inited"
        assert isinstance(total_img, np.ndarray), f"input img is ({type(total_img)}) not ndarray"
        pred_dataset = PredImages([total_img], self.model_label_num)
        testloader = torch.utils.data.DataLoader(pred_dataset, batch_size=1,
                                                 shuffle=False, drop_last=False,
                                                 num_workers=1)
        preds_list = []
        for i, (images, labels) in enumerate(testloader):
            try:
                images = images.to(self.device)
                with torch.no_grad():
                    preds = self.cls_model(images).sigmoid()
                for pred in preds:
                    preds_list.append(pred.cpu().numpy().tolist())
            except Exception as excpt:
                logging.exception(sys.exc_info())
                error_word = f"predict_skin_img_total: predict fail, Exception={excpt}"
                logging.error(error_word)
                raise RuntimeError(error_word)

        assert len(preds_list) == 1
        pred_scores = preds_list[0]
        assert len(pred_scores) == self.model_label_num
        ret_result = { 
            self.model_label_list[i]: pred_scores[i]
            for i in range(self.model_label_num)
        }
        return ret_result


    def merge_img_results(self, img_scores):
        assert self.inited, f"DiagnosisClassifyAlg not inited"

        merge_score = {}
        for img_score in img_scores:
            for lb in img_score:
                if lb not in merge_score:
                    merge_score[lb] = img_score[lb]
                else:
                    merge_score[lb] = max(merge_score[lb], img_score[lb])

        cls_result = []
        for pred_name in self.model_label_list:
            score = merge_score[pred_name]
            f_thr = self.model_thres_dict[pred_name]
            f_label = bool(score >= f_thr)

            if pred_name == 'D-acne_rosacea_sd-base':
                rosacea_label = f_label
                seb_der_label = f_label
                cls_result.append({'disease_name': 'D-acne_rosacea-base', 'score': score, 'label': rosacea_label})
                cls_result.append({'disease_name': 'D-seborrheic_dermatitis-base', 'score': score, 'label': seb_der_label})
            else:
                fin_name = pred_name
                cls_result.append({'disease_name': fin_name, 'score': score, 'label': f_label})

        cls_result = sorted(cls_result, key=lambda _r: _r['score'], reverse=True)
        for i in range(len(cls_result)):
            cls_result[i]['rank_id'] = i
            if cls_result[i]['disease_name'] in disease_set_LID2zh:
                cls_result[i]['disease_name_cn'] = disease_set_LID2zh[cls_result[i]['disease_name'] ]
            else:
                cls_result[i]['disease_name_cn'] = "其他"
        return cls_result


    def process_score_norm(self, input_results):
        assert self.inited, f"DiagnosisClassifyAlg not inited"
        if self.score_norm_rule == 'poly':
            output_result = []
            for ires in input_results:
                assert ires['disease_name'] in self.fin_thres_dict, f"{ires['disease_name']}"
                poly_list = {"from": [], "to": []}
                for flag in ['from', 'to']:
                    res_label = ires['label']
                    for item in self.score_norm_poly_config[res_label][flag]:
                        if isinstance(item, str) and item == 'thr':
                            poly_list[flag].append(self.fin_thres_dict[ires['disease_name']])
                        else:
                            poly_list[flag].append(item)
                old_score = ires['score']
                new_score = None
                for i in range(len(poly_list['from']) - 1):
                    if poly_list['from'][i] <= old_score <= poly_list['from'][i + 1]:
                        ratio = (poly_list['to'][i+1] - poly_list['to'][i]) / (poly_list['from'][i+1] - poly_list['from'][i] + 1e-6)
                        new_score = ratio * (old_score - poly_list['from'][i]) + poly_list['to'][i]
#                        logging.debug(f"{poly_list=}, {ratio=}, {old_score=}, {new_score=}")
                        break
                if new_score is None:
                    error_word = f"error config, {self.score_norm_poly_config}, {ires=}"
                    logging.error(error_word)
                    raise ValueError(error_word)
                output_result.append(deepcopy(ires))
                output_result[-1]['score'] = round(new_score, 4)
            return output_result
        else:
            error_word = f"{self.score_norm_rule=}, not supported" 
            logging.error(error_word)
            raise ValueError(error_word)

    def process_result_post(self, input_results, post_rule, score_norm=True):
        assert self.inited, f"DiagnosisClassifyAlg not inited"
        if post_rule.lower() == 'simple_top2':
            if score_norm:
                norm_result = self.process_score_norm(input_results)
            else:
                norm_result = input_results
            pos_item = [res for res in norm_result if res['label']]
            if len(pos_item) <= 0:
                output_res = [{
                    'disease_name': 'D-other-base',
                    'score': 0.6,
                    'label': True
                }]
            elif len(pos_item) == 1:
                output_res = [deepcopy(pos_item[0])]
                other_highs = [res for res in norm_result 
                               if not res['label'] and res['score'] >= self.simple_top2_config['other_high_thres']]
                other_highs = sorted(other_highs, key=lambda _oh: _oh['score'], reverse=True)
                if len(other_highs) > 0:
                    output_res.append(deepcopy(other_highs[0]))
            elif len(pos_item) >= 2:
                pos_res = sorted(pos_item, key=lambda _ir: _ir['score'], reverse=True)
                output_res = deepcopy(pos_res[:2])
        else:
            error_word = f'process_result_post:{post_rule=} not supported'
            logging.error(error_word)
            raise ValueError(error_word)

        for i in range(len(output_res)):
            output_res[i]['rank_id'] = i
            if output_res[i]['disease_name'] in disease_set_LID2zh:
                output_res[i]['disease_name_cn'] = disease_set_LID2zh[output_res[i]['disease_name'] ]
            else:
                output_res[i]['disease_name_cn'] = "其他"

        return output_res

    def run_diagnosis(self, total_imgs: list, lesion_bboxs: list):
        def _error_and_exit(excp, part):
            logging.exception(sys.exc_info)
            error_word = f"ERROR: DiagnosisClassifyAlg-run_diagnosis: {part}. exception_str={excp}"
            logging.error(error_word)
            return error_word, 500

        try:
            assert self.inited, f"DiagnosisClassifyAlg not inited"
            assert len(total_imgs) == len(lesion_bboxs), f"({len(total_imgs)=} != ({len(lesion_bboxs)=}))"
        except Exception as excp:
            return _error_and_exit(excp, 'pre_assert')

        try:
            cls_result_list = []
            for i in range(len(total_imgs)):
                cls_result_list.append(self.predict_skin_img_total(lesion_bboxs[i], total_imgs[i]))
        except Exception as excp:
            return _error_and_exit(excp, 'cls_model_predict')

        try:
            cls_result = self.merge_img_results(cls_result_list)
            norm_result = self.process_score_norm(cls_result)

            fin_result = {
                "disease_cls": norm_result
            }
        except Exception as excp:
            return _error_and_exit(excp, 'merge_result')
        
        return fin_result, 200

diagnosis_cls_alg = DiagnosisClassifyAlg()
