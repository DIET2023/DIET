import logging
import os
import time
import urllib
import sys
import cv2
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import math
from .src.utils import (
    init_distributed_mode,
)
from .src import resnet50 as resnet_models
from mapping import output_disease_set
from copy import deepcopy
from .score_norm import score_norm_func_dict, BaseNormFunc

class SWAVPredDataset(Dataset):
    def __init__(self, image_list, bbox_list, num_class, transform, use_bbox=False):
        assert isinstance(image_list, list), f"{type(image_list)=} is not list"
        assert isinstance(bbox_list, list), f"{type(bbox_list)=} is not list"
        self.org_image_list = image_list
        self.num_class = num_class
        self.transform = transform
        self.use_bbox = use_bbox
        self.image_list = []
        for img in self.org_image_list:
            self.image_list.append(img)
        if self.use_bbox:
            self.bbox_list = bbox_list
            assert len(self.org_image_list) == len(self.bbox_list), f"{len(self.org_image_list)=} != {len(self.bbox_list)}"
            for idx in range(len(self.bbox_list)):
                self.image_list.append(SWAVPredDataset.crop_img(org_img=self.org_image_list[idx],
                                                                crop_bbox=self.bbox_list[idx]))
                self.image_list.append(SWAVPredDataset.crop_center_img(org_img=self.org_image_list[idx]))
                logging.debug(f"{bbox_list=}")
#                save_path = f"/tmp/{bbox_list=}.jpg".replace('\'','').replace(' ','')
#                logging.debug(f"{save_path=}")
#                cv2.imwrite(save_path, self.image_list[-1])
    @staticmethod
    def crop_center_img(org_img):
        org_img_shape = org_img.shape
        xcent = org_img_shape[1] * 0.5
        ycent = org_img_shape[0] * 0.5
        new_w = org_img_shape[1] * 0.5
        new_h = org_img_shape[0] * 0.5
        new_w = max(new_w, 512)
        new_h = max(new_h, 512)
        def _xy_norm(xy, xyL, xyR):
            return int(min(max(xyL, xy), xyR))
        def _x_norm(x):
            return _xy_norm(x, 0, org_img_shape[1] - 1)
        def _y_norm(y):
            return _xy_norm(y, 0, org_img_shape[0] - 1)
        n_xmin = _x_norm(xcent - new_w * 0.5)
        n_xmax = _x_norm(xcent + new_w * 0.5)
        n_ymin = _y_norm(ycent - new_h * 0.5)
        n_ymax = _y_norm(ycent + new_h * 0.5)
        logging.debug(f"{n_xmin=},{n_xmax=},{n_ymin=},{n_ymax =}")
        crop_img = deepcopy(org_img[n_ymin: n_ymax + 1, n_xmin: n_xmax + 1, :])
        logging.debug(f"{type(crop_img)=}")
        logging.debug(f"{crop_img.shape=}")
        return crop_img


    @staticmethod
    def crop_img(org_img, crop_bbox):
        for _key in ['xmin', 'ymin', 'xmax', 'ymax']:
            assert _key in crop_bbox, f"{_key=} not in crop_bbox"
        org_img_shape = org_img.shape
        xmin = crop_bbox['xmin']
        xmax = crop_bbox['xmax']
        ymin = crop_bbox['ymin']
        ymax = crop_bbox['ymax']
        assert xmin < xmax, f"{xmin=} >= {xmax=}"
        assert ymin < ymax, f"{ymin=} >= {ymax=}"
        logging.debug(f"{crop_bbox=}")
        logging.debug(f"{org_img_shape=}")
        xcent = (xmin + xmax) * 0.5
        ycent = (ymin + ymax) * 0.5
        w = xmax - xmin
        h = ymax - ymin
        new_w = w * 1.5
        new_h = h * 1.5
        new_w = max(new_w, 512)
        new_h = max(new_h, 512)
        def _xy_norm(xy, xyL, xyR):
            return int(min(max(xyL, xy), xyR))
        def _x_norm(x):
            return _xy_norm(x, 0, org_img_shape[1] - 1)
        def _y_norm(y):
            return _xy_norm(y, 0, org_img_shape[0] - 1)
        n_xmin = _x_norm(xcent - new_w * 0.5)
        n_xmax = _x_norm(xcent + new_w * 0.5)
        n_ymin = _y_norm(ycent - new_h * 0.5)
        n_ymax = _y_norm(ycent + new_h * 0.5)
        logging.debug(f"{n_xmin=},{n_xmax=},{n_ymin=},{n_ymax =}")
        crop_img = deepcopy(org_img[n_ymin: n_ymax + 1, n_xmin: n_xmax + 1, :])
        logging.debug(f"{type(crop_img)=}")
        logging.debug(f"{crop_img.shape=}")
        return crop_img

    def __getitem__(self, index):
        try:
            img = self.image_list[index]
            assert img.dtype == np.uint8, f"{img.dtype=}"
            assert img.ndim == 3, f"{str(img.shape)}="
            assert img.shape[2] == 3, f"{img.shape[2]=} not 3"

            #if self.use_bbox:
            #    bbox = self.bbox_list[index] # not use yet
            #    img = SWAVPredDataset.crop_img(img, bbox)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            #float_image = rgb_img.transpose((2, 0, 1)).astype(np.float32) / 255
            #torch_img = torch.from_numpy(float_image).contiguous()
            #torch_img = torch.from_numpy(rgb_img.transpose((2, 0, 1))).contiguous()
            ret_img = self.transform(pil_img)

            label = torch.zeros(self.num_class)
            return ret_img, label

        except Exception as excp:
            logging.exception(sys.exc_info())
            logging.error(f"Error!, loading {index=} img, {excp=}")
            raise RuntimeError

    def __len__(self):
        return len(self.image_list)

class SwavDiseaseClsAlg():
    def __init__(self) -> None:
        self.inited = False

    def init(self, args):
        # init args
        logging.info("SwavDiseaseClsAlg.init()")
        logging.info(f"{args=}")

        self.model_dump_path = args['model_dump_path']
        self.model_arch = args['model_arch']
        self.model_layer_num = args['model_layer_num']
        self.gpu_id = args['gpu_id']
        
        self.input_size = args['input_size']
        self.use_bbox = args['use_bbox']
        self.num_class = args['num_class']
        norm_mean = args['norm']['mean']
        norm_std = args['norm']['std']

        self.multi_input_merge_func = args['multi_input_merge_func']
        self.disease_name_map_path = args['disease_name_map']
        self.output_display_thr = args['output_display_thr']

        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
        def _load_model():
            model = resnet_models.__dict__[self.model_arch](output_dim=self.model_layer_num)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

            #state_dict = torch.load(self.model_dump_path, map_location="cuda:" + str(self.gpu_id))
            state_dict = torch.load(self.model_dump_path)
            #state_dict = torch.load(self.model_dump_path)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            # remove prefixe "module."
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            for k, v in model.state_dict().items():
                if k not in list(state_dict):
                    logging.info('key "{}" could not be found in provided state dict'.format(k))
                elif state_dict[k].shape != v.shape:
                    logging.info('key "{}" is of different shape in model and provided state dict'.format(k))
                    state_dict[k] = v
            msg = model.load_state_dict(state_dict, strict=False)
            logging.info("Load pretrained model with msg: {}".format(msg))
            model = model.cuda()
            #model = nn.parallel.DistributedDataParallel(
            #    model,
            #    device_ids=[self.gpu_id],
            #    find_unused_parameters=True,
            #)
            model.eval()
            return model
        self.model = _load_model()

        self.label_names = []
        self.label_trans_map = {}
        self.label_filter_list = []
        self._load_name_map_json(self.disease_name_map_path)
        assert self.num_class == len(self.label_names),\
            f"{self.num_class=}!={len(self.label_names)}"

        self.score_norm_name = args['score_norm_func']
        assert self.score_norm_name in args, f"{self.score_norm_name=} not in args"
        self.score_norm_func = score_norm_func_dict[self.score_norm_name](
            args=args[self.score_norm_name],
            upper_alg=self
        )
        assert isinstance(self.score_norm_func, BaseNormFunc), \
            f"{type(self.score_norm_func)=} is not norm func"

        self.model_version = args['version']
        self.inited = True
        logging.info("SwavDiseaseClsAlg.init()...done")

    def _load_name_map_json(self, to_load_json_path):
        assert os.path.isfile(to_load_json_path), \
            f"{to_load_json_path} is not a file"
        with open(to_load_json_path, 'r') as fp:
            jd = json.load(fp)
        self.label_names = jd['label_names']
        self.label_trans_map = jd['label_trans_map']
        self.label_filter_list = jd['label_filter_list']
        self.label_LID2zh = {}
        for item in self.label_names + self.label_filter_list:
            LID = item['LID']
            zh_name = item['zh']
            if LID not in self.label_LID2zh:
                self.label_LID2zh[LID] = zh_name
            elif self.label_LID2zh[LID] != zh_name:
                logging.error(f"{LID=}, {self.label_LID2zh[LID]=}, but {zh_name=}")
                raise ValueError
        self.label_zh2LID = {self.label_LID2zh[_key]: _key for _key in self.label_LID2zh}

    def run_nn_network(self, image_list, bbox_list):
        assert self.inited, f"SwavDiseaseClsAlg.run_nn_network not inited"
        pred_dataset = SWAVPredDataset(image_list=image_list,
                                       bbox_list=bbox_list,
                                       num_class=self.num_class,
                                       transform=self.transform,
                                       use_bbox=self.use_bbox)
        pred_data_loader = torch.utils.data.DataLoader(
            pred_dataset,
            batch_size=1, num_workers=1, pin_memory=True
        )

        predict_scores_list = []
        with torch.no_grad():
            for i, (input_tensor, target) in enumerate(pred_data_loader):
                input_tensor = input_tensor.cuda(non_blocking=True)
                # label_list.append(target.cuda().data.cpu().numpy())
                logging.debug(f"start running img #{i}")
                output = self.model(input_tensor)
                logging.debug(f"end running img #{i}")
                output: torch.Tensor
                # output.cuda().data.cpu().numpy()
                for pred in output:
                    predict_scores = pred.cuda().data.cpu().numpy().tolist()
                    predict_scores_list.append(predict_scores)

        return predict_scores_list

    def trans_result_nn_to_id(self, merge_predict_score):
        assert self.inited, f"SwavDiseaseClsAlg.trans_result_nn_to_id not inited"
        assert len(merge_predict_score) == self.num_class, f"{len(merge_predict_score)=}!={self.num_class=}"

        # label score link
        label_score_dict = { 
            self.label_names[i]['LID']: merge_predict_score[i]
            for i in range(self.num_class)
        }

        # score norm
        norm_label_score_dict = {}
        norm_label_score_dict = self.score_norm_func(label_score_dict)

        # label div
        dived_label_score_dict = {}
        for lb in norm_label_score_dict:
            if lb not in self.label_trans_map:
                dived_label_score_dict[lb] = norm_label_score_dict[lb]
            else:
                for trans2lb in self.label_trans_map[lb]:
                    dived_label_score_dict[trans2lb] = norm_label_score_dict[lb]

        #filter label
        filter_label_score_dict = {}
        for item in self.label_filter_list:
            lib = item['LID']
            filter_label_score_dict[lib] = dived_label_score_dict.setdefault(lib, 0.0)

        # global filter
        for _key in list(filter_label_score_dict.keys()):
            if _key not in output_disease_set:
                del filter_label_score_dict[_key]
        for _key in output_disease_set:
            if _key not in filter_label_score_dict:
                filter_label_score_dict[_key] = 0.0

        return filter_label_score_dict

    def _output_proc(self, filter_label_score_dict):
        label_names = list(filter_label_score_dict.keys())
        name_rank = sorted(label_names, key=lambda lb: filter_label_score_dict[lb], reverse=True)
        output_items = []
        for idx in range(len(name_rank)):
            _name = name_rank[idx]
            _score = filter_label_score_dict[_name]
            output_items.append({
                'rank_id': idx,
                'disease_name': _name,
                'score': _score,
                'label': bool(_score >= self.output_display_thr)
            })
        return output_items

    def run_classify(self, image_list, bbox_list):
        def _error_and_exit(excp, part):
            logging.exception(sys.exc_info)
            error_word = f"ERROR: DiagnosisClassifyAlg-run_diagnosis: {part}. exception_str={excp}"
            logging.error(error_word)
            return error_word, 500

        try:
            assert self.inited, f"SwavDiseaseClsAlg.run_classify not inited"
            assert len(image_list) == len(bbox_list), f"({len(image_list)=} != ({len(bbox_list)=}))"
        except Exception as excp:
            return _error_and_exit(excp, "pre_assert")

        try:
            predict_scores_list = self.run_nn_network(image_list, bbox_list)
        except Exception as excp:
            return _error_and_exit(excp, "run_nn_network")

        try:
            merge_predict_score = []
            if self.multi_input_merge_func == 'mean':
                merge_predict_score = np.asarray(predict_scores_list).mean(axis=0).tolist()
            elif self.multi_input_merge_func == 'max':
                merge_predict_score = np.asarray(predict_scores_list).max(axis=0).tolist()
            elif self.multi_input_merge_func == 'min':
                merge_predict_score = np.asarray(predict_scores_list).min(axis=0).tolist()
            else:
                logging.error(f"{self.multi_input_merge_func=} not support")
                raise ValueError
        except Exception as excp:
            return _error_and_exit(excp, "merge_predict_score")

        try:
            filter_label_score_dict = self.trans_result_nn_to_id(merge_predict_score)
            fin_result = {
                "disease_cls": self._output_proc(filter_label_score_dict)
            }
        except Exception as excp:
            return _error_and_exit(excp, 'post_procces_result')
        
        return fin_result, 200        

swav_disease_cls_alg = SwavDiseaseClsAlg()
