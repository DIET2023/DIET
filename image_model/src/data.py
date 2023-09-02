#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
from tqdm import tqdm
import random
# import torchvision.datasets.ImageFolder as datasets
from torchvision.datasets import ImageFolder

def write_tsv(tsv, path):
    with open(path, "w") as f:
        for line in tsv:
            f.write("\t".join(json.dumps(seg, ensure_ascii=False) for seg in line) + "\n")

def get_tsv(label_list, pred_list, image_list, new_out_features_list=None, old_out_features_list=None):
    res = []
    len_preds_list = len(pred_list)

    for i in range(len_preds_list):
        if new_out_features_list != None and old_out_features_list != None:

            res.append((image_list[i], pred_list[i], label_list[i]), new_out_features_list[i], old_out_features_list[i])
        else:
            res.append((image_list[i], pred_list[i], label_list[i]))


    # logging.debug(f"[{dataset_name}] predict result save at {outputdir}")
    return res


class val_data_loader(ImageFolder):
    def __init__(self, root):
        # datasets.__init()
        # super(datasets, self).__init__(name, age)
        super(val_data_loader, self).__init__(root)
        self.image_name_list = []

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        self.image_name_list.append(path)
        # self.image_name_list.append()
        return sample, target

    def clear_image_list(self):
        self.image_name_list = []

if __name__ == '__main__':
    pass
