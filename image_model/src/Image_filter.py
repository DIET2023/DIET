#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import ImageFile
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import traceback
import logging
import sys

def error_file_list(image_list: list):
    """

    :param file_list: 软链接存放路径
    :return:
    """
    # file_list = [train_dataset, valid_dataset, test_dataset]
    # print(image_list)
    cnt_erro_list = []
    error_image_list = []
    #bar = tqdm(total=len(image_list))
    cnt = 0
    for fn in image_list:
        #bar.update(1)
        fn_real_path = fn
        #fn_real_path = os.path.join(self.root_image_path, fn)

        try:
            #print(fn_real_path)
            im = ImageFile.Image.open(fn_real_path)
            im2 = im.convert('RGB')
        except OSError as e:
            #print(e)
            cnt = cnt + 1
            #print("Cannot load : {}".format(fn_real_path))
            error_image_list.append(fn_real_path)
            # os.system('rm -rf {}'.format(fn_real_path))  # 删除图片
        # exit()
    cnt_erro_list.append(cnt)
    # print('total cnt_error num is \n{}'.format(sum(cnt_erro_list)))
    # print('total cnt_erro_image_list is \n{}'.format(error_image_list))

    return error_image_list

def __error_file_list(sub_args):
    try:
        image_list = sub_args
        return error_file_list(image_list)
    except Exception as e:
        #logging.error("Error: {}".format(sub_args))
        exc_type, exc_value, exc_obj = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_obj, file=sys.stdout)


def multi_process_file(file_path, error_file_name='error_list.list'):
    image_list = [os.path.join(file_path, name) for name in os.listdir(file_path)]
    data_image_num = 5000
    num_len = len(image_list)
    list_index = list(range(0, num_len, data_image_num))

#        cpu_cnt = os.cpu_count()
#        pool = mp.Pool(processes=cpu_cnt-4)
#        error_list = []
#        for i in list_index:
#            start = int(i)
#            if i + data_image_num > num_len:
#                end = num_len
#            else:
#                end = i + data_image_num
#            error_image_list = pool.apply_async(self.error_file_list, (image_list[start:end], ), callback=self.my_callback)
#            error_list.append(error_image_list)
#        print(error_list)

    process_num = cpu_count() - 4
    logging.debug('process_num = {}'.format(process_num))
    args_batch = []
    for i in range(0, len(image_list), data_image_num):
        args_batch += [image_list[i: i+data_image_num]]

    read_tqdm = tqdm(args_batch, total=len(args_batch), desc='mark text - start', position=0)
    error_list = []
    with Pool(process_num) as pool:
        ress = list(tqdm(pool.imap(__error_file_list, read_tqdm),
                        total=len(args_batch),
                        desc='mark text - done', position=1))
    for res in ress:
        error_list += res
    with open(error_file_name, 'w') as fp:
        for row in error_list:
            fp.write(row + '\n')
    
    return error_file_name

def get_erro_soft_list(error_list_path):
    soft_link_list = []
    with open(error_list_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            line: str
            # print('line')
            # print(line)
            soft_link = line.split('/')[-1]
            soft_link_list.append(soft_link.replace('\n', ''))
            # print(soft_link)
            # exit()
    return soft_link_list

def del_soft_link(error_list_path, ori_soft_link_path):
    error_soft_link_list = get_erro_soft_list(error_list_path=error_list_path)
    image_list = os.listdir(ori_soft_link_path)
    bar = tqdm(total=len(image_list))
    # print(error_soft_link_list[:5])
    # print(image_list[:5])
    # exit()
    for temp_soft_link in image_list:
        bar.update(1)
        if temp_soft_link in error_soft_link_list:
            fn_real_path = os.path.join(ori_soft_link_path, temp_soft_link)
            os.system('rm -rf {}'.format(fn_real_path))  # 删除图片
            print('error link is {}'.format(fn_real_path))

if __name__ == '__main__':
    file_list = ['']  # 图片文件保存路径

    error_file_name = multi_process_file(file_path=file_list[0])
    del_soft_link(error_list_path=error_file_name, ori_soft_link_path=file_list[0])  # 删除出问题的图像
    # image_filter.error_file_list(image_list)


