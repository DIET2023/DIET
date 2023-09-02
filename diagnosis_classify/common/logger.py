"""
Created by yizhi.chen.
"""

import logging
import os.path
import time


logger = logging.getLogger("skin_seg")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s")
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
