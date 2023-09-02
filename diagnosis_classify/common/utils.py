
import os
import shutil
import json

import numpy as np
import imageio
import cv2
from tqdm import tqdm
import pytorch_lightning as pl

from common.logger import logger


class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
        save_step_frequency: how often to save in steps
        use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
        default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        os.makedirs("model", exist_ok=True)

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch  """""
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"model_{global_step:08d}.ckpt"
            ckpt_path = os.path.join("model", filename)
            trainer.save_checkpoint(ckpt_path)


def draw_debug_image_from_dataloader(train_dataset, output_root, max_size=5):
    try:
        os.makedirs(output_root)
    except Exception as e:
        print("%s exists. Jump" % output_root)
        return
    choices = np.random.randint(len(train_dataset), size=max_size)
    for step, choice in enumerate(tqdm(choices)):
        raw_image, raw_label = train_dataset.get_raw_data(choice)
        ratio = 256 / raw_image.shape[0]
        raw_image = cv2.resize(raw_image, (int(raw_image.shape[1]*ratio), 256))

        image, label = train_dataset[choice]
        image = image.cpu().numpy()
        image = np.transpose(image, [1,2,0])
        label = label.cpu().numpy()
        label = np.transpose(label, [1,2,0])
        label = np.concatenate([label,label,label], axis=-1)
        paint = np.concatenate([raw_image/255, image, label, image*0.7 + label*0.3], axis=1)
        paint = (paint*255).astype(np.uint8)

        imageio.imwrite(os.path.join(output_root, "%d.png" % step), paint)

    logger.info("Drawed %d debug images under %s" % (max_size, output_root))


def write_jsonl(jsons, path):
    with open(path, "w") as f:
        for unit in jsons:
            f.write(json.dumps(unit, ensure_ascii=False)+"\n")


def read_jsonl(path):
    coll = []
    for line in open(path):
        coll.append(json.loads(line.strip()))
    return coll


def read_tsv(path):
    coll = []
    for line in open(path):
        line = line.strip()
        segments = []
        for seg in line.split("\t"):
            segments.append(json.loads(seg))
        coll.append(segments)
    return coll


def write_tsv(tsv, path):
    with open(path, "w") as f:
        for line in tsv:
            f.write("\t".join(json.dumps(seg, ensure_ascii=False) for seg in line) + "\n")


def _merge_color(_img, _masks):
    zero_img = np.zeros_like(_img[:, :, 0])
    _single_mask = []
    if not isinstance(_masks, list):
        _masks = [_masks]
    assert len(_masks) <= 3

    for _mask in _masks:
        if len(_mask.shape) == 2:
            _single_mask.append(_mask)
        else:
            _single_mask.append(_mask[:,:,0])
    _single_mask = [zero_img] * (3 - len(_single_mask)) + _single_mask
    color_mask = np.stack(_single_mask, axis=-1)
    viz_img = cv2.addWeighted(_img, 0.62, color_mask, 0.38, 1)
    return  viz_img


def get_maskMerged_image(org_img, org_pre_img, org_label_img):
    zero_img = np.zeros_like(org_img[:, :, 0])
    fa_label_img = np.clip(org_pre_img.astype(float) - org_label_img.astype(float), 0, 255).astype('uint8')
    miss_label_img = np.clip(org_label_img.astype(float) - org_pre_img.astype(float), 0, 255).astype('uint8')
    hit_label_img = np.clip(org_pre_img.astype(float) - fa_label_img.astype(float), 0, 255).astype('uint8')

    merge_vis_img = _merge_color(org_img, [miss_label_img, hit_label_img, fa_label_img])
    pre_vis_img = _merge_color(org_img, [zero_img, zero_img, org_pre_img])
    label_vis_img = _merge_color(org_img, [org_label_img, zero_img, zero_img])
    return merge_vis_img, pre_vis_img, label_vis_img
