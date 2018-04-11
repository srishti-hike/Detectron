
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
from queue import Queue
import json
import yaml
import numpy as np
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
from utils.io import cache_url
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import utils.segms as segms
import utils.videoutils as vid_utils
from apiutils.config import config

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class Segmenter(object):
    """Extracts segments from Detectron and processes them"""

    def __init__(self):
        # Logging initialization
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        utils.logging.setup_logging(__name__)
        self.logger = logging.getLogger(__name__)

        # Loading conf
        self.conf = config.get_config()

        # Model initialization
        merge_cfg_from_file(self.conf['detectron']['cfg'])
        cfg.NUM_GPUS = 1
        weights = cache_url(self.conf['detectron']['weights'], cfg.DOWNLOAD_CACHE)
        assert_and_infer_cfg()
        self.model = infer_engine.initialize_model_from_cfg(weights)
        self.dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    # function overview
    # def get_all_segments(self, im):
    # def pick_out relevant segments():
    # def style_transfer_output():
    # def process_image(self):

    def resize_image(self, image, height=None, width=None, inter=cv2.INTER_AREA):
        """Resize input image based on a given height or width"""
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def write_to_local(self, filename, filevalue):
        """Write file to local disk"""
        cv2.imwrite(filename, filevalue)
        self.logger.info("written to local: " + filename)

    def write_to_gcs(self, local_filepath, gcs_filename):
        """Write file to GCS bucket"""
        cmd = "gsutil cp " + local_filepath + " " + self.conf['io']['gs_bucket'] + gcs_filename
        returned_value = os.system(cmd)
        self.logger.info("written to gcs: " + local_filepath + ", returned value: " + str(returned_value))
        return returned_value

    def extract_segments(self, im_path, im):
        """ Use Detectron to extract all the segments in the image,
        their corresponding classes,
        and their binary masks"""


        # im = cv2.imread(im_path)

        if (im.shape[0] > 650):
            im = self.image_resize(im, height=600)

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                self.model, im, None, timers=timers
            )
        self.logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            self.logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

        segmented_images, classes, scores, segmented_binary_masks = vis_utils.segmented_images_in_original_image_size(
            im,
            im_path,
            self.conf['io']['output_dir'],
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=self.dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
        return segmented_images, classes, scores, segmented_binary_masks


    def compute_mask(self, im_path, im):
        """Use segment list given by def extract_segments to filter desired segments
        and compute a combined image and binary mask using all the segments"""

        segmented_images, classes, scores, segmented_binary_masks = self.extract_segments(im_path, im)

        super_mask = None
        if len(segmented_images) > 0:
            for index, value in enumerate(segmented_images):
                if classes[index] in self.conf['detectron']['class_label']:
                    if super_mask is None:
                        super_mask = np.zeros(value[:2])
                    bin_mask = segmented_binary_masks[index]
                    super_mask = np.logical_or(super_mask, bin_mask)
        return super_mask


    def process_image(self, im_path):
        im = cv2.imread(im_path)

        out_name = os.path.join(
            self.conf['io']['output_dir'], '{}'.format(os.path.basename(im_path) + '.png')
        )
        k = out_name.rfind("/")
        output_filename = out_name[k + 1:]

        self.logger.info('Processing {} -> {}'.format(im_path, out_name))
        mask = self.compute_mask(im_path, im)
        img = np.zeros(im.shape)

        for x in xrange(im.shape[0]):
            for y in xrange(im.shape[1]):
                if mask[x, y] == 0:
                    img[x, y, :] = [255, 255, 255]
                else:
                    img[x, y, :] = im[x, y, :]

        self.write_to_local(out_name, img)
        self.write_to_gcs(out_name, output_filename)
        return output_filename









