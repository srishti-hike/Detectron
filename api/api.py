
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
import time
import uuid

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import utils.segms as segms

from flask import Flask, render_template, request
from werkzeug import secure_filename
from flask import send_from_directory

app = Flask(__name__)

INPUT_FILE_PATH = "/mnt/api_files/input/"
GS_BUCKET = "gs://microapps-175405.appspot.com/srishti/"
OUTPUT_FILE_EXTENSION = '_output.png'

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--class-label',
        dest='class_label',
        help='class label to extract',
        default='person',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

# Main code on startup
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
utils.logging.setup_logging(__name__)
args = parse_args()
logger = logging.getLogger(__name__)
merge_cfg_from_file(args.cfg)
cfg.TEST.WEIGHTS = args.weights
cfg.NUM_GPUS = 1
assert_and_infer_cfg()
#model = infer_engine.initialize_model_from_cfg()
#dummy_coco_dataset = dummy_datasets.get_coco_dataset()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload')
def upload_file():
    return render_template('uploader.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_done():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(INPUT_FILE_PATH + secure_filename(str(key) + '.jpg'))
        return 'file uploaded successfully with UUID : '+ GS_BUCKET + str(key) + OUTPUT_FILE_EXTENSION

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8000)