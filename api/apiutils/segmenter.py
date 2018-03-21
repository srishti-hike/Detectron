
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
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

MILLISECONDS_IN_SECOND= 1000.0
DIRECTORY_TO_WATCH = "/mnt/api_files/input/"
DIRECTORY_TO_WRITE = "/mnt/api_files/output/"
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

def segment(im_list, filename):
    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
        segmented_images, classes, scores = vis_utils.segmented_images(
            im,
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dummy_coco_dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
        found = False
        if len(segmented_images) > 0:
            for index, value in enumerate(segmented_images):
                if classes[index] == args.class_label and not found:
                    logger.info('Writing output file to: {}'.format(str(i)))
                    # cv2.imwrite(args.output_dir + '/' + str(i) + '_' + args.class_label + '_' + `index` + ".png", value)
                    cv2.imwrite(DIRECTORY_TO_WRITE +filename.rstrip(".jpg") + OUTPUT_FILE_EXTENSION, value)
                    cmd = "gsutil cp " + DIRECTORY_TO_WRITE + filename.rstrip(".jpg") + OUTPUT_FILE_EXTENSION + " " + GS_BUCKET
                    returned_value = os.system(cmd)
                    logger.info("written file to gcs: "+ str(returned_value))
                    found = True

        return found

class Watcher:


    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(50.0/ MILLISECONDS_IN_SECOND)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)
            im_list = [event.src_path]
            k = event.src_path.rfind("/")
            found = segment(im_list, event.src_path[k+1:])
            logger.info("segmentation done and saved: " + str(found))


# Global variables on start

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
utils.logging.setup_logging(__name__)
args = parse_args()

logger = logging.getLogger(__name__)
merge_cfg_from_file(args.cfg)
cfg.TEST.WEIGHTS = args.weights
cfg.NUM_GPUS = 1
assert_and_infer_cfg()
model = infer_engine.initialize_model_from_cfg()
dummy_coco_dataset = dummy_datasets.get_coco_dataset()

if __name__ == '__main__':
    logger.info("Done initialising model")
    w = Watcher()
    w.run()


