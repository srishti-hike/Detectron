
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
import utils.videoutils as vid_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

MILLISECONDS_IN_SECOND= 1000.0
DIRECTORY_TO_WATCH = "/mnt/api_files/input/"
DIRECTORY_TEMP = "/mnt/api_files/tmp/"
DIRECTORY_TO_WRITE = "/mnt/api_files/output/"
GS_BUCKET = "gs://microapps-175405.appspot.com/srishti/"
OUTPUT_FILE_EXTENSION = '_output.png'
STICKER_SELFIE_HIT = "sticker"

VIDEO_BG_RESOURCES_DIRECTORY = "/mnt/video_bg_resources/"

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

def write_to_local(filename, filevalue):
    cv2.imwrite(filename, filevalue)
    logger.info("written to local: "+ filename)


def write_to_gcs(local_filepath, gcs_filename):
    cmd = "gsutil cp " + local_filepath + " " + GS_BUCKET + gcs_filename
    returned_value = os.system(cmd)
    logger.info("written to gcs: " + local_filepath + ", returned value: "+ str(returned_value))
    return returned_value



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
        segmented_images, classes, scores, segmented_binary_masks = vis_utils.segmented_images(
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
                    found = True
                    bin_mask = segmented_binary_masks[index]
                    return found,  value, bin_mask

        return found, "", "", ""

def style_transfer(input_file_path, input_file_name, mask):
    logger.info("in function style_transfer")
    tmp_file_path = input_file_path
    output_file_name = input_file_name.rstrip(".png") + "styled.png"
    logger.info("calling style transfer model, output file in tmp folder: " + output_file_name)
    cmd = "cd /mnt/fast-style-transfer; " \
          +"python evaluate.py --checkpoint /mnt/fast-style-transfer/checkpoints/johnny2 --in-path "\
          + input_file_path + input_file_name + " --out-path " + tmp_file_path + output_file_name\
          + " --device '/gpu:1'; cd /mnt/Detectron"
    logger.info("running command: "+ cmd)
    returned_val = os.system(cmd)

    logger.info("drawing border, returned value from style transfer: " + str(returned_val))
    logger.info("segmented image path: "+ tmp_file_path + output_file_name)
    logger.info("styled image path tmp: "+ tmp_file_path + output_file_name)

    img = cv2.imread(input_file_path + input_file_name)
    styled_img = cv2.imread(tmp_file_path + output_file_name)
    img_final = vis_utils.add_sticker_border(img, styled_img, mask)

    return returned_val, DIRECTORY_TO_WRITE,  output_file_name, img_final

# def video_processing(filepath, filename):
#     clip = VideoFileClip(filepath)
#     modified_video = clip.fl_image(video_image_segment)
#     modified_video.write_videofile("/home/srishti/outputvideo.mp4", audio=False);

def video_image_segment(im):
    filename="random" #not being used by vis_utils.segmented_images

    timers = defaultdict(Timer)
    t = time.time()

    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
    logger.info('Inference time: {:.3f}s'.format(time.time() - t))
    for k, v in timers.items():
        logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

    segmented_images, classes, scores, segmented_binary_masks = vis_utils.segmented_images_in_original_image_size(
        im,
        filename,
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
                found = True
                bin_mask = segmented_binary_masks[index]
                return found, value, bin_mask

    logger.warn(("PERSON NOT FOUND IN IMG!!!!!"))
    return found, segmented_images[0], segmented_binary_masks[0]

def video_processing_cv(filepath, filename, bg_filename):
    image_list = []

    bg_im = cv2.imread(VIDEO_BG_RESOURCES_DIRECTORY + bg_filename)

    vidcap = cv2.VideoCapture(filepath)
    success, image = vidcap.read()
    count = 0

    success = True
    while True:
        if (success and count<1):
            found, segmented_image,mask = video_image_segment(image)
            im_mask = vis_utils.vis_binary_mask(image, mask)
            image = vid_utils.process(image, im_mask, bg_im)
            image_list.insert(len(image_list), image)
        else:
            break
        success, image = vidcap.read()
        logger.info("Video inference count: "+ str(count))
        count += 1

    logger.info("Total number of frames in video: "+ str(count))
    new_video_filepath = DIRECTORY_TO_WRITE + "newVideo.mp4"
    vid_utils.write_images(image_list, new_video_filepath )
    write_to_gcs(new_video_filepath, gcs_filename="newVideo.mp4")
    logger.info("Done writing")


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
            logger.error("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            logger.info("Received created event - %s." % event.src_path)
            im_list = [event.src_path]
            k = event.src_path.rfind("/")
            original_filename = event.src_path[k+1:]

            if ".mp4" in original_filename:
                logger.info("need to proccess video")
                video_processing_cv(event.src_path, original_filename, "sky_news.jpg")
                logger.info("done mp4 processing")



            # Image segmentation
            else:
                found, filevalue, binmask_value = segment(im_list, event.src_path[k+1:])
                style = STICKER_SELFIE_HIT in original_filename
                if found:
                    gcs_filename = original_filename.rstrip(".jpg") + OUTPUT_FILE_EXTENSION
                    final_local_file = DIRECTORY_TO_WRITE + original_filename.rstrip(".jpg") + OUTPUT_FILE_EXTENSION
                    if style:
                        tmp_local_file = DIRECTORY_TEMP + original_filename.rstrip(".jpg") + "segmented.png"
                        write_to_local(tmp_local_file, filevalue)
                        returned_value, output_file_path, output_file_name, output_file_value = style_transfer(DIRECTORY_TEMP, original_filename.rstrip(".jpg") + "segmented.png", binmask_value)
                        write_to_local(output_file_path+output_file_name, output_file_value)
                        write_to_gcs(output_file_path + output_file_name, gcs_filename)
                    else:
                        write_to_local(final_local_file, filevalue)
                        write_to_gcs(final_local_file, gcs_filename)
                logger.info("segmentation done and saved: " + str(gcs_filename))



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


