
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

INPUT_VIDEO_PATH_METADATA = "/mnt/api_files/video/input/"
OUTPUT_VIDEO_FILE_EXTENSION = "_output.mp4"
VIDEO_METADATA_FILE_EXTENSION = "_metadata.txt"

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

def image_resize(image, height = None, width = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def segment(im_list, filename):
    for i, im_name in enumerate(im_list):
        out_name = os.path.join(
            args.output_dir, '{}'.format(os.path.basename(im_name) + '.pdf')
        )
        logger.info('Processing {} -> {}'.format(im_name, out_name))
        im = cv2.imread(im_name)
        if (im.shape[0] > 650):
            im = image_resize(im, height=600)
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

        return found, "", ""

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
    img_final = vis_utils.adjust_gamma(img_final, 0.5)

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

def video_processing_cv(filepath, filename, metadata):

    print(type(metadata['topLeft_bg_normalized_1']))
    print(type(metadata['topLeft_bg_normalized_1']))

    if isinstance(metadata['topLeft_bg_normalized_1'],float) == False:
        topLeft_bg_normalized =[float(metadata['topLeft_bg_normalized_1']), float(metadata['topLeft_bg_normalized_2'])]
        selected_bg_width_normalized = float(metadata['selected_bg_width_normalized'])
        selected_bg_height_normalized = float(metadata['selected_bg_height_normalized'])


    bg_filename = metadata['bg_filename']
    output_video_filename = filename.rstrip(".mp4") + OUTPUT_VIDEO_FILE_EXTENSION

    image_list = []
    input_image_list = []
    bin_mask_list = []
    bin_mask_img_list = []
    bg_im = cv2.imread(VIDEO_BG_RESOURCES_DIRECTORY + bg_filename)
    # returned_value = vid_utils.extract_audio(filepath, output_video_filename)
    logger.info("audio extraction done")

    vidcap = cv2.VideoCapture(filepath)
    success, image = vidcap.read()
    count = 0

    success = True
    while True:
        if (success):
            found, segmented_image,mask = video_image_segment(image)
            im_mask = vis_utils.vis_binary_mask(image, mask)
            # input_image_list.insert(len(input_image_list), image)
            # bin_mask_list.insert(len(bin_mask_list), mask)
            # bin_mask_img_list.insert(len(bin_mask_img_list), im_mask)
            image = vid_utils.process(image, im_mask, mask,bg_im, topLeft_bg_normalized, selected_bg_width_normalized, selected_bg_height_normalized)
            image_list.insert(len(image_list), image)
        else:
            break
        success, image = vidcap.read()
        logger.info("Video inference count: "+ str(count))
        count += 1

    # processed_images =[]
    # new_bin_masks = []
    #
    #
    #
    # for counter, bin_mask_img in enumerate(bin_mask_img_list):
    #     # logger.info("len(bin_mask_img_list): "+str(len(bin_mask_img_list)))
    #     # if counter == 0 or counter == 1 or counter == len(bin_mask_img_list)-1 or counter == len(bin_mask_img_list)-2:
    #     if counter == 0  or counter == len(bin_mask_img_list)-1:
    #         logger.info("in if: counter: "+ str(counter))
    #         new_bin_masks.insert(len(new_bin_masks), bin_mask_img)
    #         processed_image = vid_utils.process(input_image_list[counter], bin_mask_img, bg_im, topLeft_bg_normalized, selected_bg_width_normalized, selected_bg_height_normalized)
    #         processed_images.insert(len(processed_images), processed_image)
    #     else:
    #         logger.info("in else: counter: " + str(counter))
    #         mask_average = np.mean(bin_mask_img_list[counter-1:counter+1], axis=0)
    #         mask_round = np.round(mask_average)
    #         processed_image = vid_utils.process(input_image_list[counter], mask_round, bg_im, topLeft_bg_normalized,
    #                                             selected_bg_width_normalized, selected_bg_height_normalized)
    #         processed_images.insert(len(processed_images), processed_image)
    #

    logger.info("Total number of frames in video: "+ str(count))

    new_video_filepath = DIRECTORY_TO_WRITE + output_video_filename
    vid_utils.write_images(image_list, new_video_filepath, output_video_filename)
    write_to_gcs(new_video_filepath, output_video_filename)
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
                meta_filename = original_filename.rstrip(".mp4") + VIDEO_METADATA_FILE_EXTENSION
                metadata = yaml.safe_load(open(INPUT_VIDEO_PATH_METADATA + meta_filename))
                logger.info(metadata)
                logger.info(type(metadata))
                video_processing_cv(event.src_path, original_filename, metadata)
                logger.info("done mp4 processing")



            # Image segmentation
            else:
                logger.info("IMLIST size: " + str(len(im_list)))
                found, filevalue, binmask_value = segment(im_list, event.src_path[k+1:])
                style = STICKER_SELFIE_HIT in original_filename
                gcs_filename = original_filename.rstrip(".jpg") + OUTPUT_FILE_EXTENSION
                if found:
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


