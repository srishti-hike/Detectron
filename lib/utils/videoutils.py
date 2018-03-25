# import skvideo.io
# import cv2
import imageio
import numpy as np
import cv2

MILLISECONDS_IN_SECOND= 1000.0
DIRECTORY_TO_WATCH = "/mnt/api_files/input/"
DIRECTORY_TEMP = "/mnt/api_files/tmp/"
DIRECTORY_TO_WRITE = "/mnt/api_files/output/"
GS_BUCKET = "gs://microapps-175405.appspot.com/srishti/"
OUTPUT_FILE_EXTENSION = '_output.png'
STICKER_SELFIE_HIT = "sticker"

VIDEO_BG_RESOURCES_DIRECTORY = "/mnt/video_bg_resources/"

def write_images(images, new_video_filepath):
    writer = imageio.get_writer(new_video_filepath, fps=25)
    count = 0
    for im in images:
        count = count + 1
        print(count)
        writer.append_data(im)
    print("out of for loop")
    writer.close()
    print("done write_images")


def process(image, mask, bg, topLeft_bg_normalized, selected_bg_width_normalized, selected_bg_height_normalized):


    height_image, width_image, depth_image = image.shape
    height_bg, width_bg, depth_bg = image.shape


    topLeft_bg = [topLeft_bg_normalized[0] * width_bg, topLeft_bg_normalized[1] * height_bg]
    selected_bg_width = selected_bg_width_normalized * width_bg
    selected_bg_height = selected_bg_height_normalized * height_bg

    bg_image = np.zeros(shape=(int(selected_bg_height), int(selected_bg_width), 3))

    print("selected_bg_height" + str(selected_bg_height))
    print("selected_bg_width" + str(selected_bg_width))
    for i in range(0, int(selected_bg_height)):
        for j in range(0, int(selected_bg_width)):
            for k in range(0, 3):  # ...
                bg_image[i, j, k] = bg[int(topLeft_bg[1]) + i, int(topLeft_bg[0]) + j, k]

    bg_image_resized = cv2.resize(bg_image, (width_image, height_image))

    for i in range(0, height_image):
        for j in range(0, width_image):
            if mask[i, j] == 0:
                image[i, j, :] = bg_image_resized[i, j, :]

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # width_offset = 10
    # height_offset = 10*height_image/width_image
    # final_image = image[height_offset/2:height_image - height_offset/2, width_offset/2:width_image - width_offset/2]
    return image