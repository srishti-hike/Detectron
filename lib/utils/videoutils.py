import skvideo.io
# import cv2
# import imageio

MILLISECONDS_IN_SECOND= 1000.0
DIRECTORY_TO_WATCH = "/mnt/api_files/input/"
DIRECTORY_TEMP = "/mnt/api_files/tmp/"
DIRECTORY_TO_WRITE = "/mnt/api_files/output/"
GS_BUCKET = "gs://microapps-175405.appspot.com/srishti/"
OUTPUT_FILE_EXTENSION = '_output.png'
STICKER_SELFIE_HIT = "sticker"

VIDEO_BG_RESOURCES_DIRECTORY = "/mnt/video_bg_resources/"

# def write_images(images, new_video_filepath):
#     writer = imageio.get_writer(new_video_filepath, fps=25)
#     count = 0
#     for im in images:
#         count = count + 1
#         print(count)
#         writer.append_data(im)
#     print("out of for loop")
#     writer.close()
#     print("done write_images")


def write_images(images, new_video_filepath):
    print("in function write_images")
    writer = skvideo.io.FFmpegWriter(new_video_filepath)
    count = 0
    for im in images:
        print("writing: "+ str(count))
        writer.writeFrame(im)
        count = count + 1
    print("finished writing")
    writer.close()
    print("returning from write_images")
