
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import uuid
import json

from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from flask import send_from_directory


app = Flask(__name__)

INPUT_FILE_PATH = "/mnt/api_files/input/"
GS_BUCKET = "gs://microapps-175405.appspot.com/srishti/"
CURL_PATH = "https://storage.googleapis.com/microapps-175405.appspot.com/srishti/"
OUTPUT_FILE_EXTENSION = '_output.png'
STICKER_SELFIE_HIT = "sticker"


INPUT_VIDEO_PATH_METADATA = "/mnt/api_files/video/input/"
OUTPUT_VIDEO_FILE_EXTENSION = "_output.mp4"
VIDEO_METADATA_FILE_EXTENSION = "_metadata.txt"

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/upload')
def upload_file():
    return render_template('uploader.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_done():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(INPUT_FILE_PATH + secure_filename(str(key) + '.jpg'))
        return jsonify(
            status='file uploaded successfully. Hit this to see results.',
            url = CURL_PATH + str(key) + OUTPUT_FILE_EXTENSION
        )

@app.route('/sticker', methods = ['POST'])
def upload_style_transfer_input():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(INPUT_FILE_PATH + secure_filename(str(key)) + STICKER_SELFIE_HIT + ".jpg")
        return jsonify(
            url = CURL_PATH + str(key) + STICKER_SELFIE_HIT + OUTPUT_FILE_EXTENSION
        )

@app.route('/potrait', methods = ['POST'])
def upload_potraitsegmentation():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(INPUT_FILE_PATH + secure_filename(str(key)) + ".jpg")
        return jsonify(
            url = CURL_PATH + str(key) + OUTPUT_FILE_EXTENSION
        )

@app.route('/test', methods = ['POST'])
def test():
    if request.method == 'POST':
        print(request)
        print(request.form)
        print(request.args)
        print(request.files)
        print(request.values)
        f = request.files['uploaded_file']
        filename = secure_filename(f.filename)
        return jsonify(
            status= "file found",
            name = filename
        )

@app.route('/video', methods = ['POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['uploaded_file']
        filename = secure_filename(f.filename)
        key = uuid.uuid4()


        parameters = {}
        parameters['topLeft_bg_normalized_1'] = request.args.get('topLeft_bg_normalized_1')
        parameters['topLeft_bg_normalized_2'] = request.args.get('topLeft_bg_normalized_2')
        parameters['selected_bg_width_normalized'] = request.args.get('selected_bg_width_normalized')
        parameters['selected_bg_height_normalized'] = request.args.get('selected_bg_height_normalized')
        parameters['bg_filename'] = request.args.get('bg_filename')

        print(type(parameters['topLeft_bg_normalized_1']))
        print(type(parameters['bg_filename']))
        print(parameters)

        with open(INPUT_VIDEO_PATH_METADATA + str(key) +VIDEO_METADATA_FILE_EXTENSION, 'w') as outfile:
            print("Dumping at :" + INPUT_VIDEO_PATH_METADATA + str(key) +VIDEO_METADATA_FILE_EXTENSION)
            json.dump(parameters, outfile)

        f.save(INPUT_FILE_PATH + str(key) + ".mp4")

        return jsonify(
            url = CURL_PATH + str(key) + OUTPUT_VIDEO_FILE_EXTENSION
        )

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)