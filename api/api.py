
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import uuid

from flask import Flask, render_template, request
from werkzeug import secure_filename
from flask import send_from_directory

app = Flask(__name__)

INPUT_FILE_PATH = "/mnt/api_files/input/"
GS_BUCKET = "gs://microapps-175405.appspot.com/srishti/"
CURL_PATH = "https://storage.googleapis.com/microapps-175405.appspot.com/srishti/"
OUTPUT_FILE_EXTENSION = '_output.png'
STICKER_SELFIE_HIT = "sticker"

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
        return 'file uploaded successfully. Hit this to see results: : '+ CURL_PATH + str(key) + OUTPUT_FILE_EXTENSION

@app.route('/sticker', methods = ['POST'])
def upload_style_transfer_input():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(INPUT_FILE_PATH + secure_filename(str(key)) + STICKER_SELFIE_HIT + ".jpg")
        return CURL_PATH + str(key) + STICKER_SELFIE_HIT + OUTPUT_FILE_EXTENSION

@app.route('/potrait', methods = ['POST'])
def upload_potraitsegmentation():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(INPUT_FILE_PATH + secure_filename(str(key)) + ".jpg")
        return CURL_PATH + str(key) + OUTPUT_FILE_EXTENSION

@app.route('/video', methods = ['POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        return filename

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)