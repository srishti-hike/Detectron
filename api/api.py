
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import uuid
import json

from flask import Flask, render_template, request, jsonify
from werkzeug import secure_filename
from flask import send_from_directory

from apiutils.config import config
from apiutils import segmentation


app = Flask(__name__)

# INPUT_FILE_PATH = "/mnt/api_files/input/"
# GS_BUCKET = "gs://microapps-175405.appspot.com/srishti/"
# CURL_PATH = "https://storage.googleapis.com/microapps-175405.appspot.com/srishti/"
# OUTPUT_FILE_EXTENSION = '_output.png'
# STICKER_SELFIE_HIT = "sticker"

conf = config.get_config()
seg = segmentation.Segmentation()

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/sticker', methods = ['POST'])
def upload_style_transfer_input():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        im_path = conf['io']['input_dir'] + secure_filename(str(key)) + conf['io']['selfie_sticker_hit'] + ".jpg"
        f.save(im_path)
        output_filename = seg.process_image(im_path)
        return jsonify(
            url = conf['io']['curl_path'] + output_filename
        )

@app.route('/upload')
def upload_file():
    return render_template('uploader.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file_done():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(conf['io']['input_dir'] + secure_filename(str(key) + '.jpg'))
        return jsonify(
            status='file uploaded successfully. Hit this to see results.',
            url = conf['io']['curl_path'] + str(key) + conf['io']['output_file_extension']
        )

@app.route('/potrait', methods = ['POST'])
def upload_potraitsegmentation():
    if request.method == 'POST':
        f = request.files['file']
        key = uuid.uuid4()
        f.save(conf['io']['input_dir'] + secure_filename(str(key)) + ".jpg")
        return jsonify(
            url = conf['io']['curl_path'] + str(key) + conf['io']['output_file_extension']
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)