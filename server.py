import os
import json

from flask import Flask, jsonify
from flask import request
from flask_cors import CORS
from flask_cors import cross_origin
from flask.json import JSONEncoder
from datetime import datetime
import gevent
from werkzeug.utils import secure_filename
import pickle

import utils

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'wav'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)

import soundfile as sf
import numpy as np
import tensorflow as tf

block_len = 512
block_shift = 128
in_buffer = np.zeros((block_len))
out_buffer = np.zeros((block_len))
# load model
model = tf.saved_model.load('./pretrained_model/dtln_saved_model')
infer = model.signatures["serving_default"]

def _get(dictionary, key, empty_allowed=True):
    if key in dictionary:
        _val = dictionary[key]
        if empty_allowed:
            return True, _val
        else:
            if sutil.is_null_or_empty(_val):
                return False, _val
            else:
                return True, _val
    else:
        return False, None

@app.route("/call_denoise/", methods=["POST"])
@cross_origin(supports_credentials=True)
def update_basic_detail_profile_image():
    audio_file = request.files['file']
    audio,fs = sf.read(audio_file)
    # check for sampling rate
    if fs != 16000:
        raise ValueError('This model only supports 16k sampling rate.')
    # preallocate output audio
    out_file = np.zeros((len(audio)))
    # create buffer
    # calculate number of blocks
    num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
    
    for idx in range(num_blocks):
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
        # create a batch dimension of one
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
        # process one block
        out_block= infer(tf.constant(in_block))['conv1d_1']
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer  += np.squeeze(out_block)
        # write block to output file
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    return jsonify(status=200, data=out_file)

if __name__ == "__main__":
    from gevent.pywsgi import WSGIServer
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    log.info('after initing to run on 5000')