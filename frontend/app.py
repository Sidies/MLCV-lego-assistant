#!/usr/bin/env python3
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import os
import flask
import argparse

from stream import Stream
#from streamMock import StreamMock
from utils import rest_property
from handler import Handler
    
# run without jetson: python app.py --ssl-key data/RootCA.key --ssl-cert data/RootCA.crt --labels ../models/Single_Object/Iteration_1/lego_Iteration_1_labels.txt

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=Stream.usage())
#parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
parser.add_argument("--ssl-key", default="../../jetson-inference/data/key.pem", type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
parser.add_argument("--ssl-cert", default="../../jetson-inference/data/cert.pem", type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
parser.add_argument("--title", default='Jetson Lego Classifier', type=str, help="the title of the webpage as shown in the browser")
parser.add_argument("--input", default='webrtc://@:8554/input', type=str, help="input camera stream or video file")
parser.add_argument("--output", default='webrtc://@:8554/output', type=str, help="WebRTC output stream to serve from --input")
parser.add_argument("--detection", default='../models/lego3.onnx', type=str, help="load object detection model (see detectNet arguments)")
parser.add_argument("--labels", default='../data/lego/labels.txt', type=str, help="path to labels.txt for loading a custom model")
parser.add_argument("--colors", default='', type=str, help="path to colors.txt for loading a custom model")
parser.add_argument("--input-layer", default='input_0', type=str, help="name of input layer for loading a custom model")
parser.add_argument("--output-layer", default='', type=str, help="name of output layer(s) for loading a custom model (comma-separated if multiple)")

args = parser.parse_known_args()[0]
    
# create Flask & stream instance
app = flask.Flask(__name__)

stream = Stream(args)
#stream = StreamMock(args)
handler = Handler(stream, args.labels)

# Whenever a user visits the root URL path, the index function is called and returns the rendered HTML 
# template for the index.html file. This allows the server to dynamically generate HTML content based 
# on the user's request and provide a customized response.
@app.route('/') # A flask app route is a URL that the app “listens” for.
def index():
    return flask.render_template('index.html', title=args.title, send_webrtc=args.input.startswith('webrtc'),
                                 input_stream=args.input, output_stream=args.output,                                  
                                 detection=os.path.basename(args.detection))
        
if args.detection:
    @app.route('/detection/enabled', methods=['GET', 'PUT'])
    def detection_enabled():
        return rest_property(stream.model.IsEnabled, stream.model.SetEnabled, bool)
      
    @app.route('/detection/confidence_threshold', methods=['GET', 'PUT'])
    def detection_confidence_threshold():
        return rest_property(stream.model.net.GetConfidenceThreshold, stream.model.net.SetConfidenceThreshold, float)
      
    @app.route('/detection/clustering_threshold', methods=['GET', 'PUT'])
    def detection_clustering_threshold():
        return rest_property(stream.model.net.GetClusteringThreshold, stream.model.net.SetClusteringThreshold, float)
        
    @app.route('/detection/overlay_alpha', methods=['GET', 'PUT'])
    def detection_overlay_alpha():
        return rest_property(stream.model.net.GetOverlayAlpha, stream.model.net.SetOverlayAlpha, float)
        
    @app.route('/detection/tracking_enabled', methods=['GET', 'PUT'])
    def detection_tracking_enabled():
        return rest_property(stream.model.net.IsTrackingEnabled, stream.model.net.SetTrackingEnabled, bool)

    @app.route('/detection/tracking_min_frames', methods=['GET', 'PUT'])
    def detection_tracking_min_frames():
        return rest_property(stream.model.net.GetTrackingParams, stream.model.net.SetTrackingParams, int, key='minFrames')

    @app.route('/detection/tracking_drop_frames', methods=['GET', 'PUT'])
    def detection_tracking_drop_frames():
        return rest_property(stream.model.net.GetTrackingParams, stream.model.net.SetTrackingParams, int, key='dropFrames')

    @app.route('/detection/tracking_overlap_threshold', methods=['GET', 'PUT'])
    def detection_tracking_overlap_threshold():
        return rest_property(stream.model.net.GetTrackingParams, stream.model.net.SetTrackingParams, int, key='overlapThreshold')
    
    @app.route('/detection/current_label', methods=['GET'])
    def detection_get_latest_label():
        return rest_property(handler.get_current_detection, None, str)
    
    @app.route('/detection/next_label', methods=['GET'])
    def detection_get_next_label():
        return rest_property(handler.get_next_detection, None, str)
    
    @app.route('/detection/current_label_image', methods=['GET'])
    def detection_get_latest_label_image():
        return rest_property(handler.get_imagepath_for_currentlabel, None, str)
    
    @app.route('/detection/next_label_image', methods=['GET'])
    def detection_get_next_label_image():
        return rest_property(handler.get_imagepath_for_nextlabel, None, str)
    
    @app.route('/detection/set_direction', methods=['PUT'])
    def detection_set_direction():
        return rest_property(None, handler.set_direction, int)
   
    
# start stream thread
stream.start()

# check if HTTPS/SSL requested
ssl_context = None

if args.ssl_cert and args.ssl_key:
    ssl_context = (args.ssl_cert, args.ssl_key)
    
# start the webserver
app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)
