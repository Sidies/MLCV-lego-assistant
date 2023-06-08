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

#from stream import Stream
from utils import rest_property
    
    
#parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=Stream.usage())
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
parser.add_argument("--ssl-key", default=".\\data\\localhost.key", type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
parser.add_argument("--ssl-cert", default=".\\data\\localhost.crt", type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
parser.add_argument("--title", default='Jetson Lego Classifier', type=str, help="the title of the webpage as shown in the browser")
parser.add_argument("--input", default='webrtc://@:8554/input', type=str, help="input camera stream or video file")
parser.add_argument("--output", default='webrtc://@:8554/output', type=str, help="WebRTC output stream to serve from --input")
parser.add_argument("--classification", default='', type=str, help="load classification model (see imageNet arguments)")
parser.add_argument("--detection", default='', type=str, help="load object detection model (see detectNet arguments)")
parser.add_argument("--segmentation", default='', type=str, help="load semantic segmentation model (see segNet arguments)")
parser.add_argument("--background", default='', type=str, help="load background removal model (see backgroundNet arguments)")
parser.add_argument("--action", default='', type=str, help="load action recognition model (see actionNet arguments)")
parser.add_argument("--pose", default='', type=str, help="load action recognition model (see actionNet arguments)")
parser.add_argument("--labels", default='', type=str, help="path to labels.txt for loading a custom model")
parser.add_argument("--colors", default='', type=str, help="path to colors.txt for loading a custom model")
parser.add_argument("--input-layer", default='', type=str, help="name of input layer for loading a custom model")
parser.add_argument("--output-layer", default='', type=str, help="name of output layer(s) for loading a custom model (comma-separated if multiple)")

args = parser.parse_known_args()[0]
    
    
# create Flask & stream instance
app = flask.Flask(__name__)
#stream = Stream(args)

# Whenever a user visits the root URL path, the index function is called and returns the rendered HTML 
# template for the index.html file. This allows the server to dynamically generate HTML content based 
# on the user's request and provide a customized response.
@app.route('/') # A flask app route is a URL that the app “listens” for.
def index():
    return flask.render_template('index.html', title=args.title, send_webrtc=args.input.startswith('webrtc'),
                                 input_stream=args.input, output_stream=args.output,
                                 classification=os.path.basename(args.classification), 
                                 detection=os.path.basename(args.detection), 
                                 segmentation=os.path.basename(args.segmentation), 
                                 pose=os.path.basename(args.pose),
                                 action=os.path.basename(args.action), 
                                 background=os.path.basename(args.background))
        
'''if args.detection:
    @app.route('/detection/enabled', methods=['GET', 'PUT'])
    def detection_enabled():
        return rest_property(stream.models['detection'].IsEnabled, stream.models['detection'].SetEnabled, bool)
      
    @app.route('/detection/confidence_threshold', methods=['GET', 'PUT'])
    def detection_confidence_threshold():
        return rest_property(stream.models['detection'].net.GetConfidenceThreshold, stream.models['detection'].net.SetConfidenceThreshold, float)
      
    @app.route('/detection/clustering_threshold', methods=['GET', 'PUT'])
    def detection_clustering_threshold():
        return rest_property(stream.models['detection'].net.GetClusteringThreshold, stream.models['detection'].net.SetClusteringThreshold, float)
        
    @app.route('/detection/overlay_alpha', methods=['GET', 'PUT'])
    def detection_overlay_alpha():
        return rest_property(stream.models['detection'].net.GetOverlayAlpha, stream.models['detection'].net.SetOverlayAlpha, float)
        
    @app.route('/detection/tracking_enabled', methods=['GET', 'PUT'])
    def detection_tracking_enabled():
        return rest_property(stream.models['detection'].net.IsTrackingEnabled, stream.models['detection'].net.SetTrackingEnabled, bool)

    @app.route('/detection/tracking_min_frames', methods=['GET', 'PUT'])
    def detection_tracking_min_frames():
        return rest_property(stream.models['detection'].net.GetTrackingParams, stream.models['detection'].net.SetTrackingParams, int, key='minFrames')

    @app.route('/detection/tracking_drop_frames', methods=['GET', 'PUT'])
    def detection_tracking_drop_frames():
        return rest_property(stream.models['detection'].net.GetTrackingParams, stream.models['detection'].net.SetTrackingParams, int, key='dropFrames')

    @app.route('/detection/tracking_overlap_threshold', methods=['GET', 'PUT'])
    def detection_tracking_overlap_threshold():
        return rest_property(stream.models['detection'].net.GetTrackingParams, stream.models['detection'].net.SetTrackingParams, int, key='overlapThreshold')
   '''

@app.route('/videoplayer')
def videoplayer():
    return flask.render_template('content/videoplayer.html')
    
# start stream thread
#stream.start()

# check if HTTPS/SSL requested
ssl_context = None

if args.ssl_cert and args.ssl_key:
    ssl_context = (args.ssl_cert, args.ssl_key)
    
# start the webserver
app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)
