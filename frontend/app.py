# This python file is the entry point for starting the projects frontend.
# It is a Flask web application that serves a webpage for the Jetson Nano
# based Lego classifier. The application uses a custom class called Stream
# that is used to detect Lego objects in a camera stream. The user can
# interact with the application through the web interface to start or 
# disable the object detection and adjust detection parameters.
# This file also provides REST endpoints for accessing the detection
# parameters and communicate the current detection with the web page.
# This file includes command line arguments for configuring the 
# web server, input and output video streams and the detection model

import os
import flask
import argparse

from stream import Stream
# The mocking file is used for testing the frontend without a Jetson Nano
# run without jetson: python app.py --ssl-key data/RootCA.key --ssl-cert data/RootCA.crt --labels ../models/Single_Object/Iteration_1/lego_Iteration_1_labels.txt
# from streamMock import StreamMock
from utils import rest_property
from handler import Handler

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
                                 
# Create flask app routes for the REST endpoints                                
if args.detection:      
    @app.route('/detection/confidence_threshold', methods=['GET', 'PUT'])
    def detection_confidence_threshold():
        '''
        REST endpoint for getting and setting the detection confidence threshold property
        '''
        return rest_property(stream.model.net.GetConfidenceThreshold, stream.model.net.SetConfidenceThreshold, float)
      
    @app.route('/detection/clustering_threshold', methods=['GET', 'PUT'])
    def detection_clustering_threshold():
        '''
        REST endpoint for getting and setting the detection clustering threshold property
        '''
        return rest_property(stream.model.net.GetClusteringThreshold, stream.model.net.SetClusteringThreshold, float)
        
    @app.route('/detection/overlay_alpha', methods=['GET', 'PUT'])
    def detection_overlay_alpha():
        '''
        REST endpoint for getting and setting the detection overlay alpha property
        '''
        return rest_property(stream.model.net.GetOverlayAlpha, stream.model.net.SetOverlayAlpha, float)

    @app.route('/detection/tracking_min_frames', methods=['GET', 'PUT'])
    def detection_tracking_min_frames():
        '''
        REST endpoint for getting and setting the detection tracking min frames property
        '''
        return rest_property(stream.model.net.GetTrackingParams, stream.model.net.SetTrackingParams, int, key='minFrames')
    
    @app.route('/detection/current_label', methods=['GET'])
    def detection_get_latest_label():
        '''
        REST endpoint for getting the current detection label
        '''
        return rest_property(handler.get_current_detection, None, str)
    
    @app.route('/detection/next_label', methods=['GET'])
    def detection_get_next_label():
        '''
        REST endpoint for getting the next detection label
        '''
        return rest_property(handler.get_next_detection, None, str)
    
    @app.route('/detection/current_label_image', methods=['GET'])
    def detection_get_latest_label_image():
        '''
        REST endpoint for getting the current detection label image path
        '''
        return rest_property(handler.get_imagepath_for_currentlabel, None, str)
    
    @app.route('/detection/next_label_image', methods=['GET'])
    def detection_get_next_label_image():
        '''
        REST endpoint for getting the next detection label image path
        '''
        return rest_property(handler.get_imagepath_for_nextlabel, None, str)
    
    @app.route('/detection/set_direction', methods=['PUT'])
    def detection_set_direction():
        '''
        REST endpoint for setting the direction of the building process
        '''
        return rest_property(None, handler.set_direction, int)
    
    @app.route('/detection/pause', methods=['PUT'])
    def detection_pause():
        '''
        REST endpoint for pausing the detection of the building process
        '''
        return rest_property(None, stream.model.SetEnabled, bool)
    
# start stream thread
stream.start()

# check if HTTPS/SSL requested
ssl_context = None
if args.ssl_cert and args.ssl_key:
    ssl_context = (args.ssl_cert, args.ssl_key)
    
# start the webserver
app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)




