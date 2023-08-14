import sys
import threading
import traceback
import time

from model import Model
from jetson_utils import videoSource, videoOutput


class Stream(threading.Thread):
    """
    Thread for streaming video and applying DNN inference
    """
    def __init__(self, args):
        """
        Create a stream from input/output video sources, along with DNN models.
        """
        super().__init__()
        
        self.args = args
        self.input = videoSource(args.input, argv=sys.argv)
        self.output = videoOutput(args.output, argv=sys.argv)
        self.frames = 0
        self.models = {}
        self.process_count = 0
        self.latest_class_label = "Start"
        self.latest_class_id = -1
        self.detections = {}
        self.should_run = True
        self.model = Model(model=args.detection, labels=args.labels, colors=args.colors, input_layer=args.input_layer, output_layer=args.output_layer)
            
        
    def process(self):
        """
        Capture one image from the stream, process and save it.
        """
        img = self.input.Capture()
        
        if img is None:  # timeout
            return
            
        # get model results
        results = self.model.Process(img)
        if len(results) > 0:
            
            # results is a list of detected objects. The object with the highest confidence is at the top
            print(f"The results from the model are: {results}")
            
            # get the class id of the highest confidence class
            class_id = int(results[0].ClassID)       
            model_net = self.model.net
            class_label = model_net.GetClassLabel(class_id)
            
            print(f"The class label is: {class_label}")
            # save the detection in the dictionary
            if class_id in self.detections:
                self.detections[class_id] = self.detections[class_id] + 1
            else:
                self.detections[class_id] = 1      
            
        #visualize model results
        img = self.model.Visualize(img)
        self.output.Render(img)

        # print out performance info
        if self.frames % 25 == 0 or self.frames < 15:
            print(f"captured {self.frames} frames from {self.args.input} => {self.args.output} ({img.width} x {img.height})")
        self.frames += 1
        
        # reset the process count
        self.process_count += 1
        if self.process_count > 100:
            # set the class label
            class_label = self.latest_class_label
            class_id = self.latest_class_id
            highest_value = 10 # set to a higher number to prevent short detections
            for key, value in self.detections.items():
                if value > highest_value:
                    class_label = model_net.GetClassLabel(key)
                    highest_value = value
                    class_id = key
                    
            self.latest_class_label = class_label
            self.latest_class_id = class_id
            self.process_count = 0
            # remove all detections
            self.detections.clear()
                        
        
    def run(self):
        """
        Run the stream processing thread's main loop.
        """
        while self.should_run:
            try:
                self.process()
            except:
                traceback.print_exc()
                
                
    def get_latest_class_label(self):
        '''
        Returns the latest class label
        '''
        return self.latest_class_label
        
    def get_latest_class_id(self):
        '''
        Returns the latest class id
        '''
        return self.latest_class_id
        
        
    def stop(self):
        '''
        Stops the stream
        '''
        self.should_run = False       
        
                
    @staticmethod
    def usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return videoSource.Usage() + videoOutput.Usage() + Model.Usage()
        
