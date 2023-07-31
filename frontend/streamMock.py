import sys
import threading
import traceback
import time
from unittest.mock import MagicMock

class StreamMock(threading.Thread):
    
    """
    Thread for streaming video and applying DNN inference
    """
    def __init__(self, args):
        """
        Create a stream from input/output video sources, along with DNN models.
        """
        super().__init__()
        
        self.args = args
        self.input = MagicMock()
        self.output = MagicMock()
        self.frames = 0
        self.models = {}
        self.process_count = 0
        self.latest_class_label = "Start"
        
        # these are in the order that the overlays should be composited

        self.model = MagicMock()
            
        
    def process(self):
        """
        Capture one image from the stream, process it, and output it.
        """
        img = self.input.Capture()
        
        if img is None:  # timeout
            return
            
        # get model results
        results = self.model.Process(img)
        if len(results) > 0:
            
            # results is a list of detected objects. The object with the highest confidence seems to be at the top            
            print(f"The results from the model are: {results}")
            
            # get the class id of the highest confidence class
            class_id = int(results[0].ClassID)            
            
            model_net = self.model.net
            class_label = model_net.GetClassLabel(class_id)
            
            print(f"The class label is: {class_label}")
            if self.latest_class_label == "":
                self.latest_class_label = class_label

            
        #visualize model results
        img = self.model.Visualize(img)

        self.output.Render(img)

        if self.frames % 25 == 0 or self.frames < 15:
            print(f"captured {self.frames} frames from {self.args.input} => {self.args.output} ({img.width} x {img.height})")
   
        self.frames += 1
        
        # reset the process count
        self.process_count += 1
        if self.process_count > 100:
            self.process_count = 0
            self.latest_class_label = ""
            
        
    def run(self):
        """
        Run the stream processing thread's main loop.
        """
        while True:
            try:
                #self.process()
                pass
            except:
                pass
            
    def stop(self):
        self.should_run = False               
                
    def get_latest_class_label(self):
        print("Getting latest class label")
        print("_________________________________________")
        return self.latest_class_label
                
    @staticmethod
    def usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return "videoSource.Usage() + videoOutput.Usage() + Model.Usage()"
        
