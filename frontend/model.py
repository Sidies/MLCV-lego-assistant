from jetson_inference import imageNet, detectNet, segNet, poseNet, actionNet, backgroundNet
from jetson_utils import cudaFont, cudaAllocMapped, Log
import inspect


class Model:
    """
    Loads the DNN model and provides methods for processing images and visualizing results.
    """
    def __init__(self, model, labels='', colors='', input_layer='', output_layer='', **kwargs):
        """
        Load the model, either from a built-in pre-trained model or from a user-provided model.
        
        Parameters:
        
            model (string) -- either a path to the model or name of the built-in model
            labels (string) -- path to the model's labels.txt file
            input_layer (string or dict) -- the model's input layer(s)
            output_layer (string or dict) -- the model's output layers()
        """
        self.model = model
        self.enabled = True
        self.results = None
        self.frames = 0
        
        if not output_layer:
            output_layer = {'scores': 'scores', 'boxes': 'boxes'}
        elif isinstance(output_layer, str):
            output_layer = output_layer.split(',')
            output_layer = {'scores': output_layer[0], 'boxes': output_layer[1]}
        elif not isinstance(output_layer, dict) or output_layer.keys() < {'scores', 'boxes'}:
            raise ValueError("for detection models, output_layer should be a dict with keys 'scores' and 'bbox'")

        self.net = detectNet(model=model, labels=labels, colors=colors,
                                input_blob=input_layer, 
                                output_cvg=output_layer['scores'], 
                                output_bbox=output_layer['boxes'])

            
    def Process(self, img):
        """
        Process an image with the model and return the results.
        """
        if not self.enabled:
            return []     

        self.results = self.net.Detect(img, overlay='none')
        
        self.frames += 1
        return self.results


    def Visualize(self, img, results=None):
        """
        Visualize the results on an image.
        """
        if not self.enabled:
            return img
            
        if results is None:
            results = self.results
    
        # start the overlay for detection
        self.net.Overlay(img, results)
            
        return img
        
    def IsEnabled(self):
        """
        Returns true if the model is enabled for processing, false otherwise.
        """
        return self.enabled
        
    def SetEnabled(self, enabled):
        """
        Enable/disable processing of the model.
        """
        self.enabled = enabled

        
    @staticmethod
    def Usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return imageNet.Usage() + detectNet.Usage() + segNet.Usage() + actionNet.Usage() + poseNet.Usage() + backgroundNet.Usage() 
