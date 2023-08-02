import os

class Handler():
    """
    Class for handling the detection results and providing the frontend with the
    current and next detection labels and image paths.
    """
    def __init__(self, stream, label_path:str):
        self.stream = stream
        self.labels = []
        if label_path is not "":
            self.labels = self._get_labels(label_path)
            self.labels.remove("BACKGROUND")
            print(f"The class labels are: {self.labels}")
        self.direction = 1
        
    
    def get_current_detection(self, short=True):
        '''
        Returns the current detection label.
        '''
        class_label = self.stream.get_latest_class_label()
        
        # if the class label is to long, the frontend gets issues with the visuals
        # therefore it is replaced here with a shorter name
        if short and len(class_label) > 10:
            class_label = "Stage" + str(self.labels.index(class_label))
            
        return class_label
    
    
    def get_next_detection(self, short=True):
        '''
        Returns the next detection label.
        '''
        # get current position for the current detection
        current_label = self.get_current_detection(False)
        
        if current_label == "Start" and self.direction == 1:
            return self.labels[0]
        elif current_label == "Start" and self.direction == -1:
            length = len(self.labels)
            return self.labels[length - 1]
        
        idx_current = self.labels.index(current_label)
        idx_next = idx_current + self.direction
        
        # check if this exceeds the number of labels
        if idx_next >= len(self.labels):
            return "Done"
        elif idx_next < 0:
            return "Done"
        
        next_label = self.labels[idx_next]
        # if the class label is to long, the frontend gets issues with the visuals
        # therefore it is replaced here with a shorter name
        if short and len(next_label) > 10:
            next_label = "Stage" + str(idx_next)
            
        return next_label
    
    
    def get_imagepath_for_currentlabel(self):
        '''
        Returns the path to the image for the current detection.
        '''
        label = self.get_current_detection(False)
        path = os.path.join("static", "images", label + ".jpg")
        return str(path)
    
    
    def get_imagepath_for_nextlabel(self):
        '''
        Returns the path to the image for the next detection.
        '''
        label = self.get_next_detection(False)
        path = os.path.join("static", "images", label + ".jpg")
        return str(path)
    
    
    def _get_labels(self, label_path):
        with open(label_path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels
    
    def set_direction(self, direction):
        if direction == 0:
            direction = -1
        self.direction = direction
    
    
