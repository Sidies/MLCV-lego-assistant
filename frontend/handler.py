import os

class Handler():
    def __init__(self, stream, label_path:str):
        self.stream = stream
        self.labels = []
        if label_path is not "":
            self.labels = self._get_labels(label_path)
            print(f"The class labels are: {self.labels}")
        
    
    def get_current_detection(self):
        return self.stream.get_latest_class_label()
    
    
    def get_next_detection(self):
        # get current position for the current detection
        current_label = self.get_current_detection()
        idx_current = self.labels.index(current_label)
        idx_next = idx_current + 1
        
        # check if this exceeds the number of labels
        if idx_next >= len(self.labels):
            return "Done"
        return self.labels[idx_next]
    
    
    def get_imagepath_for_currentlabel(self):
        label = self.get_current_detection()
        path = os.path.join("static", "images", label + ".jpg")
        return str(path)
    
    
    def get_imagepath_for_nextlabel(self):
        label = self.get_next_detection()
        path = os.path.join("static", "images", label + ".jpg")
        return str(path)
    
    
    def _get_labels(self, label_path):
        with open(label_path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels