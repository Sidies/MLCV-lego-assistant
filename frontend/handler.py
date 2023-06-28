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
    
    
    def _get_imagepath_for_label(self):
        label = self.get_current_detection()
        print("Getting image path for label: ", label)
        path = os.path.join("static", "images", label + ".jpg")
        print("The path is: ", path)
        return str(path)
        
    
    def _get_labels(self, label_path):
        with open(label_path) as f:
            labels = [line.strip() for line in f.readlines()]
        return labels