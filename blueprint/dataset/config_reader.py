import json

class ConfigReader():
    def __init__(self):
        self.data = None

    def read(self, file_name):
        with open(file_name) as f:
            self.data = json.load(f)
        return self.data

    def get(self, object_name):
        if self.data is not None:
            return self.data[object_name]
        else:
            return None


