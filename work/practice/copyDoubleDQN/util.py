import csv
import os

def make_directory(directory):
    current_dir = os.getcwd()
    directory = os.path.join(current_dir, directory)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

class RecordHistory:
    def __init__(self, csv_path, header):
        self.csv_path = csv_path
        self.header = header
    
    def generate_csv(self):
        with open(self.csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

    def add_histry(self, history):
        history_list = [history[key] for key in self.header]
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(history_list)
    
    def add_list(self, array):
        with open(self.csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(array)