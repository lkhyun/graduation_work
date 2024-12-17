import os
import time
import numpy as np
from PIL import Image
from object_tracking import run as object_tracking_run
from posture_classfication import run as posture_classfication_run

def process_file(file_path, filename):
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.npy':
        data = np.load(file_path)
        object_tracking_run(data)
    elif file_extension == '.png':
        data = Image.open(file_path)
        posture_classfication_run(filename, data)
    else:
        print(f"Unsupported file format: {file_extension}")

def monitor_directory(directory_path):
    print(f"Monitoring directory: {directory_path}")
    processed_files = set()

    while True:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if file_path not in processed_files and os.path.isfile(file_path):
                process_file(file_path,filename)
                processed_files.add(file_path)
        time.sleep(0.1)

# 사용 예
monitor_directory('C:/Users/dlrkd/Desktop/graduation_work/newdata')
