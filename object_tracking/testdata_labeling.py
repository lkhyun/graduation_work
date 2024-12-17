import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

os.chdir(f"C:/Users/dlrkd/Desktop/graduation_work")
file_path = f"C:/Users/dlrkd/Desktop/graduation_work/newdata/UWB_Extration.mat"

# Load the .mat file
mat_data = scipy.io.loadmat(file_path)

com = np.empty((360, 2142), dtype=np.float64)
tv = np.empty((360, 2142), dtype=np.float64)

# Extract 'com' and 'tv' data from the struct
com = mat_data['UWBFilteredData']['com'][0, 0]  # Shape (360, 4284)
tv = mat_data['UWBFilteredData']['tv'][0, 0]  # Shape (360, 4284)

data_com = np.stack(np.split(com, 42, axis=1), axis=0)
data_tv = np.stack(np.split(tv, 42, axis=1), axis=0)

labels_all = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # com label
labels_all = np.tile(labels_all, 3)

original_array = data_com
modified_array_1 = np.sum(np.split(original_array, 72, axis=1), axis=2)
print(modified_array_1.shape)
modified_array_2 = np.sum(np.split(modified_array_1, 51, axis=2), axis=3)
print(modified_array_2.shape)
data_com = modified_array_2

original_array = data_tv
modified_array_1 = np.sum(np.split(original_array, 72, axis=1), axis=2)
print(modified_array_1.shape)
modified_array_2 = np.sum(np.split(modified_array_1, 51, axis=2), axis=3)
print(modified_array_2.shape)
data_tv = modified_array_2

data_com = np.transpose(data_com)
data_tv = np.transpose(data_tv)

data_com = np.reshape(data_com, (42, 3672))
data_tv = np.reshape(data_tv, (42, 3672))

combined = np.column_stack((data_com, data_tv))  # 라벨 합치기
combined = np.column_stack((combined, labels_all))

np.save(f"./newdata/finaltest.npy", combined)


