import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

for j in range(1,49):
    file_path = f"/home/user2/radar_project/data/{j}/UWB_Extration.mat"

    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)

    com = np.empty((360,4284),dtype=np.float64)
    tv = np.empty((360,4284),dtype=np.float64)
    
    # Extract 'com' and 'tv' data from the struct
    com = mat_data['UWBFilteredData']['com'][0,0]  # Shape (360, 4284)
    tv = mat_data['UWBFilteredData']['tv'][0,0]    # Shape (360, 4284)
    
    data_com = np.stack(np.split(com,84,axis=1),axis=0)
    data_tv = np.stack(np.split(tv,84,axis=1),axis=0)
    
    labels_com = np.array([0,1,2,3,4,4,4,4,4,4,5,6,7,8]) #com label
    labels_com = np.tile(labels_com,6)
    labels_tv = np.array([0,1,2,2,2,2,2,2,2,2,3,4,5,6]) #tv label
    labels_tv = np.tile(labels_tv,6)

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
    # plt.imshow(data_com[0,:,:], cmap='viridis')
    # plt.colorbar()  # 색상 바 추가
    # plt.title('Heatmap using matplotlib')
    # plt.show()

    data_com = np.reshape(data_com, (84,3672))
    data_tv = np.reshape(data_tv, (84,3672))
    
    # 결과 출력
    print("Labels for each time step(com):")
    print(labels_com)

    # 결과 출력
    print("Labels for each time step(tv):")
    print(labels_tv)
    
    combined_com = np.column_stack((data_com,labels_com)) # 라벨 합치기
    combined_tv = np.column_stack((data_tv,labels_tv)) # 라벨 합치기
    print(combined_com.shape)
    print(combined_tv.shape)
    
    np.save(f"./{j}_com.npy",combined_com)
    np.save(f"./{j}_tv.npy",combined_tv)


