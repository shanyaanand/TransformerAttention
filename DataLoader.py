import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os 
import torch
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler



class csvloader(Dataset):

    """csv dataset"""
    def __init__(self, root_dir, idx, mode = 'Train', transform = None): 

        """
            root_dir : path to csv file folder
            idx : file index to load
        """
        self.root_dir = root_dir
        self.idx = idx
        self.transform = transform
        self.Folders = self.get_data(mode)

    def get_data(self, mode):

        """
        mode : if Train then do Undersampling 
        output : List of All unqiue folder present
        """
        # list of all .csv file in a folder 
        files = sorted(os.listdir(self.root_dir))
        files = np.array(files)
        # selecting files to load
        selected = files[self.idx]
        temp1 = pd.read_csv(self.root_dir + '/' + selected[0])
        temp1["DataFileIndex"] = np.zeros(len(temp1))# DataFileIndex : for taking account of different month change file
        temp = (temp1).values
        for i in range(1, len(selected)):
            temp1 = (pd.read_csv(self.root_dir + '/' + selected[i]))
            temp1["DataFileIndex"] = np.ones(len(temp1))*i       
            temp = np.concatenate((temp, temp1.values), axis = 0)

        if mode == 'Train':
            """
            Undersampling
            """
            # for class 0
            no_of_defects = sum(temp[:, -2])
            idx_of_nondefect = np.squeeze(np.array(np.where(temp[:, -2] == 0)))        
            selected_nondefect = np.random.choice(idx_of_nondefect, int(no_of_defects*0.12))
            temp = np.concatenate((temp[selected_nondefect], temp[np.squeeze(np.array(np.where(temp[:, -2] == 1)))]), axis=0)

            # for class 1
            no_of_defects = sum(temp[:, -3] != 1)
            idx_of_nondefect = np.squeeze(np.array(np.where(temp[:, -3] == 1)))        
            selected_nondefect = np.random.choice(idx_of_nondefect, int(no_of_defects*0.2))
            temp = np.concatenate((temp[selected_nondefect], temp[np.squeeze(np.array(np.where(temp[:, -3] != 1)))]), axis=0)

            # for class 2
            no_of_defects = sum(temp[:, -3] != 2)
            idx_of_nondefect = np.squeeze(np.array(np.where(temp[:, -3] == 2)))        
            selected_nondefect = np.random.choice(idx_of_nondefect, int(no_of_defects*0.2))
            temp = np.concatenate((temp[selected_nondefect], temp[np.squeeze(np.array(np.where(temp[:, -3] != 2)))]), axis=0)
        """
        Making lanels for multi-class classification
        """
	uqn = (temp[:, -3])
	cut1 = (uqn < 5.1)*1
	cut2 = (uqn > 5.1)*1
	temp[:, -3] = (cut1*uqn + cut2*6).astype('float64')
        self.Data = temp
        mean = np.mean(np.array(temp[:, 1:-3]))# TODO : use mean and std of training data to normalize test data 
        std = np.std(np.array(temp[:, 1:-3]))
        self.Data[:, 1:-3] = (temp[:, 1:-3] - mean)/std
        self.frequency()
        Folder = self.ListofFolders(self.Data)
        return Folder

    def Undersampling(self, data):

       print("number of data before UnderSampling {}".format(data.shape[0]))
       X_data, y_data = data, data[:,-2]
       rus = RandomUnderSampler(random_state=0)
       X_resampled, y_resampled = rus.fit_resample(X_data, y_data)
       print("number of data after UnderSampling {}".format(X_resampled.shape[0]))
       X_resampled[:, -2] = y_resampled
       return X_resampled

    def frequency(self):
        label = self.Data[:, -3]
        uq = np.unique(label)
        for i in uq:
            print(i, sum(label == i))

    def ListofFolders(self, data):
        """
        args : numpy Data, shape = [None, number of features] 
        description : Output all the unique folder present in the data
        """
        lists = data[:, 0]
        folder_lists = []
        label = []
        folder_list = []
        for i in lists:
            end = len(i) - 1
            for j in range(len(i)):
                if i[end - j] == '/':
                    folder_list.append(i[:(end-j)])
                    break
        
        return np.array(np.squeeze(np.unique(folder_list)))

    def FolderGrouping(self, folder, data):
        
        """
           Grouping .csv file in same group which are in the same folder
           output : [number of files, number of features]( number of files will be variable)
        """
        # TODO : Group file which are in the same month file window
        lists = data[:, 0]
        Grouping = []# to store group of features  
        filename = []# to store filename
        folder_list = []# to store file address except filename
        label = []
        for i in lists:
            end = len(i) - 1
            for j in range(len(i)):
                if i[end - j] == '/':
                    folder_list.append(i[:(end - j)])
                    filename.append(i[(end - j):])
                    break

        for folder_i in range(len(folder_list)):# loop iterating over list of folder address

            if folder_list[folder_i] == folder:
                Grouping.append(data[folder_i, 1:-3])
                label.append(data[folder_i, -3])
        return (Grouping), (label)                       


    def __len__(self):
        return self.Folders.shape[0]

    def __getitem__(self, idx):
        
        X, y = self.FolderGrouping(self.Folders[idx], self.Data)
        sample = {'data': torch.Tensor(X), 'label': torch.Tensor(y)}
        if self.transform:
            sample = self.transform(sample)
        return sample            


