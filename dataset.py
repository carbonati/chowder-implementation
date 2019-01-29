import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ChowderDataset(Dataset):
    """Custom Dataset to load resnet feature data"""
    def __init__(self, label_path, data_path, img_ids, augmentor=None):
        """
            Args:
                label_path: path to labeled .csv DataFrame
                data_path: path to resnet feature directory
                img_ids: image IDs for train/validation split
                augmentor: torch transformer (default: None)
        """
        df = pd.read_csv(label_path)
        df = df.loc[df['ID'].isin(img_ids)]
        
        filepaths = [os.path.join(data_path,'{0}.npy'.format(fn)) for fn in img_ids]
        self.files = filepaths
        self.labels = list(df['Target'].values)
        
        self.augmentor = augmentor
    
    
    def preprocess(self, index):
        """
            Preprocesses .npy resnet features given an `index`
            
            Args:
                index: index to sample a feature from the Dataset
            
            Returns:
                resnet feature flattened to (1, N * P)
                    P = 2048
        """
        patient_features = np.load(self.files[index])
        resnet_features = torch.from_numpy(patient_features[:, 3:])
        resnet_features = resnet_features.reshape(1, -1)
        return resnet_features.type(torch.FloatTensor)
    
    
    def get_labels(self):
        """Returns list of target values from Dataset"""
        return self.labels
    
    
    def __getitem__(self, index):
        """Returns a pair of (preprocessed resnet features, label)"""
        resnet_features = self.preprocess(index)
        label = self.labels[index]
        return resnet_features, label
    
    
    def __len__(self):
        return len(self.files)


def chowder_collate(data):
    """Takes in mini-batch tensors as tuples of (resnet features, label)
        
     Since slides range from 40-1000 each batch may have different dimensions
     for each local descriptor. To account for this we can pad the end of each
     flattened resnet feature vector with 0's such that each feature vector
     will have the same shape as the maximum N tiled vector.
         The longest a feature vector can be is P*1000 = 2048*1000 = (1,2048000)
    
    Args:
        data: list of tuple (resnet features, label)
    
    Returns:
        chowder_features: resnet tensor of shape (batch_size, 1, padded_length)
        targets: tensor of shape (batch_size, 1)
    """
    resnet_features, targets = zip(*data)
    targets = torch.FloatTensor(targets)

    # `lengths` are calculated as P * {max # of tiles in batch}
    lengths = [features.shape[1] for features in resnet_features]
    chowder_features = torch.zeros(len(targets), 1, max(lengths))   
    for i, features in enumerate(resnet_features):
        feature_length = lengths[i]
        chowder_features[i, 0, :feature_length] = features[:feature_length]
    return chowder_features, targets