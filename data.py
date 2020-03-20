import os
import json
import torch
import scipy.misc
import pandas as pd
import numpy as np
from PIL import Image

import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class PascalVOC(Dataset):
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir,dataset,transform):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.dataset=dataset
        self.transform=transform
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        ##############################################################
        ### Pre-process data into len(images) x len(labels) matrix ###
        imgls = os.listdir(self.img_dir)
        class_ls=[]
        for i in self.list_image_sets():
          class_ls.append(set(self.imgs_from_category_as_list(i)))
        
        self.target=[ [] for i in range(len(imgls)) ]
        for i in range(len(imgls)-1,-1,-1): # for each image index
          for j in class_ls: # for each set of images from that class
            if os.path.splitext(imgls[i])[0] in j: # if filename is in that set of image
              self.target[i].append(1)
            else:
              self.target[i].append(0)
              # result in len(images) x len(labels) matrix
          
          if sum(self.target[i])==0: # if all labels are 0, remove entry from set
            self.target.pop(i)
            imgls.pop(i)
        
        self.imgls = imgls   
        ##############################################################

    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + self.dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name)
        df = df[df['true'] == 1]
        return df['filename'].values

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_name = os.path.join(self.img_dir, self.imgls[index])
        img = Image.open(img_name).convert('RGB')
        img.load()
        img = self.transform(img)
        targets = self.target[index]
        return img, np.array(targets)

    def __len__(self):
        return len(self.imgls)


