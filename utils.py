import os
import time
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score

import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data.dataset import Dataset


def save_model(model, save_path):
    torch.save(model.state_dict(),save_path) 


def evaluate(model, dataloader,device):
    model.eval()
    total=0
    # losses=[]
    with torch.no_grad():
        for ctr, (inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            cpuout= outputs.to('cpu')
            total += len(labels)

            # append predictions and target labels
            if ctr <=0:
                current = cpuout.clone()
                lab = labels.to('cpu').clone()
            else:
                current = torch.cat((current,cpuout), dim=0)
                lab = torch.cat((lab,labels.to('cpu')), dim=0)
            print("\r{}%".format(100*ctr/len(dataloader)),end='') # epoch progress

        class_precision = np.array(average_precision_score(lab, current,average=None))
        ave_precision = sum(class_precision)/len(class_precision)

    return class_precision, ave_precision


    

def tailacc(model, dataloader, t, device):
  model.eval()
  total=0
  losses=[]
  
  with torch.no_grad():
    for ctr, (inputs,labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        labels = labels.type_as(outputs)
        cpuout= outputs.to('cpu')
        total += len(labels)
        
        # append predictions and target labels
        if ctr <=0:
            current = cpuout.clone()
            lab = labels.to('cpu').clone()
        else:
            current = torch.cat((current,cpuout), dim=0)    #concatenating outputs
            lab = torch.cat((lab,labels.to('cpu')), dim=0)  #concatenating labels
        print("\r{}%".format(100*ctr/len(dataloader)),end='') # epoch progress 

  pred = torch.where(current>=t,torch.ones(current.size()),torch.zeros(current.size()))   #(condition, value for true, value for false)
  score = pred * lab
  score = torch.einsum('ij->', score)
  denominator = torch.einsum('ij->', pred)

  acc = (1/denominator) * score
  return acc


def top_50_imgs(model, dataloader, device):

  model.eval()
  with torch.no_grad():
      for ctr, (inputs,labels) in enumerate(dataloader):
          inputs = inputs.to(device)
          outputs = model(inputs)
          cpuout= outputs.to('cpu')
          if ctr <=0:
            current = cpuout.clone()
          else:
            current = torch.cat((current,cpuout), dim=0)      #creating large tensor of size (dataset size, 20)
          print("\r{}%".format(100*ctr/len(dataloader)),end='') # epoch progress 

  scores = current.numpy()
  idx_high = np.argpartition(-scores,range(50),axis=0)[:50]   #top 50 images, size (50,20)
  idx_low = np.argpartition(scores,range(50),axis=0)[:50]     #bottom 50 images, size (50,20)
  
  
  return idx_high,idx_low