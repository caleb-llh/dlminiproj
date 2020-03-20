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



def evaluate(model, dataloader,criterion,device):
    model.eval()
    total=0
    losses=[]
    with torch.no_grad():
        for ctr, (inputs,labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            loss = criterion(outputs, labels.to(device))
            losses.append(loss.item())
            cpuout= outputs.to('cpu')
            total += len(labels)

            if ctr <=0:
                current = cpuout.clone()
                lab = labels.to('cpu').clone()
            else:
                current = torch.cat((current,cpuout), dim=0)
                lab = torch.cat((lab,labels.to('cpu')), dim=0)

        class_correct = np.array(average_precision_score(lab, current,average=None))
        ave_loss = sum(losses)/len(dataloader)

    return class_correct, ave_loss


    

def tailacc(model, dataloader, t, criterion, device):
  model.eval()
  total=0
  losses=[]
  
  with torch.no_grad():
    for ctr, (inputs,labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        labels = labels.type_as(outputs)
        loss = criterion(outputs, labels.to(device))
        losses.append(loss.item())
        cpuout= outputs.to('cpu')
        total += len(labels)
        if ctr <=0:
            current = cpuout.clone()
            lab = labels.to('cpu').clone()
        else:
            current = torch.cat((current,cpuout), dim=0)    #concatenating outputs
            lab = torch.cat((lab,labels.to('cpu')), dim=0)  #concatenating laels

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

  scores = current.numpy()
  idx_high = np.argpartition(-scores,range(50),axis=0)[:50]   #top 50 images, size (50,20)
  idx_low = np.argpartition(scores,range(50),axis=0)[:50]     #bottom 50 images, size (50,20)
  
  return idx_high,idx_low