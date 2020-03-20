import os
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

## other .py files
import data
import parser
import utils

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
LEARN_RATES = [0.01, 0.005, 0.001]

### training per epoch
def train_epoch(model,  trainloader,  criterion, device, optimizer):
    model.train()
    losses = []
    for batch_idx, (inputs,labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        labels = labels.type_as(outputs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    epoch_loss = sum(losses)/len(trainloader)
    return epoch_loss

### training per model
def train_modelcv(learn_rate, dataloader_cvtrain, dataloader_cvtest, model, criterion, optimizer, scheduler, num_epochs, device):
    best_measure = 0
    best_epoch =-1
    train_loss_ls = []
    val_loss_ls = []
    val_acc_ls = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train(True)
        train_loss=train_epoch(model, dataloader_cvtrain, criterion, device, optimizer)
        train_loss_ls.append(train_loss)
        #scheduler.step()
        print("train loss",train_loss)

        model.train(False)
        measure,val_loss = utils.evaluate(model, dataloader_cvtest, criterion, device)
        val_loss_ls.append(val_loss)
        val_acc_ls.append(measure)
        print('performance measure', measure)

        if measure > best_measure: 
            bestweights= model.state_dict()
            best_measure = measure
            best_epoch = epoch
            print('current best', measure, ' at epoch ', best_epoch)

    return best_epoch, best_measure, bestweights, train_loss_ls, val_loss_ls, val_acc_ls

### training pipeline for model selection
def train(device, loadertr, loadervl): 
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512,20)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    for learn_rate in LEARN_RATES:
        optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate, momentum=0.9, weight_decay=args.weight_decay)
        best_epoch, best_perfmeasure, bestweights, train_loss, val_loss, val_acc = train_modelcv(learn_rate=learn_rate,
                                                                                                    dataloader_cvtrain=loadertr, 
                                                                                                    dataloader_cvtest=loadervl,  
                                                                                                    model=model,  
                                                                                                    criterion=criterion, 
                                                                                                    optimizer=optimizer, 
                                                                                                    scheduler=None, 
                                                                                                    num_epochs=args.num_epoch, 
                                                                                                    device=device)
### produce results based on chosen model
def results(device, loadervl, validset): 
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512,20)
    model.load_state_dict(bestweights)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    ## Get tail accuracy
    # indices of top and bottom 50 images for each class, size = (50,20)
    idx_high, idx_low = utils.top_50_imgs(model,loadervl,device).numpy() 
    # dataset to print the images
    validset2=data.PascalVOC(args.data_dir,'val',transforms.Compose([ 
                                    transforms.CenterCrop(280),
                                    transforms.ToTensor()
                                    ]))
    #loader to calculate tail accuracy
    loadervl_tail = torch.utils.data.DataLoader(validset,batch_size=args.test_batch,shuffle=False,
                                            sampler=torch.utils.data.SubsetRandomSampler(idx_high.flatten().tolist()))
                                
    tail_acc = utils.tailacc(model,loadervl_tail,0.5,criterion,device).item() # change t value
    print('tail accuracy',tail_acc)

    for i in random.sample(range(20), 5): # 5 random classes out of 20
        print(validset2.list_image_sets()[i]) # class name
        time.sleep(0.5)
        for j in idx_high[:5,i]: # iterate through top 5 highest scoring images
            plt.figure()
            plt.imshow(np.transpose(validset2[j][0].numpy(),(1,2,0)))
            time.sleep(0.5)
        for j in idx_low[:5,i]: # iterate through top 5 lowest scoring images
            plt.figure()
            plt.imshow(np.transpose(validset2[j][0].numpy(),(1,2,0)))
            time.sleep(0.5)

def main():
    ## check GPU and set seed
    print("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)

    ## prepare directories
    if not os.path.exists(args.saved_img_dir):
        os.makedirs(args.saved_img_dir)
    if not os.path.exists(args.saved_model_dir):
        os.makedirs(args.saved_model_dir)
    if not os.path.exists(args.saved_pkl_dir):
        os.makedirs(args.saved_pkl_dir)

    ## prepare data
    transform = transforms.Compose([transforms.CenterCrop(280),
                                    transforms.ToTensor(),
                                    transforms.Normalize((MEAN, STD))
                                    ])
    trainset = data.PascalVOC(args.data_dir,'train',transform)
    validset = data.PascalVOC(args.data_dir,'val',transform)
    
    loadertr = torch.utils.data.DataLoader(trainset,batch_size=args.train_batch,shuffle=True)
    loadervl = torch.utils.data.DataLoader(validset,batch_size=args.test_batch,shuffle=False)
    
    ## run train/results
    if args.run == 'train':
        train(device, loadertr, loadervl)
    if args.run == 'results':
        results(device, loadervl, validset)

if __name__=='__main__':
    args = parser.arg_parse()
    main()
    
    
    