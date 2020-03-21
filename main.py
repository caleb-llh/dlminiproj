import os
import time
import random
import pickle

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
        print("\r{} % ".format(100*batch_idx/len(trainloader)),end='') # epoch progress
    print("\r",end='')
    epoch_loss = sum(losses)/len(trainloader)
    return epoch_loss

### training per model
def train_modelcv(learn_rate, dataloader_cvtrain, dataloader_cvtest, model, criterion, optimizer, scheduler, num_epochs, device):
    best_measure = 0
    best_epoch =-1
    train_loss_ls = []
    # val_loss_ls = []
    val_acc_ls = []
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        model.train()
        train_loss=train_epoch(model, dataloader_cvtrain, criterion, device, optimizer)
        train_loss_ls.append(train_loss)
        #scheduler.step()
        print("Train loss:",train_loss)

        _, measure = utils.evaluate(model, dataloader_cvtest, device)
        # val_loss_ls.append(val_loss)
        val_acc_ls.append(measure)
        print('Averge precision:', measure)

        if measure > best_measure: 
            # save best weights
            utils.save_model(model, os.path.join(args.saved_model_dir, "model_best_{}.pt".format(str(learn_rate)[2:])))
            best_measure = measure
            best_epoch = epoch
            print('Current best:', measure, ', at epoch', best_epoch)

    return best_epoch, best_measure, train_loss_ls, val_acc_ls

### training pipeline for model selection
def train(device, loadertr, loadervl): 
    criterion = nn.BCEWithLogitsLoss()
    best_measure_ls = []
    for learn_rate in LEARN_RATES:
        title = "Learn rate: {}".format(learn_rate)
        print(title)
        model = models.resnet18(pretrained=True) #reinitialise model
        model.fc = nn.Linear(512,20)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(),lr=learn_rate, momentum=0.9, weight_decay=args.weight_decay)
        best_epoch, best_measure, train_loss_ls, val_acc_ls = train_modelcv(learn_rate=learn_rate,
                                                                                                    dataloader_cvtrain=loadertr, 
                                                                                                    dataloader_cvtest=loadervl,  
                                                                                                    model=model,  
                                                                                                    criterion=criterion, 
                                                                                                    optimizer=optimizer, 
                                                                                                    scheduler=None, 
                                                                                                    num_epochs=args.num_epoch, 
                                                                                                    device=device)
        best_measure_ls.append(best_measure)
        print("Training completed for learn rate = {}.\nBest epoch: {} \nBest performance: {}".format(learn_rate,best_epoch, best_measure))
        print('-' * 10)
        print()

        ## save train log
        with open(os.path.join(args.saved_pkl_dir,'train_loss_{}.pkl'.format(str(learn_rate)[2:])), 'wb') as f:
            pickle.dump(train_loss_ls, f)
        with open(os.path.join(args.saved_pkl_dir,'val_acc_{}.pkl'.format(str(learn_rate)[2:])), 'wb') as f:
            pickle.dump(val_acc_ls, f)

        ## plot train graphs
        epochs = list(range(len(train_loss_ls)))

        fig = plt.figure()
        fig.suptitle(title)
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss_ls, '.-')
        plt.ylabel('Losses')

        plt.subplot(2, 1, 2)
        plt.plot(epochs, val_acc_ls, '.-')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Average Precision')
        
        plt.savefig(os.path.join(args.saved_img_dir,"train_graphs_{}".format(str(learn_rate)[2:])), bbox_inches='tight')
    
    ### model selection results
    print("Best accuracies: {}".format(best_measure_ls))
    print("Best learn rate: {}".format(LEARN_RATES[np.argmax(best_measure_ls)]))

### produce results based on chosen model
def results(device, loadervl, validset): 
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512,20)
    # load trained weights
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(args.saved_model_dir, "model_best_{}.pt".format(str(args.best_learn_rate)[2:]))))
    else:
        model.load_state_dict(torch.load(os.path.join(args.saved_model_dir, "model_best_{}.pt".format(str(args.best_learn_rate))[2:])),map_location=torch.device('cpu'))
    model.to(device)
    
    # ## Get class-wise average precision and mean average precision
    # class_precision, ave_precision = utils.evaluate(model, loadervl, device)
    # print("Class-wise average precision")
    # print('-' * 10)    
    # for i in range(len(class_precision)):
    #     print("{}: {}".format(validset.list_image_sets()[i],class_precision[i]))
    # print("\nMean average precision: {}".format(ave_precision))

    ## Get tail accuracy
    # indices of top and bottom 50 images for each class, size = (50,20)
    idx_high, idx_low = utils.top_50_imgs(model,loadervl,device)
    # dataset to print the images
    validset2=data.PascalVOC(args.data_dir,'val',transforms.Compose([ 
                                    transforms.CenterCrop(280),
                                    transforms.ToTensor()
                                    ]))
    #loader to calculate tail accuracy
    loadervl_tail = torch.utils.data.DataLoader(validset,batch_size=args.test_batch,shuffle=False,
                                            sampler=torch.utils.data.SubsetRandomSampler(idx_high.flatten().tolist()))
                                

    t_ls, class_wise, avg = utils.tailacc(model,loadervl_tail,0.5,device) # change t value
    # print('Tail accuracy',tail_acc)
    print("Class-wise tail accuracy shape:")
    print(class_wise.numpy().shape)
    
    plt.figure()
    plt.plot(t_ls,avg)
    plt.ylabel('Tail accuracy')
    plt.xlabel('t')
    plt.savefig(os.path.join(args.saved_img_dir,"tail_acc_graph"), bbox_inches='tight')


    for i in random.sample(range(20), 5): # 5 random classes out of 20
        class_name = validset2.list_image_sets()[i]
        plt.figure()
        fig_title = class_name+"_topbottom5"
        plot_title = class_name+": top and bottom 5"
        plt.suptitle(plot_title)
        time.sleep(0.5)
        for i,j in enumerate(idx_high[:5,i]): # iterate through top 5 highest scoring images
            plt.subplot(2,5,i+1)
            plt.axis('off')
            plt.imshow(np.transpose(validset2[j][0].numpy(),(1,2,0)))
            time.sleep(0.5)

        for i,j in enumerate(idx_low[:5,i]): # iterate through top 5 lowest scoring images
            plt.subplot(2,5,i+6)
            plt.axis('off')
            plt.imshow(np.transpose(validset2[j][0].numpy(),(1,2,0)))
            time.sleep(0.5)
        plt.savefig(os.path.join(args.saved_img_dir,fig_title), bbox_inches='tight')

def main():
    ## check GPU and set seed
    print("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.random_seed)
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
                                    transforms.Normalize(MEAN, STD)
                                    ])
    trainset = data.PascalVOC(args.data_dir,'train',transform)
    validset = data.PascalVOC(args.data_dir,'val',transform)
    
    loadertr = torch.utils.data.DataLoader(trainset,batch_size=args.train_batch,shuffle=True)
    loadervl = torch.utils.data.DataLoader(validset,batch_size=args.test_batch,shuffle=False)
    
    ## run train/results
    if args.run == 'train':
        train(device, loadertr, loadervl)
        print("\nFinished training.")
    if args.run == 'results':
        results(device, loadervl, validset)
        print("\nFinished producing results.")

if __name__=='__main__':
    args = parser.arg_parse()
    main()
    
    
    