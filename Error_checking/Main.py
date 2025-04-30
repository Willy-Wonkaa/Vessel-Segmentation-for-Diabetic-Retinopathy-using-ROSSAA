import os 
import sys 
import math 

import numpy as np

from tqdm import tqdm 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import torch.utils.data as data 

from Loss.dice import DiceLoss 
from Loss.focal import FocalLoss

# from Model.utils import UNet 
# from Model.RESUnet import ResUNeT
# from Model_new.Model import UNeT

from Updated_Model.Model import UNeT

from dataset_module import OCTADataset
from utils import Data_Augmentation, LabelGeneration, Save_Model, Load_model 
from TrainTest import train_test

def main(lr=None, num_classes=None, in_ch=None, batch_size=None, activation=None, preprocessing =None, device = None, maskfolder= None,
          num_epochs=None): 
    train_transforms = Data_Augmentation(dataType='train') 
    valid_transform  = Data_Augmentation(dataType='valid')
    lblenc = LabelGeneration(subfolder=maskfolder)
    #print(lblenc.classes_) 

    
    train_dataset = OCTADataset(train=True, lblenc=lblenc, augmen=train_transforms,)
    valid_dataset = OCTADataset(train=False, lblenc=lblenc, augmen=valid_transform, )
    print(f'length of train dataset {len(train_dataset)} and valid dataset {len(valid_dataset)}')
    #print(train_dataset.data[:10], valid_dataset.data[:10], sep='\n\n')


    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2) 
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2) 
    print(f'length of train loader {len(train_loader)} and valid loader {len(valid_loader)}')
    

    #model = UNet(in_channels=in_ch, num_classes=num_classes).to(device) 
    #model = ResUNeT(in_channels=in_ch, num_classes=num_classes).to(device)
    model = UNeT(in_channels=in_ch, num_layers=3, num_classes=num_classes, attn=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    dice_loss = DiceLoss()#, classes=np.array([0.25, 0.75]))
    focal_loss = FocalLoss()
    #cosine_loss = CosineLoss(exp=2)
    losses = [dice_loss, focal_loss] 
    #losses = cosine_loss
    train_test(model=model, train_loader = train_loader, test_loader = valid_loader, optimizer=optimizer, 
                                    losses = losses, device = device, num_epochs = num_epochs, 
                                    maskfolder=maskfolder)

    model, optimizer, epoch, score = Load_model(model=model, optimizer=optimizer, maskfolder=maskfolder,) 
    print(epoch, score) 

if __name__ == '__main__':
    os.system('clear') 
    lr = 1e-4
    
    in_ch = 1
    num_classes = 1
    batch_size = 16
    activation = 'sigmoid'
    maskfolder = os.path.join('train','label')
    num_epochs = 500

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    #print(device, torch.cuda.is_available(), torch.cuda.current_device(), sep='\n')  
    
    main(lr=lr, num_classes = num_classes, batch_size=batch_size, activation=activation, in_ch= in_ch, maskfolder= maskfolder,
                     device=device,  num_epochs=num_epochs)
    
