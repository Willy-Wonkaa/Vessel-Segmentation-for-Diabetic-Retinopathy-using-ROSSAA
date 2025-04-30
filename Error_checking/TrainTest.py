import os 
import sys 
import math 

import copy 

import numpy as np 


from tqdm import tqdm 

import torch 
import torch.nn as nn 
import torch.nn.functional as F  


from Binarymetric import BinaryMetric
from utils import Save_Model 

def train_model(model, train_loader, optimizer, losses, device=None):
    
    model.train() 
    iou_score = 0.0 
    dice_score  = 0.0 
    preci_score = 0.0 
    speci_score = 0.0 
    recall_score = 0.0 
    accuracy  = 0.0 
    count = 0 
    Avg_losses = 0.0

    metric = BinaryMetric()
    
    for _, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False): 
        x =  x.to(device).float() 
        y = y.to(device).long()
        
        optimizer.zero_grad() 
        out = model(x)
        Dloss = losses[0](out, y)  
        Floss = losses[1](out, y) 
        #print(f'DIce loss {Dloss} and focal loss {Floss}') 
        #Dloss = losses(out, y)
        #Floss = losses(out, y)
        total_loss  = Dloss + Floss  
        #total_loss  = Floss 
        total_loss.backward() 
        optimizer.step() 

        Avg_losses += total_loss.item()  *x.size(0)

        
        acc, dice, iou, precision, specificity, recall  = metric(y_pred=out, y_true=y)
    
        accuracy += acc *x.size(0)
        dice_score += dice *x.size(0) 
        iou_score += iou *x.size(0)  
        preci_score += precision *x.size(0) 
        speci_score += specificity *x.size(0) 
        recall_score += recall *x.size(0) 
        count += x.size(0)
    
    Avg_losses = Avg_losses / count 
    accuracy = accuracy /count 
    dice_score  = dice_score / count 
    iou_score  = iou_score /count 
    preci_score  = preci_score / count 
    speci_score = speci_score / count 
    recall_score = recall_score/ count
    print(f'Train: losses { Avg_losses}  dice score {dice_score}  iou score {iou_score}\
          precision score {preci_score}  specificity { speci_score} recall {recall_score}') 
    
    return model

def test_model(model, test_loader, losses, device=None):
    model.eval() 
    Avg_losses = 0.0 
    iou_score = 0.0 
    dice_score  = 0.0 
    preci_score = 0.0 
    speci_score = 0.0 
    recall_score = 0.0 
    accuracy  = 0.0
    count = 0 

    metric = BinaryMetric()
    for _, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader), leave=False): 
        x = x.to(device).float()
        y = y.to(device).long() 

        with torch.no_grad():
            out = model(x) 

        Dloss = losses[0](out, y) 
        Floss = losses[1](out, y) 

        total_loss  = Dloss + Floss 
        #total_loss = Dloss
        Avg_losses += total_loss.item() *x.size(0) 

        acc, dice, iou, precision, specificity, recall  = metric(y_pred=out, y_true=y)
    
        accuracy += acc *x.size(0)
        dice_score += dice *x.size(0) 
        iou_score += iou *x.size(0)  
        preci_score += precision *x.size(0) 
        speci_score += specificity *x.size(0) 
        recall_score += recall *x.size(0) 
        count += x.size(0)

    Avg_losses = Avg_losses / count 
    accuracy = accuracy /count 
    dice_score  = dice_score / count 
    iou_score  = iou_score /count 
    preci_score  = preci_score / count 
    speci_score = speci_score / count 
    recall_score = recall_score/ count
    print(f'Test: losses {Avg_losses} dice score {dice_score} iou score { iou_score}  \
          accuracy {accuracy} SE {recall_score}  SP { speci_score} ') 
    return dice_score

def train_test(model, train_loader, test_loader, optimizer, losses, device=None, num_epochs=10, maskfolder=None,):
    best_modelwts = copy.deepcopy(model) 
    best_score = 0.0 

    for epoch in range(num_epochs):
        print(epoch)
        model = train_model(model, train_loader=train_loader, optimizer=optimizer, losses=losses, device=device) 

        dice_score = test_model(model, test_loader, losses, device=device)
        best_epoch = 0 

        if dice_score >  best_score:
            best_modelwts = copy.deepcopy(model) 
            best_score = dice_score 
            best_epoch = epoch 
        print('\n\n')

    Save_Model(model=best_modelwts, optimizer=optimizer, epoch=best_epoch, maskfolder=maskfolder, 
               dice_Score=best_score)