import torch.nn as nn
import torch.nn.functional as F
import time 
import torch
import os
import numpy as np
from utils import adjust_learning_rate, get_performance
logger = open("Logger.txt","w")
logger.write("Training starts\n")

apply_softmax = torch.nn.Softmax(dim=-1)

def train_model(train_loader, model, optimizer, epoch, cv_id, LOG_DIR, args):

    model.train()
    start = time.time()
    total_loss = 0    
    acc, recall, precision, f1 = 0, 0, 0, 0
    count = 0
    optimizer.zero_grad()
    logits = []
    label = []
    for i, batch in enumerate(train_loader):# batch size in dataloader should be one
        
        src = batch['data']
        targets = batch['label'].view(-1).long() 
        if src.data.numpy().shape[1] > 1024:
            continue
        trg_input = torch.randn(src.size())
        preds = model(src, trg_input)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets)# TODO include weights in loss function in run time
        loss.backward()
        total_loss += loss.data.numpy()
        pred = (preds.view(-1, preds.size(-1)).data.numpy())
        logits.append(np.argmax(pred, axis = 1))
        label.append(targets.data.numpy())
        if i%(args.batch_size) == (args.batch_size - 1):
            optimizer.step()
            optimizer.zero_grad()
            adjust_learning_rate(optimizer, args)
            count = count + 1
            acc_, recall_, precision_, f1_ = get_performance([item for sublist in label for item in sublist], [item for sublist in logits for item in sublist], avg = 'macro')
            acc = acc + acc_
            recall = recall + recall_
            precision = precision + precision_
            f1 = f1 + f1_	
            label = []
            logits = []
    logger.write('Loss on Train data : {}'.format(total_loss/count))
    print('Loss on Train data : {}'.format(total_loss/count))
    logger.write('Train_epoch_and_cv_id  : {} & {} => Accuracy : {} recall : {} precision : {} f1_score : {}\n'.format(epoch, cv_id, acc/count, recall/count, precision/count, f1/count))
    print('Train_epoch_and_cv_id : {} & {} => Accuracy : {} recall : {} precision : {} f1_score : {}\n'.format(epoch, cv_id, acc/count, recall/count, precision/count, f1/count))
    print("Time Taken : {}:{} mins".format(int((time.time() - start)//60), int((time.time() - start)%60)))

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}/checkpoint_{}.pth'.format(LOG_DIR, epoch))



def test_model(test_loader, model, epoch, LOG_DIR, args):
    
    model.eval()
    start = time.time()
    total_loss = 0    
    count = 0
    label, logits = [], []
    acc, recall, precision, f1 = 0, 0, 0, 0
    for i, batch in enumerate(test_loader):
        
        src = batch['data']
        targets = batch['label'].view(-1).long()
        trg_input = torch.randn(src.size())   
        preds = model(src, trg_input)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets)
        total_loss += loss.data.numpy()
        pred = (preds.view(-1, preds.size(-1)).data.numpy())
        logits.append(np.argmax(pred, axis = 1))
        label.append(targets.data.numpy())
        if i%(args.batch_size) == (args.batch_size - 1):
            count = count + 1
            acc_, recall_, precision_, f1_ = get_performance([item for sublist in label for item in sublist], [item for sublist in logits for item in sublist], avg = 'macro')
            acc = acc + acc_
            recall = recall + recall_
            precision = precision + precision_
            f1 = f1 + f1_	
            label = []
            logits = []
    logger.write('Loss on Test data : {}'.format(total_loss/count))
    print('Loss on Test data : {}'.format(total_loss/count))
    logger.write('Test_epoch_and_cv_id  : {} => Accuracy : {} recall : {} precision : {} f1_score : {}\n'.format(epoch, acc/count, recall/count, precision/count, f1/count))
    print('Test_epoch_and_cv_id : {} => Accuracy : {} recall : {} precision : {} f1_score : {}\n'.format(epoch, acc/count, recall/count, precision/count, f1/count))
    print("Time Taken : {}:{} mins".format(int((time.time() - start)//60), int((time.time() - start)%60)))

