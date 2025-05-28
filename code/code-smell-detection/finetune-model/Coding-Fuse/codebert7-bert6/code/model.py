# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

import pandas as pd
    
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        #x = torch.tanh(x)
        x = self.out_proj(x)
        return x

    
class Model(nn.Module):   
    def __init__(self, encoder,encoder_2,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.encoder_2 = encoder_2 #codebert-base
        self.config=config
        self.tokenizer=tokenizer
        self.args=args

        self.classifier=RobertaClassificationHead(config)
        self.classifier_2 = RobertaClassificationHead(config)
        
    def forward(self, input_ids=None,input_ids_2=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(0))[0]
        outputs_2=self.encoder_2(input_ids_2,attention_mask=input_ids_2.ne(1))[0]
        output = outputs[:, 0, :]  #隐藏层的第一个标记对应的向量  
        output_2 = outputs_2[:, 0, :]  #隐藏层的第一个标记对应的向量 

       # 两个分类头的 logits
        logits = self.classifier(output)
        logits_2 = self.classifier_2(output_2)

        prob=F.softmax(logits)
        prob_2=F.softmax(logits_2)
        #prob=torch.sigmoid(logits)
        #prob_2=torch.sigmoid(logits_2)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            loss_2 = loss_fct(logits_2, labels)
  
            #labels=labels.float()
            #loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            #loss_2=torch.log(prob_2[:,0]+1e-10)*labels+torch.log((1-prob_2)[:,0]+1e-10)*(1-labels)

            #loss=-loss.mean()
            #loss_2=-loss_2.mean()

            fused_prob = (prob + prob_2) / 2

            return (loss + loss_2) / 2, fused_prob
        else:
            fused_prob = (prob + prob_2) / 2
            return fused_prob 
      
        
 
