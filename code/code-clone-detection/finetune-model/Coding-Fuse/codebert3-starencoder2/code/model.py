# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        #x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder_1, encoder_2,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder_1 = encoder_1 # starencoder
        self.encoder_2 = encoder_2  #codebert-base
        self.config=config 
        self.tokenizer=tokenizer 

        # 为每个模型创建独立的分类器
        self.classifier_1 = RobertaClassificationHead(config)
        self.classifier_2 = RobertaClassificationHead(config)
        self.args=args 

    
        
    def forward(self, input_ids_1=None,input_ids_2=None,labels=None): 
        input_ids_1=input_ids_1.view(-1,self.args.block_size) 
        input_ids_2=input_ids_2.view(-1,self.args.block_size) 
        outputs_1 = self.encoder_1(input_ids= input_ids_1,attention_mask=input_ids_1.ne(49152))[0] # starencoder
        outputs_2 = self.encoder_2(input_ids= input_ids_2,attention_mask=input_ids_2.ne(1))[0] #codebert-base
        output_1 = outputs_1[:, 0, :]  #隐藏层的第一个标记对应的向量  
        output_2 = outputs_2[:, 0, :]  #隐藏层的第一个标记对应的向量 


        # 两个分类头的 logits
        logits_1 = self.classifier_1(output_1)
        logits_2 = self.classifier_2(output_2)

        prob_1 = F.softmax(logits_1, dim=-1)
        prob_2 = F.softmax(logits_2, dim=-1)

        if labels is not None:
            # 分别计算两个分类头的 loss
            loss_fct = CrossEntropyLoss()
            loss_1 = loss_fct(logits_1, labels)
            loss_2 = loss_fct(logits_2, labels)

            # 返回平均损失和融合的概率
            fused_prob = (prob_1 + prob_2) / 2
            return (loss_1 + loss_2) / 2, fused_prob
        else:
            # 推理阶段返回融合的概率
            fused_prob = (prob_1 + prob_2) / 2
            return fused_prob
      
        
 
        


