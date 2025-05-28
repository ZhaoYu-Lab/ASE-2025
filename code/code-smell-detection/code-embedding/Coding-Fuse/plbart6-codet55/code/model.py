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
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out_proj = nn.Linear(config.d_model, 2)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        #x = torch.tanh(x)
        x = self.out_proj(x)
        return x

    
class Model(nn.Module):   
    def __init__(self, encoder,encoder_2,config,tokenizer,tokenizer_2,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.encoder_2 = encoder_2 #codebert-base
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        self.tokenizer_2=tokenizer_2 #plbart
        self.classifier=RobertaClassificationHead(config)
        self.classifier_2 = RobertaClassificationHead(config)
        
    def forward(self, input_ids=None,input_ids_2=None,labels=None): 
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id) # pad is 1
        model_outputs = self.encoder(input_ids= input_ids, attention_mask=attention_mask, labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = model_outputs['encoder_hidden_states'][-1]

        #######bos+eos codet5 ########
        #print(np.shape(hidden_states))
        #print(hidden_states)
        #bos => eos 互换分别得到第一个和最后一个 这一行代码创建了一个名为 eos_mask 的布尔型张量，用于指示哪些位置包含了T5的结束标记（<eos>）。这是通过检查输入 source_ids 是否等于T5配置中的结束标记ID来实现的。
        bos_mask = input_ids.eq(self.encoder.config.bos_token_id)
        #print(np.shape(eos_mask))
        #print(eos_mask)
        if len(torch.unique(bos_mask.sum(1))) > 1:
           raise ValueError("All examples must have the same number of <eos> tokens.")

        #这一行代码根据 eos_mask 从隐藏状态中选择包含结束标记的位置，然后将这些位置的隐藏状态合并成一个向量。最后的结果是一个二维张量，每一行对应一个输入示例，每一行中的向量是该示例的编码表示。
        outputs = hidden_states[bos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, 0, :]
        #print(np.shape(outputs))
        #print(outputs)
        #######bos+eos codet5 ########       


        attention_mask_2=input_ids_2.ne(self.tokenizer_2.pad_token_id) # pad is 1
        model_outputs_2 = self.encoder_2(input_ids= input_ids_2, attention_mask=attention_mask_2, labels=input_ids, decoder_attention_mask=attention_mask_2, output_hidden_states=True)
        hidden_states_2 = model_outputs_2['encoder_hidden_states'][-1]

        #######bos+eos plbart ########
        #print(np.shape(hidden_states))
        #print(hidden_states)
        #bos => eos 互换分别得到第一个和最后一个 这一行代码创建了一个名为 eos_mask 的布尔型张量，用于指示哪些位置包含了T5的结束标记（<eos>）。这是通过检查输入 source_ids 是否等于T5配置中的结束标记ID来实现的。
        bos_mask_2 = input_ids_2.eq(self.encoder_2.config.bos_token_id)
        #print(np.shape(eos_mask))
        #print(eos_mask)
        if len(torch.unique(bos_mask_2.sum(1))) > 1:
           raise ValueError("All examples must have the same number of <eos> tokens.")

        #这一行代码根据 eos_mask 从隐藏状态中选择包含结束标记的位置，然后将这些位置的隐藏状态合并成一个向量。最后的结果是一个二维张量，每一行对应一个输入示例，每一行中的向量是该示例的编码表示。
        outputs_2 = hidden_states_2[bos_mask_2, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, 0, :]
        #print(np.shape(outputs))
        #print(outputs)
        #######bos+eos plbart########  

        output = outputs
        output_2 = outputs_2 #隐藏层的第一个标记对应的向量 

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
      
        
 
