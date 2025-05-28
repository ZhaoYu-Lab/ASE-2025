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

# this is update by zy according to the config file
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model*2, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate) # original is comment ##
        self.out_proj = nn.Linear(config.d_model, 2)

    def forward(self, features, **kwargs):
        #x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)  # original is comment ##
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)  # original is comment ##
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,encoder_2,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.encoder_2 = encoder_2  #starencoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.classifier_2 = RobertaClassificationHead(config)
        self.args=args
    
        
    def forward(self, input_ids=None,input_ids_2=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        input_ids_2=input_ids_2.view(-1,self.args.block_size) 
        #print("self.tokenizer.pad_token_id", self.tokenizer.pad_token_id)##0000
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id) # pad is 0  tokens like  1 *******000000 2
        
        #outputs = self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(0))[0]
        
        model_outputs = self.encoder(input_ids= input_ids, attention_mask=attention_mask, labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = model_outputs['encoder_hidden_states'][-1]
                
        #######bos+eos########
        ##print(np.shape(hidden_states))
        ##print(hidden_states)
        ##bos => eos 互换分别得到第一个和最后一个 这一行代码创建了一个名为 eos_mask 的布尔型张量，用于指示哪些位置包含了T5的结束标记（<eos>）。这是通过检查输入 source_ids 是否等于T5配置中的结束标记ID来实现的。
        bos_mask = input_ids.eq(self.config.bos_token_id)
        ##print(np.shape(bos_mask))
        #print('self.config.eos_token_id',self.config.eos_token_id)
        if len(torch.unique(bos_mask.sum(1))) > 1:
           raise ValueError("All examples must have the same number of <eos> tokens.")

        ##这一行代码根据 eos_mask 从隐藏状态中选择包含结束标记的位置，然后将这些位置的隐藏状态合并成一个向量。最后的结果是一个二维张量，每一行对应一个输入示例，每一行中的向量是该示例的编码表示。
        outputs = hidden_states[bos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, 0, :]
        ##print(np.shape(outputs))
        ##print(outputs)
        #######bos+eos########        
        
        outputs_2 = self.encoder_2(input_ids= input_ids_2,attention_mask=input_ids_2.ne(49152))[0] #32,768 batch_size,diemention
        output_2 = outputs_2[:, 0, :]  #隐藏层的第一个标记对应的向量 

        logits=self.classifier(outputs)
        logits_2 = self.classifier_2(output_2)

        prob=F.softmax(logits)
        prob_2 = F.softmax(logits_2, dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            loss_2 = loss_fct(logits_2, labels)
            # 返回平均损失和融合的概率
            fused_prob = (prob + prob_2) / 2
            return (loss + loss_2) / 2, fused_prob
        else:
            # 推理阶段返回融合的概率
            fused_prob = (prob + prob_2) / 2
            return fused_prob
      
        
 
        


