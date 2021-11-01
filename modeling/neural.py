import math
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, zscore
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys


def metrics(loss, preds, metric_name, labels=None, other_data=None):
    """
        Determines the values of any desired metrics(Perplexity, F1, etc)
        Currently returns MSE loss, Correlation, and Avg prediction loss/corr

        args:
            loss: from forward pass
            preds: from forward pass
            metric_name: Toggle between seen/unseen users
            labels: target message for prediction
            other_data: dictionary of other data to calculate metrics on
        outputs:
            results: dictionary of metrics and values
    """
    
    preds = preds.cpu().numpy().astype(float)
    labels = labels.cpu().numpy().astype(float)
    corr = calc_corr(labels, preds)
                
    corr_metric = metric_name + '_corr'
    
    # build score dictionary
    score_dict = {metric_name: loss, corr_metric: torch.tensor(corr).to(loss.device)}

    if other_data:
        apreds = other_data['apred'].cpu().numpy().astype(float)
        aloss = other_data['aloss']
        acorr = calc_corr(labels, apreds)
        score_dict[metric_name+'_acorr'] = torch.tensor(acorr).to(aloss.device)
    
    if 'seen_user' in score_dict.keys():
        score_dict['val_loss'] = score_dict['seen_user'] # need val_loss present for saving

    return score_dict

def calc_sim(labels, preds):
    """
        Calculates cosine similaritiy between predicted message vector(s) and ground truth vector(s)

        args:
            labels: sequence of message vectors selected for masking
            predS: output predictions for selected messages
        outputs:
            similaritiy: cosine similaritiy between all message predictions
    """
    labels = np.transpose(labels)
    preds = np.transpose(preds)

    return cosine_similarity(std_labels, std_preds)

def calc_corr(labels, preds):
    """
        Calculaties pearson correlation between predicted message vector(s) and ground truth vector(s)

        args:
            labels: sequence of message vectors selected for masking
            predS: output predictions for selected messages
        outputs:
            corr: average correlation across all message predictions
    """
    
    preds = (preds - np.mean(preds, axis=0, keepdims=True) ) / np.std(preds, axis=0, keepdims=True)
    labels = (labels - np.mean(preds, axis=0, keepdims=True) ) / np.std(preds, axis=0, keepdims=True)

    avg_corr_obs = []
    for i in range(labels.shape[0]):
        corr = np.mean(pearsonr(preds[i], labels[i])[0])
        avg_corr_obs.append(corr)
   
    return np.mean(avg_corr_obs)

def calc_avg_preds(usr_batches, msg_mask, attn_mask, msg_unmasked):
    """
        Calculate average message vector of user sequence ignoring masked messages. Used as baseline
        predictor

        args:
            usr_batches: sequences of user messages. shape=[num_usr, num_msgs, embed_dim]
            msg_mask: boolean tensor set to 1 where a message is selected for masking. Used to ignore message in average calculation
            attn_mask: boolean tensor set to 1 where a message is real and 0 whee pad. Used to ignore pad tokens in average calculation
            msg_unmasked: boolean tensor set to 1 where a message is selected for masking but is left alone (10% chance)
        output:
            avg_preds: collection of average predictions per user sequence. shape=[num_usr, embed_dim]
    """
    new_avg_preds = torch.tensor([])
    
    for idx, usr in enumerate(usr_batches):
        usr_masked_msgs = msg_mask[idx]
        usr_masked_unmasked_msgs = msg_unmasked[idx] # masked msgs that 10% don't actually get masked
        usr_attn_mask = attn_mask[idx]
        usr_real_msgs = []
        for i in range(usr.shape[0]):
            # if msg is a real one (non-mask and non-pad)
            if usr_masked_msgs[i].item() == False and usr_attn_mask[i].item() == True:
                msg_vec = usr[i].cpu().data.numpy()
                usr_real_msgs.append(msg_vec)

        # use avg prediction for as many masked msg in usr seq
        for _ in range(torch.sum(usr_masked_msgs.to(dtype=torch.long) - usr_masked_unmasked_msgs.to(dtype=torch.long))):
            new_avg_preds = cat_tensors(new_avg_preds, torch.Tensor(np.mean(usr_real_msgs,axis=0)))
               
    return new_avg_preds.to(usr_batches.device)


def cat_tensors(tensor1, tensor2):
    if tensor1.shape[0] == 0:
        tensor1 = tensor2.unsqueeze(0)
    else:
        tensor1 = torch.cat((tensor1, tensor2.unsqueeze(0)), 0)

    tensor1.detach()
    return tensor1

def batch_by_usr(batch, usr_id, pad_vec, mask_vec, max_num_msgs=512, mask_rate=15):
        """
            Organizes message vectors by their user_id. Selects which messages to be masked and builds
            the attn_mask & other boolean sequences.

            args:
                batch: collection of message vectors shape=[msg_num, emb_dim]
                usr_id: ordered list of user_ids (assumption: matches order of msg's)
                pad_vec: encoder embedding for a pad token
                mask_vec: encoder embedding for a mask token
                max_num_msgs: hard cut off for message sequence length per user(default=512)
                mask_rate: percent of messages to mask per user
            outputs:
                msg_batch: grouped msg vectors shape==[num_user, max_num_msgs, emb_dim]
                unpadded: a boolean mask of real messages vs pad tokens
                inverse_mask: a boolean mask highlighting which messages are selected for masking
                labels: selected messages to be masked
        """
        
        # Groups msg vectors by user_id
        msg_seq_by_usr = dict()
        usr_id = usr_id.reshape(-1)

        for idx,msg_vec in enumerate(batch):
            uid = str(usr_id[idx].item())
            
            if torch.isnan(torch.sum(msg_vec)).item() == True or torch.sum(msg_vec).item() == 0:
                continue # skip the fake padded messages from batching
            try:
                msg_seq_by_usr[uid] = cat_tensors(msg_seq_by_usr[uid], msg_vec) 
            except KeyError:
                msg_seq_by_usr[uid] = torch.unsqueeze(msg_vec, 0)


        # initialize output variables
        msg_batch = torch.Tensor([])
        labels = torch.Tensor([])
        unpadded = []
        inverse = []
        unmasked = [] 
        unmasked_labels = []

        # perform masking and padding to user sequences
        for k,v in msg_seq_by_usr.items():
            usr_msgs = [] 
            usr_masks = [] # binary mask for if a message was selected for masking or not 
            usr_masks_unswapped = [] # binary mask for if a message was "masked" but not actually
            msg_seq_len = len(v)
            
            if msg_seq_len < max_num_msgs:
                pad_amt = max_num_msgs - len(v)
                upper_bound = msg_seq_len
            else:
                pad_amt = 0
                upper_bound = max_num_msgs
            

            masked_seq = torch.Tensor([]).type(torch.FloatTensor)
            
            # uniform likelihood of 0-100 for length of user's sequence
            usr_masked_idx = np.random.randint(0,101,upper_bound)
            
            # each user gets at least 1 message masked
            if not any(e < mask_rate for e in usr_masked_idx):
                usr_masked_idx[np.random.randint(0,upper_bound)] = -1

            for idx,msg in enumerate(v):
                # if user hit cap then stop adding
                if idx >= max_num_msgs:
                    break

                # apply masking to actual sequence and gen atn_masks
                # if msg is selected for mask 3 options
                # 1. just mask it (80%)
                # 2. replace with random msg (10%)
                # 3. don't mask it (10%)
                if usr_masked_idx[idx] < mask_rate:
                    usr_masks.append(1)
                    usr_masks_unswapped.append(0)
                    unmasked_labels.append(0)
                    
                    labels = cat_tensors(labels, msg) # label is always original message

                    mask_option = np.random.randint(0,101) 

                    if mask_option < 10:
                        msg = batch[np.random.randint(0,batch.shape[0])]
                    elif mask_option > 20: 
                        msg = mask_vec.clone().detach()
                    else: # 10-20 is for leaving msg as is
                        usr_masks_unswapped.pop()
                        usr_masks_unswapped.append(1)
                        unmasked_labels.pop()
                        unmasked_labels.append(1)

                else:
                    usr_masks.append(0)
                    usr_masks_unswapped.append(0)
                
                masked_seq = cat_tensors(masked_seq, msg) # masked seq gets mask, original, or random msg
                usr_msgs.append(1)

            for _ in range(pad_amt):
                masked_seq = cat_tensors(masked_seq, pad_vec.clone().detach()) 
                usr_msgs.append(0)
                usr_masks.append(0)
                usr_masks_unswapped.append(0)

            # stack inverse mask 
            inverse.append(usr_masks)
            unmasked.append(usr_masks_unswapped)
            msg_batch = cat_tensors(msg_batch, masked_seq)
            unpadded.append(usr_msgs)
        
        unpadded = torch.BoolTensor(unpadded)
        inverse = torch.BoolTensor(inverse)
        fake_masks = torch.BoolTensor(unmasked)
        unmasked_labels = torch.BoolTensor(unmasked_labels)
        
        # batch, attn_mask, masked_msgs, labels, ordered_usr_id
        return msg_batch, unpadded, (inverse, fake_masks), (labels, unmasked_labels), msg_seq_by_usr.keys()

def batch_by_usr_no_mask(batch, usr_id, pad_vec, max_num_msgs=512):
    """
        Same as batch_by_usr without the masking logic. Used when not pre-training (e.g. fine-tuning) where we do not
        want to mask since we are not doing the MLM task.
    """

    # Groups msg vectors by user_id
    msg_seq_by_usr =  dict()
    usr_id = usr_id.reshape(-1)
    
    for idx,msg_vec in enumerate(batch):
        uid = str(usr_id[idx].item())
        try:
            msg_seq_by_usr[uid] = cat_tensors(msg_seq_by_usr[uid], msg_vec) #torch.cat( (msg_seq_by_usr[uid], msg_vec.unsqueeze(0)), 0)
        except KeyError:
            msg_seq_by_usr[uid] = torch.unsqueeze(msg_vec, 0)
    
    # initialize output variables
    msg_batch = torch.Tensor([]).type(torch.FloatTensor)
    attn_mask = []

    # perform padding to user sequences
    for k,v in msg_seq_by_usr.items():
        usr_mask = [] # mask out pad vectors from real ones for attn
        msg_seq_len = len(v)
        usr_seq = torch.Tensor([])

        if msg_seq_len < max_num_msgs:
            pad_amt = max_num_msgs - len(v)
        else:
            pad_amt = 0
    
        for idx,msg in enumerate(v):
            # if user hit cap then stop adding
            if idx >= max_num_msgs:
                break

            usr_seq = cat_tensors(usr_seq, msg)
            usr_mask.append(1)
        
        for i in range(pad_amt):
            usr_seq = cat_tensors(usr_seq, pad_vec.clone().detach()) 
            usr_mask.append(0)
           
        msg_batch = cat_tensors(msg_batch, usr_seq)
        attn_mask.append(usr_mask)
    
    attn_mask = torch.BoolTensor(attn_mask)

    return msg_batch, attn_mask, msg_seq_by_usr.keys()

class PositionalEncoding(nn.Module):
    """
    Sinusoidal pos encoding for transformers

    Args:
        dropout(float): dropout rate
        dim(int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pos_embed = torch.zeros(max_len, dim)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)))
        
        pos_embed[:, 0::2] = torch.sin(position.float() * div_term)
        pos_embed[:, 1::2] = torch.cos(position.float() * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if step:
            emb = emb + self.pos_embed[:, step][:,None,:]
        else:
            emb = emb + self.pos_embed[:, :emb.size(1)]
        
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pos_embed[:, :emb.size(1)]

class PositionalFeedForward(nn.Module):
    """
    FeedForward NN as defined in Vaswani 2017

    Args:
        input_dim(int): size of input entering the NN
        ff_dim(int): size of hidden layers of NN
        dropout(float): dropout rate
    """

    def __init__(self, input_dim, ff_dim, dropout=0.1):
        super(PositionalFeedForward, self).__init__()
        self.layer1 = nn.Linear(input_dim, ff_dim)
        self.layer2 = nn.Linear(ff_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, inputs):
        layer1_out = self.drop1(self.relu(self.layer1(self.layer_norm(inputs))))
        layer2_out = self.drop2(self.layer2(layer1_out))
        residual = layer2_out + inputs
        return residual
