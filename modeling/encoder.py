"""
    Defines the MeLT encoder which leverages building blocks from encoder_layers.py
"""

import math
import torch.nn as nn
import torch

from modeling.attn import MultiHeadedAttn
from modeling.neural import PositionalFeedForward, PositionalEncoding, batch_by_usr, batch_by_usr_no_mask, metrics, calc_avg_preds
from modeling.encoder_layers import TransformerEncoderLayer
from modeling.datahandler import dataloader
from transformers import DistilBertModel, DistilBertConfig
from transformers import AutoModel, AutoConfig
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import numpy as np

# debugging
from pprint import pprint
import sys

# tuning
import optuna

class MeLT(pl.LightningModule):
    
    def __init__(self, hparams, trial=None):
        super(MeLT, self).__init__()
        self.hparams = hparams
        self.seed = self.hparams.seed
        self.mask_rate = self.hparams.mask_rate
        self.num_trans_layers = self.hparams.tlayers
        self.max_msg_train = self.hparams.max_msg_seq
        self.max_msg_dev = self.hparams.max_msg_seq_dev

        self.extract_layer = self.hparams.extract_layer 
        self.load_bert = self.hparams.load_bert
        
        self.freeze_bert = True # always freeze during PT, FT scrpt will toggle after importing model
        
        self.pretraining = self.hparams.pretrain

        self.model_dim = self.hparams.dmodel
        self.ff_dim = self.hparams.dff
        self.inter_heads = self.hparams.num_heads

        self.dropout = self.hparams.dropout
        self.learn_rate = self.hparams.lr
        self.batch_size = self.hparams.bs
        self.epochs = self.hparams.epochs
        self.max_seq_len = self.hparams.max_seq_len
        self.emb_size = self.hparams.embed_dim 
        
        # params to tune
        if trial:
            self.hparams.lr = trial.suggest_loguniform('lr', low=5e-4,high=4e-1)
            self.hparams.reg = trial.suggest_uniform('reg', low=1e-4, high=1)


        if self.load_bert:
            if self.bert_model == 'distilbert-base-uncased':
                self.bert_layers = DistilBertModel(DistilBertConfig(n_layers=6))
            else:
                self.bert_layers = AutoModel.from_config(AutoConfig.from_pretrained('distilroberta-base'))
         
        # freeze bert layers
        if self.freeze_bert:
            for param in self.bert_layers.parameters():
                param.requires_grad = False

        self.trans_layers = nn.ModuleList([TransformerEncoderLayer(self.model_dim, self.inter_heads, self.ff_dim, self.dropout) for i in range(self.num_trans_layers)])
        self.pos_emb = PositionalEncoding(self.dropout, self.emb_size, max_len=512)
        self.layer_norm = nn.LayerNorm(self.model_dim, eps=1e-6)
        self.dense = nn.Linear(self.emb_size, self.emb_size)

        self.loss_func = nn.MSELoss(reduction='mean')
        
        # define mask and pad vectors for grouping function
        self.mask_vec = nn.Parameter(torch.randn(self.emb_size))
        self.pad_vec = nn.Parameter(torch.randn(self.emb_size))
   
    def forward(self, inputs, attn_mask, usr_id, max_msg_len):
        """
            Forward pass of the MeLT
        """

        if self.load_bert:
            # extract BERT emb for each message(batch_size=num_msg)
            inputs = torch.reshape(inputs,(inputs.shape[0]*inputs.shape[1], inputs.shape[2]))
            msg_attn_mask = attn_mask.reshape(attn_mask.shape[0]*attn_mask.shape[1], attn_mask.shape[2])
            bert_emb = self.bert_layers(inputs, msg_attn_mask)[0] 
        else:
            bert_emb = inputs # assume someone is passing the word embeds with attn mask

        # generate msg representations and group by user_id
        pooler = (bert_emb*msg_attn_mask.view(bert_emb.shape[0], bert_emb.shape[1], -1).expand(-1,-1,768))
        avg_msg_reps = torch.sum(pooler,dim=1)/torch.sum(msg_attn_mask, dim=1).view(-1,1).expand(-1,768)
        valid_msgs = ~(avg_msg_reps != avg_msg_reps) # detects nans from avging over messages with entirely 0 attn mask
        avg_msg_reps[avg_msg_reps != avg_msg_reps] = 0 # remove nans
        
        if self.pretraining:
            with torch.no_grad(): # just grouping and normalizing, don't need to calc grads
                # normalize
                mean = torch.sum(avg_msg_reps * valid_msgs.float(), axis=0) / torch.sum(valid_msgs,axis=0)
                std = torch.sqrt(torch.mean(torch.pow(torch.abs(avg_msg_reps[valid_msgs].view(-1,self.emb_size) - mean),2), axis=0))
                avg_msg_reps = (avg_msg_reps - mean) / std
                avg_msg_reps[~valid_msgs] = 0 # make fake msgs easier to filter
                
                usr_batch = batch_by_usr(avg_msg_reps, usr_id, self.pad_vec, self.mask_vec, max_msg_len, self.mask_rate)
                new_batch, new_attn_mask, (masked_msgs, mask_unmasked), (labels, unmasked_labels), ord_uid = usr_batch
                            
                avg_preds = calc_avg_preds(new_batch, masked_msgs, new_attn_mask, mask_unmasked)

            # add in positional encodings
            data = self.pos_emb(new_batch)

            # pass sequence of message vectors to transformer layers
            for trans_layer in self.van_layers:
                data = trans_layer(data, new_attn_mask)
        
            # get predictions and calculate loss
            output = self.dense(self.layer_norm(data))
            preds = output[masked_msgs]
            loss = self.loss_func(preds, labels)
            
            # filter out preds/labels for messages that were "masked" but 10% chance of being untouched
            preds2 = output[~mask_unmasked & masked_msgs]
            labels2 = labels[~unmasked_labels]

            with torch.no_grad(): # just metric comparisons
                avg_loss = self.loss_func(avg_preds, labels2)
                       
            return loss, preds2, labels2, (avg_preds, avg_loss), (output, ord_uid, attn_mask), (masked_msgs, mask_unmasked)
        else:
            new_batch, attn_mask, ordered_uid = batch_by_usr_no_mask(avg_msg_reps, usr_id, self.pad_vec, max_num_msgs=max_msg_len)
            
            # weird bug where tenors weren't auto put to GPU
            new_batch = new_batch.type(torch.FloatTensor).to(bert_emb.device)
            attn_mask = attn_mask.to(bert_emb.device)
            
            data = self.pos_emb(new_batch)
            
            for idx,trans_layer in enumerate(self.van_layers):
                data = trans_layer(data, attn_mask)
                if self.extract_layer == idx:
                    break

            data = self.layer_norm(data)
            return (data, ordered_uid, attn_mask, new_batch)

    def configure_optimizers(self):
        """
            Defines otpimizers
        """

        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.reg)
        return opt

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, opt_closure):
        """
            Set warmup step scheduler for pre-training routine

            LR is linearly scaled for the first 2000 steps. 
        """
        if self.pretraining:
            if self.trainer.global_step < 2000:
                lr_scale = min(1., float(self.trainer.global_step + 1) / 2000.)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.hparams.lr

            optimizer.step()
            optimizer.zero_grad()

    def training_step(self, batch, batch_idx):
        """
            Called inside Lightnings train.fit()
        """
        input_ids, attn_mask, usr_id = batch[0], batch[1], batch[2]
        loss, _, _, (apreds, aloss), _, _ = self(input_ids, attn_mask, usr_id, self.max_msg_train)
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)
            aloss = aloss.unsqueeze(0)
        
        # tensorboard logger
        tb_logs = {'train_loss': loss, 'avg_loss': aloss}

        return {'loss': loss, 'avg_loss': aloss, 'log': tb_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """
            Called for each batch in validation set
        """
        input_ids, attn_mask, usr_id = batch[0], batch[1], batch[2]
        loss_names = ['seen_user', 'unseen_user']
        
        loss, output, labels, (avg_preds, avg_loss), _, _ = self(input_ids, attn_mask, usr_id, self.max_msg_dev)
        other_data = {'apred': avg_preds, 'aloss': avg_loss}
        scores_dict = metrics(loss, output, loss_names[dataloader_idx], labels, other_data)
        
        # in DP mode make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            scores = [score.unsqueeze(0) for score in scores_dict.values()]
            scores_dict = {key: value for key,value in zip(scores_dict.keys(), scores)}
            scores_dict.pop('seen_usr', None) # seen_usr is moved to val_loss
        
        return scores_dict

    def validation_epoch_end(self, outputs):
        """
            Gathers results at end of validation loop
        """
        tqdm_dict = {}
        for dataset in outputs:
            num_batches = 0
            for metric_name in dataset[0].keys():
                total = 0

                for output in dataset:
                    val = output[metric_name]
                    # average across each GPU per batch
                    if self.trainer.use_dp or self.trainer.use_ddp2:
                        val = torch.mean(val)
                    total += val

                tqdm_dict[metric_name] = total/len(dataset) # avg over all batches

        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'seen_user': tqdm_dict['seen_user'], 'unseen_user': tqdm_dict['unseen_user']}
        return result

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def train_dataloader(self):
        """
            Train dataset of tweets
        """
        return dataloader_v2('twitter_train', self.hparams.max_seq_len, self.hparams.bs, self.hparams.max_msg_seq)
    
    def val_dataloader(self):
        """
            validation dataset(s). Secondary FB dataset was used to explore domain transfer, not used in final publication
        """
        d1 = dataloader_v2('twitter_dev', self.hparams.max_seq_len, self.hparams.bs, self.hparams.max_msg_seq_dev)
        d2 = dataloader_v2('fb_dev', self.hparams.max_seq_len, self.hparams.bs, self.hparams.max_msg_seq_dev)
        return [d1,d2]

    def test_dataloader(self):
        """
            test dataset(s). Secondary FB dataset was used to explore domain transfer, not used in final publication
        """
        d1 = dataloader_v2('twitter_test', self.hparams.max_seq_len, self.hparams.bs, self.hparams.max_msg_seq_dev)
        d2 = dataloader_v2('fb_test', self.hparams.max_seq_len, self.hparams.bs, self.hparams.max_msg_seq_dev)
        return [d1,d2]
        


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        """
            parameters defined here are available through self.model_params
        """

        parser = HyperOptArgumentParser(parents=[parent_parser])
        
        # BERT related parameters
        parser.add_argument('--bert_model', default='distilbert-base-uncased', type=str, help="config bert base or large")

        # encoder params
        parser.add_argument('--seed', default=1337, type=int, help="random seed")
        parser.add_argument('--tlayers', default=2, type=int, help="num transformer layers on top of BERT")
        parser.add_argument('--epochs', default=5, type=int, help="total num of epochs")
        parser.add_argument('--dropout', default=0.1, type=float, help="droprate throughout model")
        parser.add_argument('--lr', default=1e-5, type=float, help="intiial learning rate")
        parser.add_argument('--reg', default=0.1, type=float, help="L2 reg applied to optimizer")
        parser.add_argument('--bs', default=10, type=int, help="minibatch size") 
        parser.add_argument('--max_seq_len', default=50,type=int, help="max number of input tokens for model")
        parser.add_argument('--dmodel', default=768, type=int, help="dimension size of transformer representations")
        parser.add_argument('--dff', default=2048, type=int, help="dimension size of FFNN layers")
        parser.add_argument('--num_heads', default=8, type=int, help="num of inter attn heads")
        parser.add_argument('--mask_rate', default=15, type=int, help="% of msgs to mask for upper transformer layer")
        parser.add_argument('--max_msg_seq', default=40, type=int, help="maximum message vectors to append to the seq")
        parser.add_argument('--max_msg_seq_dev',default=20, type=int, help="maximum message vectors to append to the seq in dev/test" )
        parser.add_argument('--embed_dim', default=768, type=int, help="Size of embedding dimension for model")
        parser.add_argument('--pretrain', default=True, type=bool, help="Toggle pretraining objective")
        parser.add_argument('--extract_layer', default=1, type=int, help="Select layer to extract embeddings from. Count starts at 0")
        parser.add_argument('--load_bert', default=True, type=bool, help="Toggle loading in HF layers")
        parser.add_argument('--apply_loss', default=True, type=bool, help="Calculate loss if running as single module")
        parser.add_argument('--freeze_bert', default=True, type=bool, help="Freeze underlying word level model")

        return parser

