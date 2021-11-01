import logging
logging.disable(logging.CRITICAL)
import torch
import pandas as pd
from transformers import BertTokenizer
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, AlbertTokenizerFast, RobertaTokenizerFast
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

import sys
import numpy as np
from collections import defaultdict, Counter
from functools import partial

import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def open_data(table):
    """
        Opens connection to HuLM DB and generates a pd.DataFrame
        from selected tables

        note: Currently only supports small facebook data
    """

    myDB = URL(drivername='mysql', host='localhost',
                database='HuLM', query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})
    engine = create_engine(myDB, encoding='latin1')
    conn = engine.connect()
    
    print('Fetching data...')
    
    select = conn.execute('select user_id, message_id, message, updated_time from ' + table + ' order by user_id, updated_time')
    df = pd.DataFrame(select.fetchall()) 
    df.columns = select.keys()
    
    df = df[df.message.notnull()]

    print(f'data load from {table} complete')
   
    conn.close()
    return df

# add 'PAD' message to all users with < msg_seq_len and then 
# check after tokenize if all input_token_type = 0, set attn_mask=0 for msg
def transform_usr_data(df, max_seq_len, msg_seq_len=5, bert_type='distilbert-base-uncased'):
    start_time = time.time()
    print('START TOKENIZATION...')
    
    if bert_type == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(bert_type)
    else: # distil-roberta version
        tokenizer = RobertaTokenizerFast.from_pretrained('distilroberta-base')

    def tokenize(data):
        return tokenizer.encode_plus(data, add_special_tokens=True, max_length=max_seq_len, pad_to_max_length=True, truncation_strategy='longest_first')

    all_tokens = tokenizer.batch_encode_plus(df['message'].values.tolist(), add_special_tokens=True, max_length=max_seq_len, pad_to_max_length=True, truncation_strategy='longest_first')
    print('BATCH_ENCODE  %s ----' % (time.time() - start_time))
    df['input_ids'] = all_tokens['input_ids']
    df['attention_mask'] = all_tokens['attention_mask']
    
    df_np = df.to_numpy()
    print("--- end HF tokenizer %s ---" % (time.time() - start_time))
    
    # gather messages by usr_id
    msgs_by_usr = defaultdict(list)
    attn_mask_by_usr = defaultdict(list)
    usr_list = defaultdict(list)
    for usr_msg in df_np:
        #usr_id, msg_id, msg, timestamp, tokenized = usr_msg
        usr_id, msg_id, msg, timestamp, input_ids, att_mask = usr_msg
        usr_id = str(usr_id)

        if len(msgs_by_usr[usr_id]) < msg_seq_len:
            msgs_by_usr[usr_id].append(input_ids)
            attn_mask_by_usr[usr_id].append(att_mask)
            usr_list[usr_id].append(int(usr_id))
        else: # treat usrs with more than max msgs as separate users
            for num in range(1,100):
                alt_id = usr_id + '_' + str(num)
                # alternate ID for user is tracked up to max if exists
                if alt_id in msgs_by_usr.keys() and len(msgs_by_usr[alt_id]) < msg_seq_len:
                    msgs_by_usr[alt_id].append(input_ids)
                    attn_mask_by_usr[alt_id].append(att_mask)
                    usr_list[alt_id].append(int(alt_id))
                    break
                elif alt_id not in msgs_by_usr.keys(): # add alternate ID for user
                    msgs_by_usr[alt_id].append(input_ids)
                    attn_mask_by_usr[alt_id].append(att_mask)
                    usr_list[alt_id].append(int(alt_id))
                    break # only create the 1 proxy user we need, don't loop all
    
    # ensure all "users" have padded sequence to msg_seq_len so tensor is not jagged
    for k,v in msgs_by_usr.copy().items():
        if len(v) < msg_seq_len:
            missing_amt = msg_seq_len - len(msgs_by_usr[k])
            # get most recent messages if usr is over the max
            # case wehre they are under is not handled as max should always equal min user msgs
            if '_' in k:
                orig_id, alt_ver = k.split('_')
                if int(alt_ver) > 1:
                    recent_id = orig_id + '_' + str(int(alt_ver) - 1)
                else:
                    recent_id = orig_id
                
                msgs_by_usr[k] = msgs_by_usr[recent_id][-missing_amt:] + msgs_by_usr[k]
                attn_mask_by_usr[k] = attn_mask_by_usr[recent_id][-missing_amt:] + attn_mask_by_usr[k]
                usr_list[k] = missing_amt*[int(k)] + usr_list[k] 
            else:
                for _ in range(missing_amt):
                    fake_msg = '[PAD]'
                    tokenized = tokenize(fake_msg)
                    tokenized['attention_mask'] = np.zeros(max_seq_len)
                    msgs_by_usr[k].append(input_ids)
                    attn_mask_by_usr[k].append(att_mask)
                    usr_list[k].append(int(k))

    attn_mask_by_usr = np.array(list(attn_mask_by_usr.values()))
    msgs_by_usr = np.array(list(msgs_by_usr.values()))
    usr_ids = np.array(list(usr_list.values()))
    
    print('attnusr: ', attn_mask_by_usr.shape)
    print('msgusr: ', msgs_by_usr.shape)
    print('usrids: ', usr_ids.shape)
    
    msgs_by_usr = torch.tensor(msgs_by_usr)
    attn_mask_by_usr = torch.tensor(attn_mask_by_usr, dtype=torch.float32)
    usr_ids = torch.tensor(usr_ids)

    print("--- total %s seconds ---" % (time.time() - start_time))
    return msgs_by_usr, attn_mask_by_usr, usr_ids #, usr_counts


def dataloader(table, max_seq_len, batch_size, msg_seq_len=20, use_ddp=False):
    df = open_data(table)
    padded,attn_mask, usr_id = transform_usr_data(df, max_seq_len, msg_seq_len)

    dataset = TensorDataset(padded, attn_mask, usr_id) # no labels

    if use_ddp:
        train_sampler = DistributedSampler(dataset)
    else:
        train_sampler = None

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler, num_workers=-1)

