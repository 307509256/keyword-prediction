#!/usr/bin/env python 

import re
import collections 
import numpy as np

from pymongo import MongoClient 
from collections import namedtuple

hyperparams = namedtuple('hyperparams',
    'mode, lr, batch_size, hidden_size, embed_size, '
    'enc_timesteps, dec_timesteps, num_kw, enc_layers')

db = MongoClient()['nlp']

START_TOKEN  = '<s>'
END_TOKEN = '</s>'
UNK_TOKEN = '<UNK>'
SEP_TOKEN = '<SEP>'
PAD_TOKEN = '<PAD>'

class Vocab(object):

    def __init__(self, dataset):
        self._build_vocab(dataset)
        self._kw_freqs = self._collect_keyword_freqs(dataset)

    @property
    def kw_freqs(self):
        return self._kw_freqs

    def get_word(self, id):
        return self._id_to_word[id] 

    def get_words(self, ids):
        return [self.get_word(id) for id in ids]

    def get_word_id(self, word):
        return self._word_to_id[word] 

    def get_word_ids(self, sequence):
        return [self.get_word_id(w) for w in sequence]
        
    def _build_vocab(self, dataset):
        '''
        Build vocabulary

        '''
        words = [PAD_TOKEN, START_TOKEN, END_TOKEN, SEP_TOKEN, ]
        for d in db[dataset].find():
            words += d['title'].split()
            words += d['abstract'].split()
            for kw in d['keywords']:
                words += kw.split()
        words = sorted(collections.Counter(words).items(),
                key=lambda t: (-t[1], t[0]))
        words, _ = zip(*words)
        ids = xrange(len(words))
        self._word_to_id = dict(zip(words, ids))
        self._id_to_word = dict(zip(ids, words))

    def _collect_keyword_freqs(self, dataset):
        keywords = []
        for d in db[dataset].find():
            keywords += d['keywords']
        return collections.Counter(keywords)
    
    def __len__(self):
        return len(self._word_to_id)

class DataManager(object):

    def __init__(self, vocab, cfg):
        '''
        
        '''
        self._cfg = cfg
        self._vocab = vocab
        self._data = self._transform_data()

    @property
    def vocab(self):
        return self._vocab
    
    def data_iterator(self):
        '''
        
        '''
        data = self._data
        batch_size = self._cfg.batch_size
        mode = self._cfg.mode
        enc_timesteps = self._cfg.enc_timesteps 
        dec_timesteps = self._cfg.dec_timesteps 

        idx = 0
        #for idx in xrange(0, len(data), batch_size):
        while True:
            batch = data[idx: idx + batch_size]

            enc_batch = np.zeros([batch_size, enc_timesteps], dtype=np.int32)
            enc_input_lens = np.zeros([batch_size], dtype=np.int32)
            dec_batch = np.zeros([batch_size, dec_timesteps], dtype=np.int32)
            dec_output_lens = np.zeros([batch_size], dtype=np.int32)
            target_batch = np.zeros([batch_size, dec_timesteps], dtype=np.int32)
            loss_weights = np.zeros([batch_size, dec_timesteps], dtype=np.float32)
            orig_abstracts = [None] * batch_size 
            orig_keywords = [None] * batch_size

            for i, d in enumerate(batch):
                (enc_inputs, dec_inputs, targets, enc_input_len, dec_output_len, 
                    abstract, keywords) = d
                enc_batch[i, :] = enc_inputs[:]
                dec_batch[i, :] = dec_inputs[:]
                target_batch[i, :] = targets[:]
                enc_input_lens[i] = enc_input_len
                dec_output_lens[i] = dec_output_len
                orig_abstracts[i] = abstract
                orig_keywords[i] = keywords
                for j in xrange(dec_output_len):
                    loss_weights[i][j] = 1
            #if mode == 'train':
            #    idx = (idx + batch_size)%len(data)
            #else:
            idx += batch_size
            if idx >= len(data):
                break 
            yield (enc_batch, dec_batch, target_batch, enc_input_lens, dec_output_lens, 
                    loss_weights, orig_abstracts, orig_keywords)

    def _transform_data(self):
        '''
        
        '''
        data = []
        enc_timesteps = self._cfg.enc_timesteps 
        dec_timesteps = self._cfg.dec_timesteps
        num_kw = self._cfg.num_kw
        start_id = self._vocab.get_word_id(START_TOKEN)
        end_id = self._vocab.get_word_id(END_TOKEN)
        docs = [d for d in db[self._cfg.mode].find()][:1000]
        #docs = [d for d in db['train'].find()][:1000]
        if self._cfg.mode == 'eval':
            docs = docs[:500]
        #docs = [d for d in db['train'].find()][:500]
        for d in docs:
            sorted_kws = self._sort_keywords_by_freq(
                d['keywords'], num_kw, self._vocab.kw_freqs)
            text = self._vocab.get_word_ids(
                d['abstract'].split()[:enc_timesteps])
            keywords = self._vocab.get_word_ids(
                (' %s '%SEP_TOKEN).join(sorted_kws).split()[:dec_timesteps - 1])
            enc_input = text
            dec_input = [start_id] + keywords 
            targets = keywords + [end_id]
            enc_input_len = len(enc_input)
            dec_output_len = len(targets)

            if enc_input_len < enc_timesteps:
                enc_input += [end_id] * (enc_timesteps - enc_input_len)
            if len(dec_input) < dec_timesteps:
                dec_input += [end_id] * (dec_timesteps - len(dec_input)) 
            if dec_output_len < dec_timesteps:
                targets += [end_id] * (dec_timesteps - dec_output_len) 

            data.append((enc_input, dec_input, targets, 
                enc_input_len, dec_output_len, 
                d['abstract'], (' %s '%SEP_TOKEN).join(sorted_kws)))
        return data

    def _sort_keywords_by_freq(self, keywords, num_kw, kw_freqs):
        '''

        :rtype tuple
        
        '''
        keywords = sorted([(kw, kw_freqs.get(kw)) for kw in keywords],
                key=lambda x: -x[1])
        return zip(*keywords)[0]

if '__main__' == __name__:
    cfg = hyperparams(
            mode='eval',
            lr=.15,
            batch_size=1,
            enc_layers=4,
            enc_timesteps=100,
            dec_timesteps=30, 
            num_kw=3,
            hidden_size=256,
            embed_size=128,
            )
    vocab = Vocab('pubmed')
    dm = DataManager(vocab, cfg)
    import time
    gen = dm.data_iterator()
    while True:
        b = gen.next()
        print b[-1]
        time.sleep(1)

