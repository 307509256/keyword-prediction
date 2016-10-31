#!/usr/bin/env python 

import os
import sys
import tensorflow as tf
import numpy as np
import time

from pymongo import MongoClient
from collections import Counter, namedtuple

FLAGS = tf.app.flags.FLAGS 
tf.app.flags.DEFINE_string('mode', 'train', 'Mode')
tf.app.flags.DEFINE_integer('num_kw', 5, 
    'Number of predicting keywords')
tf.app.flags.DEFINE_string('dataset', 'pubmed',
    'Dataset to use')
tf.app.flags.DEFINE_string('log_root', '',
    'Log directory')
tf.app.flags.DEFINE_string('model_name', 'keyword', 
    'Model name')
tf.app.flags.DEFINE_integer('vocab_size', 10000,
    'Vocabulary sample size')
tf.app.flags.DEFINE_integer('keyword_size', 500,
    'Keywords sample size')
tf.app.flags.DEFINE_integer('max_steps', 100000,
    'Max steps to run')
tf.app.flags.DEFINE_integer('max_docs', 500,
    'Max input documents')

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
db = MongoClient()['nlp']

config = namedtuple('config',
    'mode, max_text_len, embed_size, hidden_size, num_layers, '
    'batch_size, learning_rate, dropout, num_kw, max_docs, '
    'max_steps')

class KeywordsPredictionModel(object):
    '''

    '''
    
    def __init__(self, vocab, keywords,  
        max_text_len, embed_size, hidden_size, num_kw,
        num_layers=2, learning_rate=.005, is_training=True):
        '''

        '''
        self._vocab = vocab 
        self._keywords = keywords
        self._max_text_len = max_text_len
        self._embed_size = embed_size 
        self._hidden_size = hidden_size 
        self._num_kw = num_kw
        self._num_layers = num_layers
        self._lr = learning_rate

        self._add_placeholders()
        self._add_model()
        self._add_cost_function()
        if is_training:
            self._train_op = self._train_op()

    def run_train_step(self, sess, text, text_len, keywords, dropout):
        '''

        '''
        _, _, cost = sess.run([self._train_op, self._global_step, self._cost], {
            self._text: text, 
            self._text_len: text_len,
            self._targets: keywords,
            self._dropout: dropout
        })
        return cost

    def run_test_step(self, sess, text, text_len, keywords):
        '''
        
        '''
        pred = sess.run([self._prediction], {
            self._text: text, 
            self._text_len: text_len,
            self._dropout: 1.
        })
        pred = pred[0]

        for i, p in enumerate(pred):
            _, topk_pred = tf.nn.top_k(p, self._num_kw)
            pred = self._keywords.get_keywords(topk_pred.eval())
            kws = self._keywords.get_keywords(keywords[i])
            sys.stdout.write('%s\n%s\n\n'%('<SEP>'.join(pred), '<SEP>'.join(kws)))
            sys.stdout.flush()
            #time.sleep(3)

    def _add_placeholders(self):
        '''

        '''
        # text, [batch_size x max_text_len]
        self._text = tf.placeholder(tf.int32, [None, self._max_text_len], name='text')
        # actual text length, [batch_size]
        self._text_len = tf.placeholder(tf.int32, [None], 'text_len')
        self._targets = tf.placeholder(tf.float32, [None, len(self._keywords)], name='keywords')
        # dropout
        self._dropout = tf.placeholder(tf.float32, name='dropout')

    def _add_model(self):
        '''

        '''
        W = tf.Variable(tf.random_uniform([len(vocab), self._embed_size], -1., 1.), 
            name='W')
        # embedding, [batch_size x max_text_len x embed_size]
        embedding = tf.nn.embedding_lookup(W, self._text)
        
        # reshape embedding to a list of [batch_size x embed_size] as required by RNN
        embedding = tf.transpose(embedding, [1, 0, 2])
        embedding = tf.reshape(embedding, [-1, self._embed_size])
        embedding = tf.split(0, self._max_text_len, embedding)

        # build n-layer RNN with LSTM cells
        cell = tf.nn.rnn_cell.LSTMCell(self._hidden_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout)
        if self._num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self._num_layers)
        output, _ = tf.nn.rnn(cell, embedding,
            dtype=tf.float32,
            sequence_length=self._text_len)
        last_output = output[-1]

        #linear transform 
        w = tf.Variable(tf.truncated_normal([self._hidden_size, len(self._keywords)], stddev=.01),
            name='weight')
        b = tf.Variable(tf.constant(.1, shape=[len(self._keywords)]),
            name='bias')

        #softmax
        self._prediction = tf.nn.softmax(tf.matmul(last_output, w) + b)

    def _add_cost_function(self):
        '''
        cross entropy

        '''
        self._cost = -tf.reduce_sum(self._targets * tf.log(self._prediction))

    def _train_op(self):
        '''
        
        '''
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self._lr)
        grads_and_vars = optimizer.compute_gradients(self._cost)
        return optimizer.apply_gradients(grads_and_vars, global_step=self._global_step)
    
    def _error(self):
        '''
        
        '''
        _, topk_targets = tf.nn.top_k(self._targets, self._num_kw)
        _, topk_predicts = tf.nn.top_k(self._prediction, self._num_kw)
        error = tf.not_equal(topk_targets, topk_predicts)
        return tf.reduce_mean(tf.cast(error, tf.float32))

class Vocab(object):

    def __init__(self, dataset, topk=0):
        '''

        '''
        self._dataset = dataset
        self._db = db[dataset]
        self._topk = topk
        self._build_vocab()

    @property
    def dataset(self):
        return self._dataset

    def get_word(self, id):
        return self._id_to_word[id] if id in self._id_to_word \
            else UNK_TOKEN

    def get_word_id(self, word):
        return self._word_to_id[word] if id in self._word_to_id \
            else self._word_to_id[UNK_TOKEN]

    def get_words(self, ids):
        return [self.get_word(id) for id in ids]

    def get_word_ids(self, words):
        return [self.get_word_id(w) for w in words]

    def _build_vocab(self):
        '''

        '''
        words = []
        for d in self._db.find():
            words += d['abstract'].split()
        words = sorted(Counter(words).items(),
            key=lambda x: -x[1])
        if self._topk:
            words = words[:self._topk]
        words, _ = zip(*words)
        words = list(words) + [PAD_TOKEN, UNK_TOKEN, ]
        ids = xrange(len(words))
        self._id_to_word = dict(zip(ids, words))
        self._word_to_id = dict(zip(words, ids))

    def __len__(self):
        return len(self._id_to_word)


class Keywords(object):
    '''
    
    '''
    def __init__(self, dataset, topk=0):
        '''

        '''
        self._dataset = dataset
        self._db = db[dataset]
        self._topk = topk
        self._build_keyword_dict()

    @property
    def dataset(self):
        return self._dataset

    def get_keyword(self, id):
        return self._id_to_keyword[id] if kw in self._id_to_keyword else None

    def get_keyword_id(self, kw):
        return self._keyword_to_id[kw] if kw in self._keyword_to_id else None

    def get_keywords(self, ids):
        return [self.get_keyword(id) for id in ids]

    def get_keyword_ids(self, keywords):
        return [self.get_keyword_id(kw) for kw in keywords]

    def _build_keyword_dict(self):
        '''

        ''' 
        keywords = []
        for d in self._db.find():
            keywords += d['keywords']
        keywords = sorted(Counter(keywords).items(),
            key=lambda x: -x[1])
        if self._topk:
            keywords = keywords[:self._topk]
        keywords, _ = zip(*keywords)
        ids = xrange(len(keywords))
        self._id_to_keyword = dict(zip(ids, keywords))
        self._keyword_to_id = dict(zip(keywords, ids))

    def __len__(self):
        return len(self._id_to_keyword)

class DataIterator(object):

    def __init__(self, vocab, keywords, cfg):
        '''
        
        '''
        self._vocab = vocab
        self._keywords = keywords
        self._cfg = cfg
        self._data = self._collect_data()

    def iterator(self):
        '''

        '''
        batch_size = self._cfg.batch_size
        if self._cfg.mode == 'train':
            i = 0
            j = 0
            steps = 0
            (text_batch, text_len_batch, 
                keywords_batch) = self._new_batch()
            while steps < self._cfg.max_steps * batch_size:
                text, text_len, keywords = self._data[j]
                for kw in keywords:
                    text_batch[i][:] = text[:]
                    text_len_batch[i] = text_len 
                    keywords_batch[i][kw] = 1
                    i = (i + 1)%batch_size
                    steps += 1
                    if (i + 1)%batch_size == 0:
                        text_batch_cp = text_batch.copy()
                        text_len_batch_cp = text_len_batch.copy()
                        keywords_batch_cp = keywords_batch.copy()
                        (text_batch, text_len_batch, 
                            keywords_batch) = self._new_batch()
                        yield (text_batch_cp, text_len_batch_cp, 
                                keywords_batch_cp)
                j = (j + 1)%len(self._data)
        else:
            for i in xrange(0, len(self._data), batch_size):
                batch = self._data[i : i + batch_size]
                text_batch = np.zeros([batch_size, self._cfg.max_text_len], np.int32)
                text_len_batch = np.zeros([batch_size], np.int32)
                keywords_batch = []
                for j, (text, text_len, keywords) in enumerate(batch):
                    text_batch[j][:] = text[:]
                    text_len_batch[j] = text_len
                    keywords_batch.append(keywords)
                yield text_batch, text_len_batch, keywords_batch
                        
    def _new_batch(self):
        batch_size = self._cfg.batch_size
        max_text_len = self._cfg.max_text_len 

        return (np.zeros([batch_size, max_text_len], np.int32),
            np.zeros([batch_size], np.int32),
            np.zeros([batch_size, len(self._keywords)], np.float32))

    def _collect_data(self):
        '''

        '''
        data = []
        pad_id = self._vocab.get_word_id(PAD_TOKEN)
        docs = [d for d in db[self._cfg.mode].find()]
        if self._cfg.max_docs:
            docs = docs[:self._cfg.max_docs]
        for d in docs:
            text = self._vocab.get_word_ids(
                d['abstract'].split()[:self._cfg.max_text_len])
            text_len = len(text)
            if text_len < self._cfg.max_text_len:
                text += [pad_id] * (self._cfg.max_text_len - text_len)
            keywords = self._keywords.get_keyword_ids(d['keywords'])
            keywords = [kwid for kwid in keywords if kwid]
            if not len(keywords):
                continue
            data.append((text, text_len, keywords, ))
        return data

def train(sess, saver, data_iter):
    model_path = '%s/%s'%(FLAGS.log_root, FLAGS.model_name)
    with sess:
        if os.path.exists(model_path):
            saver.restore(sess, model_path)
        sess.run(tf.initialize_all_variables())
        step = 0
        for texts, text_lens, keywords in data_iter.iterator():
            cost = model.run_train_step(
                sess, texts, text_lens, keywords, cfg.dropout)
            sys.stdout.write('Step: %d, cost: %.3f\n'%(step, cost, ))
            sys.stdout.flush()
            step += 1
            if step % 360 == 0:
                saver.save(sess, model_path)

def test(sess, saver, data_iter):
    model_path = '%s/%s'%(FLAGS.log_root, FLAGS.model_name)
    with sess:
        if not os.path.exists(model_path):
            sys.exit('Model does not exist')
        saver.restore(sess, model_path)
        sess.run(tf.initialize_all_variables())
        for texts, text_lens, keywords in data_iter.iterator():
            model.run_test_step(sess, texts, text_lens, keywords)
    

if '__main__' == __name__:
    cfg = config(
        mode=FLAGS.mode,
        max_text_len=100, 
        embed_size=64,
        hidden_size=256,
        batch_size=4,
        num_layers=4,
        num_kw=FLAGS.num_kw,
        max_docs=FLAGS.max_docs,
        max_steps=FLAGS.max_steps,
        dropout=.9,
        learning_rate=.01,
    )
    if cfg.mode == 'decode':
        cfg = cfg._replace(max_steps=cfg.max_docs)
    vocab = Vocab(FLAGS.dataset, topk=FLAGS.vocab_size)
    keywords = Keywords(FLAGS.dataset, topk=FLAGS.keyword_size)
    data_iter = DataIterator(vocab, keywords, cfg)
    model = KeywordsPredictionModel(vocab, keywords,
        cfg.max_text_len, cfg.embed_size, cfg.hidden_size,
        cfg.num_kw, cfg.num_layers, cfg.learning_rate,
        cfg.mode=='train')
    saver = tf.train.Saver()
    sess = tf.Session(
        config=tf.ConfigProto(allow_soft_placement=True))
    if cfg.mode == 'train':
        train(sess, saver, data_iter)
    elif cfg.mode == 'decode':
        test(sess, saver, data_iter)
