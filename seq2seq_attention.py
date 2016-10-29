# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Trains a seq2seq model.

WORK IN PROGRESS.

Implement "Abstractive Text Summarization using Sequence-to-sequence RNNS and
Beyond."

"""
import sys
import time

import tensorflow as tf
import data_manager
import data
import seq2seq_attention_decode
import seq2seq_attention_model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path',
                           '', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('vocab_path',
                           '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('article_key', 'article',
                           'tf.Example feature key for article.')
tf.app.flags.DEFINE_string('abstract_key', 'headline',
                           'tf.Example feature key for abstract.')
tf.app.flags.DEFINE_string('log_root', '', 'Directory for model root.')
tf.app.flags.DEFINE_string('train_dir', '', 'Directory for train.')
tf.app.flags.DEFINE_string('eval_dir', '', 'Directory for eval.')
tf.app.flags.DEFINE_string('decode_dir', '', 'Directory for decode summaries.')
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode mode')
tf.app.flags.DEFINE_integer('max_epochs', 16,
                            'Max epochs for training')
tf.app.flags.DEFINE_integer('max_article_sentences', 2,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('max_abstract_sentences', 100,
                            'Max number of first sentences to use from the '
                            'abstract')
tf.app.flags.DEFINE_integer('beam_size', 4,
                            'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run eval.')
tf.app.flags.DEFINE_integer('checkpoint_secs', 60, 'How often to checkpoint.')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', False,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'Number of gpus used.')
tf.app.flags.DEFINE_integer('random_seed', 111, 'A seed value for randomness.')


def _RunningAvgLoss(loss, running_avg_loss, summary_writer, step, decay=0.999):
  """Calculate the running average of losses."""
  if running_avg_loss == 0:
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)
  loss_sum = tf.Summary()
  loss_sum.value.add(tag='running_avg_loss', simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  sys.stdout.write('\rstep: %d, avg_loss: %f\n' %(step, running_avg_loss))
  sys.stdout.flush()
  return running_avg_loss


def _Train(model, data_batcher, eval_batcher, dropout):
  """Runs model training."""
  model.build_graph()
  saver = tf.train.Saver()
  # Train dir is different from log_root to avoid summary directory
  # conflict with Supervisor.
  summary_writer = tf.train.SummaryWriter(FLAGS.train_dir)
  sv = tf.train.Supervisor(logdir=FLAGS.log_root,
                           is_chief=True,
                           saver=saver,
                           summary_op=None,
                           save_summaries_secs=60,
                           save_model_secs=FLAGS.checkpoint_secs,
                           global_step=model.global_step)
  sess = sv.prepare_or_wait_for_session(
    config=tf.ConfigProto(
      allow_soft_placement=True))
  train_avg_loss = 0.
  eval_avg_loss = 0.
  best_avg_loss = float('inf')

  epoch = 0
  while not sv.should_stop() and epoch < FLAGS.max_epochs:
    sys.stdout.write('Epoch %d\n'%epoch)
    sys.stdout.write('Start training\n')
    sys.stdout.flush()
    # start training
    step = 0
    data_batcher_iter = data_batcher.data_iterator()
    vocab = data_batcher.vocab
    for b in data_batcher.data_iterator():
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
          loss_weights, _, _) = b
      for article in article_batch:
          sys.stdout.write(' '.join(vocab.get_words(article)) + '\n')
      for abstract in abstract_batch:
          sys.stdout.write(' '.join(vocab.get_words(abstract)) + '\n')
      (_, summaries, loss, train_step) = model.run_train_step(
          sess, article_batch, abstract_batch, targets, article_lens,
          abstract_lens, loss_weights, dropout)
      summary_writer.add_summary(summaries, train_step)
      train_avg_loss = _RunningAvgLoss(
          train_avg_loss, loss, summary_writer, step)
      step += 1
      if step % 100 == 0:
        summary_writer.flush()
    sys.stdout.write('Start evaluating\n')
    sys.stdout.flush()
    # start eval 
    step = 0
    for b in eval_batcher.data_iterator():
      (article_batch, abstract_batch, targets, article_lens, abstract_lens,
          loss_weights, _, _) = b
      summaries, loss, train_step = model.run_eval_step(
          sess, article_batch, abstract_batch, targets, article_lens,
          abstract_lens, loss_weights)
      summary_writer.add_summary(summaries, train_step)
      eval_avg_loss = _RunningAvgLoss(
          eval_avg_loss, loss, summary_writer, step)
      step += 1
      if step % 100 == 0:
        summary_writer.flush()
      
    if eval_avg_loss < best_avg_loss:
        best_avg_loss = eval_avg_loss 
    else:
        break
    epoch += 1

  sv.Stop()
  return eval_avg_loss

def main(unused_argv):
  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,
      min_lr=0.01,  # min learning rate.
      lr=0.15,  # learning rate
      batch_size=4, 
      enc_layers=2,
      enc_timesteps=120,
      dec_timesteps=30,
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=128,  # for rnn cell
      emb_dim=128,  # If 0, don't use embedding
      max_grad_norm=2,
      num_softmax_samples=4096,
      num_kw=3)  # If 0, no sampled softmax.

  vocab = data_manager.Vocab('pubmed')
  batcher = data_manager.DataManager(vocab, hps)
  tf.set_random_seed(FLAGS.random_seed)

  if hps.mode == 'train':
    eval_hps = hps._replace(mode='eval')
    eval_batcher = data_manager.DataManager(vocab, eval_hps)
    dropout = .9
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Train(model, batcher, eval_batcher, dropout)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=FLAGS.num_gpus)
    _Eval(model, batcher, vocab=vocab)
  elif hps.mode == 'decode':
    
    decode_mdl_hps = hps
    # Only need to restore the 1st step and reuse it since
    # we keep and feed in state for each step's output.
    decode_mdl_hps = hps._replace(dec_timesteps=1)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    decoder = seq2seq_attention_decode.BSDecoder(model, batcher, hps, vocab)
    decoder.DecodeLoop()


if __name__ == '__main__':
  tf.app.run()
