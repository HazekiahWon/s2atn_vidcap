from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import random
import sys
from subprocess import call

import numpy as np
import tensorflow as tf
# import tqdm as tqdm
# # from colors import *
# from tqdm import *

from datasetTest import DatasetTest
from datasetTrain import DatasetTrain
from datasetVal import DatasetVal
from util import inv_sigmoid, dec_print_train, dec_print_val, dec_print_test

FLAGS = None

random.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

n_vgg = 4096
n_hidden = 600
# batch_size = 100
val_batch_size = 40  # 100
n_frames = 80
max_caption_len = 50
forget_bias_red = 1.0
forget_bias_gre = 1.0
dropout_prob = 0.5
n_attention = n_hidden

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
phases = {'train': 0, 'val': 1, 'test': 2}


class S2VT:
    def __init__(self, vocab_num=0,
                 with_attention=True,
                 lr=1e-4):

        self.vocab_num = vocab_num
        self.with_attention = with_attention
        self.learning_rate = lr
        self.saver = None

    def set_saver(self, saver):
        self.saver = saver

    def build_model(self, feat, captions=None, cap_len=None, sampling=None, phase=0):
        with tf.variable_scope('trainable_vars'):
            weights = {
                'W_feat': tf.Variable(tf.random_uniform([n_vgg, n_hidden], -0.1, 0.1), name='W_feat'),
                'W_dec': tf.Variable(tf.random_uniform([n_hidden, self.vocab_num], -0.1, 0.1), name='W_dec')
            }
            biases = {
                'b_feat': tf.Variable(tf.zeros([n_hidden]), name='b_feat'),
                'b_dec': tf.Variable(tf.zeros([self.vocab_num]), name='b_dec')
            }
            # ??
            embeddings = {
                'emb': tf.Variable(tf.random_uniform([self.vocab_num, n_hidden], -0.1, 0.1), name='emb')
            }

            if self.with_attention:
                print('wrapped with bahdanau_attention...')
                # weights['w_enc_out'] =  tf.Variable(tf.random_uniform([n_hidden, n_hidden]),
                #     dtype=tf.float32, name='w_enc_out')
                # weights['w_dec_state'] =  tf.Variable(tf.random_uniform([n_hidden, n_hidden]),
                #     dtype=tf.float32, name='w_dec_state')
                weights['w_bah_atn'] = tf.Variable(tf.random_uniform([2 * n_hidden, n_hidden]),
                                                   dtype=tf.float32, name='w_bah_atn')
                weights['v'] = tf.Variable(tf.random_uniform([n_hidden, 1]),
                                           dtype=tf.float32, name='v')
                weights['w_luong_atn'] = tf.Variable(tf.random_uniform([2 * n_hidden, n_hidden]),
                                                     dtype=tf.float32, name='w_bah_atn')

        batch_size = tf.shape(feat)[0]

        with tf.variable_scope('prep'):
            if phase != phases['test']:
                # b,max_cap_len
                cap_mask = tf.sequence_mask(cap_len, max_caption_len, dtype=tf.float32)

            if phase == phases['train']:  # add noise
                # noise = tf.random_uniform(tf.shape(feat), -0.1, 0.1, dtype=tf.float32)
                # feat = tf.add(feat,noise,name='add_noise') # TODO why noise

                # if phase == phases['train']:
                feat = tf.nn.dropout(feat, dropout_prob)

        with tf.variable_scope('vid_fur_emb'):
            # b,t,n_vgg -> b*t,n_vgg
            feat = tf.reshape(feat, [-1, n_vgg])
            # b*t,n_vgg -> b*t,n_frame_enc
            image_emb = tf.matmul(feat, weights['W_feat']) + biases['b_feat']
            # b*t,n_frame_enc -> b,t,n_frame_enc
            image_emb = tf.reshape(image_emb, [-1, n_frames, n_hidden])
            # b,t,n_frame_enc -> t,b,n_frame_enc
            image_emb = tf.transpose(image_emb, perm=[1, 0, 2])

        with tf.variable_scope('LSTM1'):
            lstm_red = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=forget_bias_red, state_is_tuple=True)
            if phase == phases['train']:
                lstm_red = tf.contrib.rnn.DropoutWrapper(lstm_red, output_keep_prob=dropout_prob)
            red_hc_statetup = lstm_red.zero_state(batch_size, dtype=tf.float32)

        with tf.variable_scope('LSTM2'):
            lstm_gre = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=forget_bias_gre, state_is_tuple=True)
            if phase == phases['train']:
                lstm_gre = tf.contrib.rnn.DropoutWrapper(lstm_gre, output_keep_prob=dropout_prob)
            gre_hc_statetup = lstm_gre.zero_state(batch_size, dtype=tf.float32)


        with tf.variable_scope('prep'):
            if self.with_attention:
                # TODO why padding as such : prev_out + current_atn
                padding = tf.zeros([batch_size, n_hidden + n_attention])
            else:
                padding = tf.zeros([batch_size, n_hidden])

        h_encs = []  # collect lstm2 outputs
        for t in range(0, n_frames):  # at each time t
            with tf.variable_scope("LSTM1"):
                # b,n_hidden
                red_h_state, red_hc_statetup = lstm_red(image_emb[t, :, :], red_hc_statetup)

            with tf.variable_scope("LSTM2"):
                gre_h_state, gre_hc_statetup = lstm_gre(tf.concat([padding, red_h_state], axis=-1), gre_hc_statetup)
                h_encs.append(gre_h_state)  # even though padding is augmented, output_gre/state_gre's shape not change

        h_encs = tf.stack(h_encs, axis=0)

        logits = []#[tf.one_hot(tf.ones(shape=(batch_size,), dtype=tf.int32), depth=self.vocab_num)]

        if self.with_attention:
            # v*tanh(w*[h_encs,h_dec])
            def bahdanau_attention(time, prev_dec_h=None):

                if time == 0:
                    H_t = h_encs[-1, :, :]  # encoder last output as first target input, H_t
                else:
                    H_t = prev_dec_h

                # t,b,n_hidden
                H_t_broadcast = tf.stack([H_t] * n_frames)
                # t,b,n_hidden*2
                H_concat = tf.concat((h_encs, H_t_broadcast), axis=-1, name='H_concat')
                # t*b,n_hidden
                tmp = tf.matmul(tf.reshape(H_concat, shape=(-1, n_hidden * 2)), weights['w_bah_atn'])
                # t,b,1
                score = tf.reshape(tf.matmul(tf.tanh(tmp), weights['v']), shape=(n_frames, batch_size, 1))
                # H_t = tf.matmul(H_t, weights['w_dec_state'])
                # # t,b,n_hidden
                # H_s = tf.identity(h_encs) # copy
                #
                # # t*b,n_hidden
                # H_s = tf.reshape(H_s, (-1, n_hidden))
                # score = tf.matmul(H_s, weights['w_enc_out'])
                # score = tf.reshape(score, (-1, batch_size, n_hidden))
                # score = tf.add(score, tf.expand_dims(H_t, 0))
                #
                # score = tf.reshape(score, (-1, n_hidden))
                # # t*b,n_hidden -> t*b,1
                # score = tf.matmul(tf.tanh(score), weights['v'])
                # t,b,1
                score = tf.reshape(score, (n_frames, batch_size, 1))
                # TODO the broadcast may go wrong
                score = tf.nn.softmax(score, dim=0, name='alpha')  # org. dim=-1

                # H_s = tf.reshape(H_s, (-1, batch_size, n_hidden))
                # b,n_hidden
                # C_i = tf.reduce_sum(tf.multiply(H_s, score), axis=0)
                C_i = tf.reduce_sum(tf.multiply(h_encs, score), axis=0)
                return C_i,score

        cross_ent_list = []
        atn_ws = []
        for t in range(0, max_caption_len):

            with tf.variable_scope("LSTM1"):
                # bos = tf.ones([batch_size, n_hidden])
                padding_in = tf.zeros([batch_size, n_hidden])
                red_h_state, red_hc_statetup = lstm_red(padding_in, red_hc_statetup)

            with tf.variable_scope("LSTM2"):
                if t == 0:
                    feed_in = tf.ones(shape=(batch_size,), dtype=tf.int32) # BOS
                    # with tf.variable_scope("LSTM2"):
                    #     con = tf.concat([bos, red_h_state], axis=-1)
                    #     if self.with_attention:
                    #         C_i = bahdanau_attention(t)
                    #         # C_i = luong_attention(i)
                    #         con = tf.concat([con, C_i], axis=-1)
                    #
                    #     gre_h_state, gre_hc_statetup = lstm_gre(con, gre_hc_statetup)
                else:
                    if phase == phases['train']:
                        if sampling[t] is True:
                            feed_in = captions[:, t - 1]  # all batches at prev time
                        else:
                            # feed_in = tf.argmax(logit_words, axis=1)  # largest word index
                            feed_in = tf.squeeze(tf.multinomial(logit_words, num_samples=1))
                            # print('mm',feed_in.shape)
                    else:  # test
                        feed_in = tf.argmax(logit_words, 1)
                        # print(feed_in.shape)

                with tf.device("/cpu:0"):
                    # word embedding lookup
                    embed_result = tf.nn.embedding_lookup(embeddings['emb'], feed_in)

                con = tf.concat([embed_result, red_h_state], axis=1)
                if self.with_attention:
                    C_i,atn_w_at_t = bahdanau_attention(t, gre_hc_statetup[1])  # (state_c, state_h)
                    atn_ws.append(atn_w_at_t) # t'*[t,b,1] -> t',t,b,1 ->(reduce_sum)->t,b,1->(-tau)->(reduce_sum)scalar
                    # C_i = luong_attention(i, state_gre[1])
                    con = tf.concat([con, C_i], axis=1)
                gre_h_state, gre_hc_statetup = lstm_gre(con, gre_hc_statetup)

            with tf.variable_scope('vocab_logits'):
                logit_words = tf.matmul(gre_h_state, weights['W_dec']) + biases['b_dec']  # b,n_vocab
                logits.append(logit_words)

                if phase != phases['test']:
                    labels = captions[:, t]
                    # onehot for the t-th word
                    one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value=1, off_value=None, axis=1)
                    # b,1
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=one_hot_labels)
                    # at time t, which loss should be counted
                    cross_entropy = cross_entropy * cap_mask[:, t]  # b,1
                    cross_ent_list.append(cross_entropy)

        with tf.variable_scope('loss'):
            loss = 0.0
            xent_loss = 0.
            atn_loss = 0.
            tau = 1./n_frames * max_caption_len*0.5  # expectation for each frame to be attended is 1/n_frames
            if phase != phases['test']:
                cross_entropy_tensor = tf.stack(cross_ent_list, 1)  # b,t,1
                xent_loss = tf.reduce_sum(cross_entropy_tensor, axis=1)  # b,1
                xent_loss = tf.divide(xent_loss, tf.cast(cap_len, tf.float32))  # b,1
                xent_loss = tf.reduce_mean(xent_loss, axis=0)  # 1,
                stacked_atn_ws = tf.squeeze(tf.stack(atn_ws)) # t_maxcap, t_frames, b,
                # cap_mask : b, t_maxcap
                expanded_mask = tf.transpose(tf.expand_dims(cap_mask, axis=-1), perm=(1,2,0))
                masked_atn_ws = tf.multiply(stacked_atn_ws, expanded_mask) # t_maxcap,t_frames,b
                atn_focus_each_frame = tf.reduce_sum(masked_atn_ws, axis=0) # t_frames,b
                # require x in [0.8,1.2]*alpha
                over = tf.nn.relu(atn_focus_each_frame-1.2*tau, ) # let those over 1.2*tau pass
                under = tf.nn.relu(0.6*tau-atn_focus_each_frame)

                avg_frame_bias = tf.reduce_sum(over+under, axis=0) / tf.cast(cap_len, tf.float32) # b,

                atn_loss = tf.reduce_mean(avg_frame_bias)
                loss = xent_loss + atn_loss


        with tf.variable_scope('vocab_logits'):
            logits = tf.stack(logits, axis=0)  # t,b,n_vocab
            # logits = tf.reshape(logits, (max_caption_len, batch_size, self.vocab_num))
            logits = tf.transpose(logits, [1, 0, 2])  # b,t,n_vocab

        with tf.variable_scope('summary'):
            summaries = []
            if phase == phases['train']:
                summaries.append(tf.summary.scalar('xent', xent_loss))
                summaries.append(tf.summary.scalar('atn', atn_loss))
                summaries.append(tf.summary.histogram('atn_focus_each_frame', atn_focus_each_frame))
                # summaries.append(tf.summary.histogram('valid atn', valid_atn_loss))
            elif phase == phases['val']:
                summaries.append(tf.summary.scalar('val xent', xent_loss))
                summaries.append(tf.summary.scalar('val atn', atn_loss))


        return logits, loss, tf.summary.merge(summaries)


    def inference(self, logits):

        # print('using greedy search...')
        # logits : b,t,n_voc -> b*t,n_voc
        shape = tf.shape(logits)[:-1]
        unpack = tf.reshape(logits, shape=(-1, self.vocab_num))
        sentences = tf.squeeze(tf.multinomial(unpack, num_samples=1)) # b*t,
        sentences = tf.reshape(sentences, shape=shape)
        # dec_pred = tf.argmax(logits, axis=-1)
        return sentences #dec_pred

    def optimize(self, loss_op):

        params = tf.trainable_variables()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)  # .minimize(loss_op)
        # gradient clipping
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        train_op = optimizer.apply_gradients(zip(gradients, params))

        return train_op


def train():
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True

    ##================== train =================================##
    datasetTrain = DatasetTrain(FLAGS.data_dir, FLAGS.batch_size)
    vocab_num = datasetTrain.construct_samples()

    print("vocab_num: " + str(vocab_num))

    train_graph = tf.Graph()

    print('train_graph: start')
    with train_graph.as_default():
        feat = tf.placeholder(tf.float32, [None, n_frames, n_vgg], name='video_features')
        captions = tf.placeholder(tf.int32, [None, max_caption_len], name='captions')
        sampling = tf.placeholder(tf.bool, [max_caption_len], name='sampling')
        cap_len = tf.placeholder(tf.int32, [None], name='cap_len')
        model = S2VT(vocab_num=vocab_num, with_attention=FLAGS.with_attention,
                     lr=FLAGS.learning_rate)
        logits_tsr, loss_op, summary_op = model.build_model(feat, captions, cap_len, sampling, phases['train'])
        pred_word_indices = model.inference(logits_tsr)
        train_op = model.optimize(loss_op)

        model.set_saver(tf.train.Saver(max_to_keep=5))
        init_op = tf.global_variables_initializer()

    train_sess = tf.Session(graph=train_graph, config=gpu_config)

    ##============================ val ============================##
    datasetVal = DatasetVal(FLAGS.data_dir, val_batch_size)
    datasetVal.construct_samples()
    datasetVal.load_tokenizer()  # vocab_num are the same
    val_graph = tf.Graph()

    print('val_graph: start')
    with val_graph.as_default():
        feat_val = tf.placeholder(tf.float32, [None, n_frames, n_vgg], name='video_features')
        captions_val = tf.placeholder(tf.int32, [None, max_caption_len], name='captions')
        cap_len_val = tf.placeholder(tf.int32, [None], name='cap_len')

        model_val = S2VT(vocab_num=vocab_num, with_attention=FLAGS.with_attention, lr=FLAGS.learning_rate)
        logits_val, loss_op_val, summary_val = model_val.build_model(feat_val,
                                                                     captions_val, cap_len_val, phase=phases['val'])
        dec_pred_val = model_val.inference(logits_val)

        model_val.set_saver(tf.train.Saver(max_to_keep=5))
    val_sess = tf.Session(graph=val_graph, config=gpu_config)

    ##====================== saver ========================##
    load = FLAGS.load_saver
    if not load:
        train_sess.run(init_op)
        print("No saver was loaded")
    else:
        saver_path = FLAGS.save_dir  # load checkpoint, but the loss in tensorboard will crash!!
        latest_checkpoint = tf.train.latest_checkpoint(saver_path)
        model.saver.restore(train_sess, latest_checkpoint)
        print("Saver Loaded: " + latest_checkpoint)

    ckpts_path = os.path.join(FLAGS.save_dir, "save_net.ckpt")

    ##===================== summary =======================##
    summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'))
    summary_writer.add_graph(train_graph)
    summary_writer.add_graph(val_graph)

    samp_prob = inv_sigmoid(FLAGS.num_epoches)
    # pbar = tqdm(range(0, FLAGS.num_epoches))  # process bar

    for epo in range(FLAGS.num_epoches):
        datasetTrain.shuffle_perm()
        num_steps = int(datasetTrain.batch_max_size / FLAGS.batch_size)
        epo_loss = 0
        for i in range(0, num_steps):
            data_batch, label_batch, caption_lens_batch, id_batch = datasetTrain.next_batch_()
            samp = datasetTrain.schedule_sampling(samp_prob[epo])

            if (i+1) % FLAGS.num_display_steps == 0:
                # training
                # print('model:display')
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                _, loss, p, summ = train_sess.run([train_op, loss_op, pred_word_indices, summary_op],
                                                  feed_dict={feat: data_batch,
                                                             captions: label_batch,
                                                             cap_len: caption_lens_batch,
                                                             sampling: samp},
                                                  # options=run_options
                                                  )

                summary_writer.add_summary(summ, global_step=(epo * num_steps) + i)
                print("\n[Train. Prediction] Epoch " + str(epo) + ", step " \
                      + str(i) + "/" + str(num_steps) + "......")
                dec_print_train(p, caption_lens_batch, label_batch,
                                datasetTrain.idx_to_word, FLAGS.batch_size, id_batch)
                print('============================================================')

            else:
                # print('model:', data_batch.shape)
                _, loss, p = train_sess.run([train_op, loss_op, pred_word_indices],
                                            feed_dict={feat: data_batch,
                                                       captions: label_batch,
                                                       cap_len: caption_lens_batch,
                                                       sampling: samp})

            epo_loss += loss

            print("Epoch " + str(epo) + ", step " + str(i) + "/" + str(num_steps) + \
                                 ", (Training Loss: " + "{:.4f}".format(loss) + \
                                 ", samp_prob: " + "{:.4f}".format(samp_prob[epo]) + ")")


        print("\n[FINISHED] Epoch " + str(epo) + \
              ", (Training Loss (per epoch): " + "{:.4f}".format(epo_loss) + \
              ", samp_prob: " + "{:.4f}".format(samp_prob[epo]) + ")")

        if epo % FLAGS.num_saver_epoches == 0:
            ckpt_path = model.saver.save(train_sess, ckpts_path,
                                         global_step=(epo * num_steps) + num_steps - 1)
            print("\nSaver saved: " + ckpt_path)

            # validation
            model_val.saver.restore(val_sess, ckpt_path)
            print("\n[Val. Prediction] Epoch " + str(epo) + ", step " + str(num_steps) + "/" \
                  + str(num_steps) + "......")

            num_steps_val = int(datasetVal.batch_max_size / val_batch_size)
            total_loss_val = 0
            txt = open(FLAGS.output_filename, 'w')
            for j in range(0, num_steps_val):

                data_batch, label_batch, caption_lens_batch, id_batch = datasetVal.next_batch_()
                loss_val, p_val, summ = val_sess.run([loss_op_val, dec_pred_val, summary_val],
                                                     feed_dict={feat_val: data_batch,
                                                                captions_val: label_batch,
                                                                cap_len_val: caption_lens_batch})
                seq = dec_print_val(p_val, caption_lens_batch,
                                    label_batch, datasetVal.idx_to_word, val_batch_size, id_batch)

                total_loss_val += loss_val
                summary_writer.add_summary(summ, global_step=(epo * num_steps_val) + j)

                for k in range(0, val_batch_size):
                    txt.write(id_batch[k] + "," + seq[k] + "\n")

            print('\nSave file: ' + FLAGS.output_filename)
            txt.close()
            call(['python3', 'bleu_eval.py', FLAGS.output_filename])

            print("Validation: " + str((j + 1) * val_batch_size) + "/" + \
                  str(datasetVal.batch_max_size) + ", done..." \
                  + "Total Loss: " + "{:.4f}".format(total_loss_val))

    print('\n\nTraining finished!')


def test():
    # data_dir no use! only for datasetTest
    datasetTest = DatasetTest(FLAGS.data_dir, FLAGS.test_dir, FLAGS.batch_size)
    datasetTest.build_test_data_obj_list()
    vocab_num = datasetTest.load_tokenizer()

    test_graph = tf.Graph()
    # gpu_config = tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth = True

    with test_graph.as_default():
        feat = tf.placeholder(tf.float32, [None, n_frames, n_vgg], name='video_features')
        model = S2VT(vocab_num=vocab_num, with_attention=FLAGS.with_attention)
        logits, _, _ = model.build_model(feat, phase=phases['test'])
        dec_pred = model.inference(logits)

        model.set_saver(tf.train.Saver(max_to_keep=3))
    sess = tf.Session(graph=test_graph, config=gpu_config)

    saver_path = FLAGS.save_dir
    print('saver path: ' + saver_path)
    latest_checkpoint = tf.train.latest_checkpoint(saver_path)

    model.saver.restore(sess, latest_checkpoint)
    print("Saver Loaded: " + latest_checkpoint)

    txt = open(FLAGS.output_filename, 'w')

    num_steps = int(datasetTest.batch_max_size / FLAGS.batch_size)
    for i in range(0, num_steps):

        data_batch, id_batch = datasetTest.next_batch_()
        p = sess.run(dec_pred, feed_dict={feat: data_batch})
        seq = dec_print_test(p, datasetTest.idx_to_word, FLAGS.batch_size, id_batch)

        for j in range(0, FLAGS.batch_size):
            txt.write(id_batch[j] + "," + seq[j] + "\n")

        print("Inference: " + str((i + 1) * FLAGS.batch_size) + "/" + \
              str(datasetTest.batch_max_size) + ", done...")

    print('\n\nTesting finished.')
    print('\n Save file: ' + FLAGS.output_filename)
    txt.close()


def main(_):
    if not FLAGS.test_mode:  # training
        if FLAGS.load_saver:
            print('load saver!!')
        else:
            print('not load saver, init')

        # whether load saver or not, reinitialized the log dir
        if tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
        print('train mode: start')
        train()
    else:
        if FLAGS.load_saver:
            print('load saver!!')
        else:
            print('ERROR: you cannot run test without saver...')
            exit(0)
        print('test mode: start')
        # pick from 1 of 2
        # test()
        train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--num_epoches', type=int, default=150)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-t', '--test_mode', type=int, default=0)
    parser.add_argument('-d', '--num_display_steps', type=int, default=15)
    parser.add_argument('-ns', '--num_saver_epoches', type=int, default=1)
    parser.add_argument('-s', '--save_dir', type=str, default='save/')
    parser.add_argument('-l', '--log_dir', type=str, default='logs/')
    parser.add_argument('-o', '--output_filename', type=str, default='output.txt')
    parser.add_argument('-lo', '--load_saver', type=int, default=0)
    parser.add_argument('-at', '--with_attention', type=int, default=1)
    parser.add_argument('--data_dir', type=str,
                        default=r'data'
                        )
    parser.add_argument('--test_dir', type=str,
                        default='/home/data/MLDS_hw2_1_data/testing_data'
                        )

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
