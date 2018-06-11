import numpy as np
import pandas as pd
import pickle
import os
import random
from collections import defaultdict
import re

# from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from datasetBase import DatasetBase, DataObject

filters = '`","?!/.()'

max_caption_len = 50

random.seed(0)
np.random.seed(0)


class DatasetTrain(DatasetBase):

    def __init__(self, data_dir, batch_size):
        super().__init__(data_dir, batch_size)
        self.feat_dir = os.path.join(self.data_dir, 'training_data' + os.path.sep + 'feat')

        self.corpus_dir = self.data_dir
        self.perm = None  # permutation numpy array
        # self.data_obj_list = [] # defined in Base

    #
    # def prep_token_list(self):
    #
    #     corpus_path = self.corpus_dir + self.json_filename
    #     train_file = pd.read_json(corpus_path)
    #     total_list = []
    #     for i in range(0, len(train_file['caption'])):
    #         str_list = train_file['caption'][i]
    #         for j in range(0, len(str_list)):
    #             total_list.append(str_list[j])
    #     return total_list
    #
    # def dump_tokenizer(self):
    #     total_list = self.prep_token_list()
    #
    #     tokenizer = Tokenizer(filters=filters, lower=True, split=" ")
    #     tokenizer.fit_on_texts(total_list)
    #
    #     for tok in tokenizer.word_counts.items():
    #         if tok[1] >= self.word_min_counts_threshold:
    #             self.word_freq_dict[tok[0]] = tok[1]
    #
    #     self.vocab_num = len(self.word_freq_dict) + 4 # init vocab_num, must add 4 special tokens!!
    #
    #     for i in range(0, 4):
    #         tok = marker[i]
    #         self.vocab_indices[tok] = i
    #         self.idx_to_word[i] = tok
    #
    #     cnt = 0
    #     for tok in tokenizer.word_index.items():
    #         if tok[0] in self.word_freq_dict:
    #             self.vocab_indices[tok[0]] = cnt + 4
    #             self.idx_to_word[cnt + 4] = tok[0]
    #             cnt += 1
    #
    #     #assert len(self.word_counts) == self.vocab_num # no!! they are not equal
    #     assert len(self.vocab_indices) == self.vocab_num # yes! they are equal
    #
    #     with open('word_index.pkl', 'wb') as handle:
    #         pickle.dump(self.vocab_indices, handle)
    #     with open('idx_to_word.pkl', 'wb') as handle:
    #         pickle.dump(self.idx_to_word, handle)
    #     with open('word_counts.pkl', 'wb') as handle:
    #         pickle.dump(self.word_freq_dict, handle)
    #     return self.vocab_num # for embedding

    def construct_samples(self):
        """
        collect captions and construct a list of samples
        :return:
        """
        vid_captions_path = os.path.join(self.corpus_dir, self.training_label_filename)

        df = pd.read_json(vid_captions_path)  # r'D:\video_captioning\data\MLDS_hw2_data\training_label.json')

        with open('train_ids.txt', 'w') as f:
            f.writelines(df['id'])

        train_caption_list = []  # a list of caption-list for each video
        for caption_list, vid in zip(df['caption'], df['id']):  # train_label is a list of caption-vid dicts

            train_caption_list.append([])
            path = os.path.join(self.feat_dir, '{}.npy'.format(vid))
            feat = np.load(path)
            self.feat_dict[vid] = feat

            for caption in caption_list:  # a list of strings
                if not all(ord(c) < 128 for c in caption):
                    continue  # abandon captions with unusual chars
                token_list = self.captionToTokenList(caption)
                train_caption_list[-1].append(token_list)  # for the last video's caption-list
                self.data_obj_list.append(DataObject(myid=vid, caption_list=token_list))

        self.data_obj_list = np.array(self.data_obj_list)
        self.batch_max_size = len(self.data_obj_list)
        self.perm = np.arange(self.batch_max_size, dtype=np.int)
        self.shuffle_perm()
        print('[Training] total data size: ' + str(self.batch_max_size))

        # construct vocab
        self.word_freq_dict = defaultdict(int)
        # for mk in self.marker:
        #     self.word_freq_dict[mk] = 0
        # total_word_count = 0.0
        for caption_list in train_caption_list:
            for caption in caption_list:
                for token in caption:
                    if token in self.marker : continue
                    self.word_freq_dict[token] += 1
                    # total_word_count += 1.0

        # also save the testing's vocab
        df = pd.read_json(os.path.join(self.corpus_dir, self.testing_label_filename))
        for caption_list in df['caption']:
            for caption in caption_list:
                token_list = self.captionToTokenList(caption)
                for token in token_list:
                    if token in self.marker : continue
                    self.word_freq_dict[token] += 1

        # for word in self.word_freq_dict:
        #     self.word_freq_dict[word] /= np.sum(self.word_freq_dict.values())
        # return a new list of k-v tuples, sorted by the freq of word (the value), in the ascending order (reverse)
        # word_freq_list = sorted(iter(self.word_freq_dict.items()), key=lambda k_v: k_v[1], reverse=True)
        self.idx_to_word = self.marker+list(self.word_freq_dict.keys())
        # self.word_index_dict = dict([(self.vocabulary[i], i) for i in range(len(self.vocabulary))])
        self.vocab_indices = {word: idx for idx, word in enumerate(self.idx_to_word)}

        # store in pickle
        with open('vocab_indices.pkl', 'wb') as handle:
            pickle.dump(self.vocab_indices, handle)
        with open('idx_to_word.pkl', 'wb') as handle:
            pickle.dump(self.idx_to_word, handle)
        with open('word_freq.pkl', 'wb') as handle:
            pickle.dump(self.word_freq_dict, handle)

        return len(self.vocab_indices)

    def shuffle_perm(self):
        np.random.shuffle(self.perm)
        # print(self.perm)

    def next_batch_(self):
        return super().next_batch(self.perm)
        # # 1. sequential chosen
        # current_index = self.batch_index
        # max_size = self.batch_max_size
        # if current_index + self.batch_size <= max_size:
        #     dat_list = self.data_obj_list[self.perm[current_index:(current_index + self.batch_size)]]
        #     self.batch_index += self.batch_size
        # else:
        #     right = self.batch_size - (max_size - current_index)
        #     dat_list = np.append(self.data_obj_list[self.perm[current_index:max_size]],
        #             self.data_obj_list[self.perm[0: right]])
        #     self.batch_index = right
        #
        # img_batch = []
        # cap_batch = []
        # id_batch = []
        # cap_len = []
        # for d in dat_list:
        #     img_batch.append(self.feat_dict[d.myid])
        #     id_batch.append(d.myid)
        #     cap, l = self.sample_one_caption(d.caption_list, d.cap_len_list)
        #     cap = np.array(cap)
        #     cap_batch.append(cap)
        #     cap_len.append(l)
        # cap_batch = self.captions_to_padded_sequences(cap_batch)
        #
        # return np.array(img_batch), np.array(cap_batch), np.array(cap_len), np.array(id_batch)

    @staticmethod
    def schedule_sampling(sampling_prob):

        sampling = np.ones(max_caption_len, dtype=bool)
        for l in range(max_caption_len):
            if np.random.uniform(0, 1, 1) < sampling_prob:
                sampling[l] = True
            else:
                sampling[l] = False

        sampling[0] = True
        return sampling


if __name__ == '__main__':
    datasetTrain = DatasetTrain(r'data', 80)
    datasetTrain.construct_samples()
