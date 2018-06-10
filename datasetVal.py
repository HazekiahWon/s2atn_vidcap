import numpy as np
import pandas as pd
import pickle
import os

# from keras.preprocessing.text import text_to_word_sequence
from datasetBase import DatasetBase, DataObject

filters = '`","?!/.()'


class DatasetVal(DatasetBase):
    def __init__(self, data_dir, batch_size):

        super().__init__(data_dir, batch_size)
        self.feat_dir = os.path.join(self.data_dir, 'testing_data' + os.path.sep + 'feat')
        self.json_filename = 'testing_label.json'
        self.corpus_dir = self.data_dir

    def load_tokenizer(self):
        # should be put in same folder!
        with open('vocab_indices.pkl', 'rb') as handle:
            self.vocab_indices = pickle.load(handle)
        with open('idx_to_word.pkl', 'rb') as handle:
            self.idx_to_word = pickle.load(handle)
        with open('word_freq.pkl', 'rb') as handle:
            self.word_freq_dict = pickle.load(handle)

        self.vocab_num = len(self.word_freq_dict)
        return self.vocab_num

    def construct_samples(self):
        corpus_path = os.path.join(self.corpus_dir, self.testing_label_filename)

        df = pd.read_json(corpus_path)

        for caption_list, vid in zip(df['caption'], df['id']):  # train_label is a list of caption-vid dicts
            for caption in caption_list:  # a list of strings
                if not all(ord(c) < 128 for c in caption):
                    continue  # abandon captions with unusual chars
                token_list = self.captionToTokenList(caption)

                self.data_obj_list.append(DataObject(myid=vid, caption_list=token_list))

        self.data_obj_list = np.array(self.data_obj_list)
        self.batch_max_size = len(self.data_obj_list)
        print('[Validation] total data size: ' + str(self.batch_max_size))

    def next_batch_(self):
        return super().next_batch(indices=None)
        # # 1. sequential chosen, batch_size should be <= 100
        # current_index = self.batch_index
        # max_size = self.batch_max_size
        # if current_index + self.batch_size <= max_size:
        #     dat_list = self.data_obj_list[current_index:(current_index + self.batch_size)]
        #     self.batch_index += self.batch_size
        # else:
        #     right = self.batch_size - (max_size - current_index)
        #     dat_list = np.append(self.data_obj_list[current_index:max_size], self.data_obj_list[0: right])
        #     self.batch_index = right
        #
        # img_batch = []
        # cap_batch = []
        # id_batch = []
        # cap_len = []
        # for d in dat_list:
        #     img_batch.append(self.feat_dict[d.myid])
        #     id_batch.append(d.myid)
        #     cap, l = self.sample_one_caption(d.caption_list, d.cap_len_list) # randomly pick one
        #     cap = np.array(cap)
        #     cap_batch.append(cap)
        #     cap_len.append(l)
        # cap_batch = self.captions_to_padded_sequences(cap_batch)
        #
        # return np.array(img_batch), np.array(cap_batch), np.array(cap_len), np.array(id_batch)
