import numpy as np
import re

np.random.seed(0)

word_min_counts_threshold = 3
max_caption_len = 50

special_tokens = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}


class DataObject:
    def __init__(self, myid, caption_list=None):
        # self.path = path
        self.myid = myid
        self.caption_list = caption_list  # no EOS, e.g. ['I', 'love', 'you']
        self.cap_len_list = len(self.caption_list) + 1  # EOS added, e.g. 4


class DatasetBase:
    MARKER_EOS = '<EOS>'
    MARKER_BOS = '<BOS>'
    MARKER_PAD = '<PAD>'
    marker = [MARKER_PAD, MARKER_BOS, MARKER_EOS, '<UNK>']

    def __init__(self, data_dir, batch_size):
        self.training_label_filename = 'training_label.json'
        self.testing_label_filename = 'testing_public_label.json'
        self.data_obj_list = []
        self.word_min_counts_threshold = word_min_counts_threshold
        self.vocab_num = 0
        self.word_freq_dict = dict()
        self.vocab_indices = dict()
        self.idx_to_word = list()
        self.feat_dict = dict()
        self.data_dir = data_dir
        self.batch_max_size = 0
        self.batch_size = batch_size
        self.batch_index = None

    @staticmethod
    def captionToTokenList(caption):
        # Remove trailing punctuations, e.g. '.'
        caption = re.sub('\W+$', '', caption)  # \W non-num\alph\_

        # Convert to lowercase
        caption = caption.lower()

        # abandon quotes : "blabla" -> blabla
        caption = re.sub('"(?P<bla>([a-zA-Z]+))"', lambda m: m.group('bla'), caption)

        # Isolate comma : add spaces to separate comma
        caption = re.sub('(?P<letter>\w),', lambda m: m.group('letter') + ' , ', caption)

        # Isolate 's : add spaces
        caption = re.sub("(?P<letter>\w)'s", lambda m: m.group('letter') + " 's ", caption)

        # Tokenize
        token_list = re.split('\s+', caption)  # separate with blank spaces

        # Add BOS,EOS
        token_list.insert(0, DatasetBase.MARKER_BOS)
        token_list.append(DatasetBase.MARKER_EOS)

        return token_list

    @staticmethod
    def sample_one_caption(captions, cap_len, is_rand=True):
        if isinstance(cap_len, int):
            return captions, cap_len
        # assert len(captions) == len(cap_len)
        if is_rand:
            r = np.random.randint(0, len(captions))
        else:
            r = 0
        return captions[r], cap_len[r]

    def captions_to_padded_sequences(self, captions, maxlen=max_caption_len):

        res = []
        for cap in captions:
            l = [self.vocab_indices[word] for word in cap]
            pad = special_tokens['<PAD>']
            l += [pad] * (maxlen - len(l))
            res.append(l)
        return res

    def next_batch(self, indices):
        # 1. sequential chosen
        if self.batch_index is None:
            self.batch_index = self.batch_max_size - 1

        current_index = self.batch_index
        idx = np.arange(current_index,current_index - self.batch_size,-1)
        # print('datasetBase:', idx, self.batch_index)
        if indices is None:
            dat_list = self.data_obj_list[idx]
        else:
            dat_list = self.data_obj_list[indices[idx]]
        self.batch_index -= self.batch_size
        self.batch_index %= self.batch_max_size
        # if current_index + self.batch_size <= max_size:
        #     dat_list = self.data_obj_list[self.perm[current_index:(current_index + self.batch_size)]]
        #     self.batch_index += self.batch_size
        # else:
        #     right = self.batch_size - (max_size - current_index)
        #     dat_list = np.append(self.data_obj_list[self.perm[current_index:max_size]],
        #                          self.data_obj_list[self.perm[0: right]])
        #     self.batch_index = right

        img_batch = []
        cap_batch = []
        id_batch = []
        cap_len = []

        for d in dat_list:
            img_batch.append(self.feat_dict[d.myid])
            id_batch.append(d.myid)
            cap, l = self.sample_one_caption(d.caption_list, d.cap_len_list)
            cap = np.array(cap)
            cap_batch.append(cap)
            cap_len.append(l)
        cap_batch = self.captions_to_padded_sequences(cap_batch)
        # print('finish next batch')

        return np.array(img_batch), np.array(cap_batch), np.array(cap_len), np.array(id_batch)
