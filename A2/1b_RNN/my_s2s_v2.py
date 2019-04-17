from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from gensim.models import KeyedVectors as w2v
from re import match, sub


class data_prep():
    def __init__(self, txtfile, num_samples=10000):
        self.data_path = txtfile
        self.num_samples = num_samples  # Number of samples to train on.
        self.input_texts = []
        self.target_texts = []
        # tokens are characters or word vectors depending on the tokeniser used
        self.input_token_index = 'undef for word2vec'
        self.target_token_index = 'undef for word2vec'
        self.encoder_input_data = 'undef; run tokeniser function'
        self.decoder_input_data = 'undef; run tokeniser function'
        self.decoder_target_data = 'undef; run tokeniser function'
        self.en_stop = ['a', 'and', 'of', 'to']

    def as_char(self):
        input_tokens = set()
        target_tokens = set()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in input_tokens:
                    input_tokens.add(char)
            for char in target_text:
                if char not in target_tokens:
                    target_tokens.add(char)

        input_tokens = sorted(list(input_tokens))
        target_tokens = sorted(list(target_tokens))
        num_encoder_tokens = len(input_tokens)
        num_decoder_tokens = len(target_tokens)
        max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        print('Number of samples:', len(self.input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(input_tokens)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(target_tokens)])

        self.encoder_input_data = np.zeros(
            (len(self.input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        self.decoder_input_data = np.zeros(
            (len(self.input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        self.decoder_target_data = np.zeros(
            (len(self.input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(
                zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):  # one-hot ecoding
                self.encoder_input_data[i, t,
                                        self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):  # one-hot encoding
                # decoder_target_data is ahead of decoder_input_data by one
                # timestep
                self.decoder_input_data[i, t,
                                        self.target_token_index[char]] = 1.
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[i, t-1,
                                             self.target_token_index[char]] = 1.

    def load_w2v(self):
        en_path = './w2v_models/GoogleNews-vectors-negative300.bin'
        fr_path = './w2v_models/fr_200_skip_cut100.bin'
        en_mod = w2v.load_word2vec_format(en_path, binary=True, limit=int(2E5))
        fr_mod = w2v.load_word2vec_format(fr_path, binary=True)

        return en_mod, fr_mod

    def line_preproc(self, line, en_mod, fr_mod):
        skip_line = 0  # boolean if true, then skip line
        input_text, target_text = line.split('\t')
        # simplifications
        if match('\$?[0-9]', line):  # skip numbers
            skip_line = 1
            return skip_line, _, _, _, _
        if "'" in input_text:  # skip abbrev english sentences
            skip_line = 1
            return skip_line, _, _, _, _
        # cannot is not in en dict, whisky is spelt whiskey
        input_text = sub('cannot', 'can not', input_text)
        input_text = sub('whisky', 'whiskey', input_text)
        input_text = sub('[^\w ]', '', input_text)
        target_text = sub("[^\w \-']", '', target_text)
        # fr_mod is all lowercase and treats m' as a word
        target_text = sub("'", "' ", target_text).lower()
        target_text = sub("-", " ", target_text)
        target_text = sub("œ", "oe", target_text)
        input_words = input_text.split()
        target_words = target_text.split()

        unk_line_inputs = []
        unk_line_targets = []
        input_vecs = np.empty((0, 300))
        target_vecs = np.empty((0, 202))
        for word in input_words:
            word = sub('[^A-Za-z]', '', word)  # remove punctuation
            if word == '':
                continue
            # 4 stop words have capitalised versions in dict
            if word in self.en_stop:
                word = word.capitalize()
            if word not in en_mod.vocab:
                word = word.lower()
            if word not in en_mod.vocab:
                skip_line = 2
                unk_line_inputs.append((word, target_words))
            else:
                input_vecs = np.vstack((input_vecs, en_mod.word_vec(word)))
        for word in target_words:
            word = sub("[^\w']", '', word)  # remove punctuation
            if word == '':
                continue
            if word not in fr_mod.vocab:
                skip_line = 2
                unk_line_targets.append((word, target_words))
            else:
                vec = fr_mod.word_vec(word)
                vec = np.pad(vec, (2, 0), 'constant')
                target_vecs = np.vstack((target_vecs, vec))
        start_of_line = np.append([1, 0], np.zeros(200))
        end_of_line = np.append([0, 1], np.zeros(200))
        target_vecs = np.vstack((start_of_line, target_vecs, end_of_line))
        if skip_line == 0:
            self.input_texts.append(input_text)
            self.target_texts.append("\t" + target_text + "\n")

        return skip_line, input_vecs, target_vecs, unk_line_inputs, \
            unk_line_targets

    def dryrun_wrdvec(self):  # show words not covered by the w2v models
        unk_input = []
        unk_target = []
        en_mod, fr_mod = self.load_w2v()

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            skip, input_vecs, target_vecs, unk_in, unk_out = \
                self.line_preproc(line, en_mod, fr_mod)
            if skip == 1:
                continue
            unk_input.extend(unk_in)
            unk_target.extend(unk_out)

        return unk_input, unk_target

    def as_wrdvec(self):
        en_mod, fr_mod = self.load_w2v()
        in_vec_list = []
        tar_vec_list = []
        self.encoder_input_data = np.empty((0, 0, 300))
        self.decoder_input_data = np.empty((0, 0, 200))
        self.decoder_target_data = np.empty((0, 0, 200))

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            skip, input_vecs, target_vecs, unk_in, unk_out = \
                self.line_preproc(line, en_mod, fr_mod)
            if skip > 0:
                continue
            in_vec_list.append(input_vecs)
            tar_vec_list.append(target_vecs)

        return in_vec_list, tar_vec_list

            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
