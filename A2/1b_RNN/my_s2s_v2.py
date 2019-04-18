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
        de_path = './w2v_models/german.model'
        en_mod = w2v.load_word2vec_format(en_path, binary=True, limit=int(2E5))
        fr_mod = w2v.load_word2vec_format(fr_path, binary=True)
        de_mod = w2v.load_word2vec_format(de_path, binary=True)
        return en_mod, fr_mod, de_mod

    def de_line_preproc(self, line, en_mod, de_mod):
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
        target_text = sub("[^\w ]", '', target_text)
        # de_mod requires no filtering
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
            if word == '':
                continue
            if word not in de_mod.vocab:
                skip_line = 2
                unk_line_targets.append((word, target_words))
            else:
                vec = de_mod.word_vec(word)
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
        en_mod, fr_mod, de_mod = self.load_w2v()

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
        en_mod, fr_mod, de_mod = self.load_w2v()
        in_max_len = 0
        tar_max_len = 0
        in_vec_list = []
        tar_vec_list = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            skip, input_vecs, target_vecs, unk_in, unk_out = \
                self.line_preproc(line, en_mod, fr_mod)
            if skip > 0:
                continue
            in_vec_list.append(input_vecs)
            tar_vec_list.append(target_vecs)
            in_max_len = max((in_max_len, input_vecs.shape[0]))
            tar_max_len = max((tar_max_len, target_vecs.shape[0]))

        n = len(in_vec_list)
        self.encoder_input_data = np.zeros((n, in_max_len, 300))
        self.decoder_input_data = np.zeros((n, tar_max_len, 202))
        self.decoder_target_data = np.zeros((n, tar_max_len, 202))
        for i, (in_arr, tar_arr) in enumerate(zip(in_vec_list, tar_vec_list)):
            in_len = in_arr.shape[0]
            tar_len = tar_arr.shape[0]
            self.encoder_input_data[i, :in_len, :] = in_arr
            self.decoder_input_data[i, :tar_len, :] = tar_arr
            self.decoder_target_data[i, :tar_len-1, :] = tar_arr[1:, :]
        # french word vecs are in the range of +- 1.35, normalise for use with
        # tanh activation function
        self.decoder_input_data /= 1.35
        self.decoder_target_data /= 1.35


class seq2seq():
    def __init__(self, data_prep):
        self.model = None
        self.enc_model = None
        self.dec_model = None
        self.data = data_prep
        # self.enc_in_dat = data_prep.encoder_input_data
        # self.dec_in_dat = data_prep.decoder_input_data
        # self.dec_tar_dat = data_prep.decoder_target_data

    def twolayer(self, act_func, loss_func, latent_dim=256):
        # act_funcs: softmax, tanh, linear
        # loss_funcs: categorical_crossentropy, mean_squared_error
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,
                                      self.data.encoder_input_data.shape[2]))
        encoder_l1 = LSTM(latent_dim, return_state=True, return_sequences=True)
        encoder_outputs, state_h1, state_c1 = encoder_l1(encoder_inputs)
        encoder_l2 = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h2, state_c2 = encoder_l2(encoder_outputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h1, state_c1, state_h2, state_c2]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,
                                      self.data.decoder_input_data.shape[2]))
        # We set up our decoder to return full output sequences, and to return
        # internal states as well. We don't use the return states in the
        # training model, but we will use them in inference.
        decoder_l1 = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_l1(decoder_inputs,
                                           initial_state=[state_h1, state_c1])
        decoder_l2 = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_l2(decoder_outputs,
                                           initial_state=[state_h2, state_c2])
        decoder_dense = Dense(self.data.decoder_input_data.shape[2],
                              activation=act_func)
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn `encoder_input_data` &
        # `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='rmsprop', loss=loss_func)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # 3) Repeat with the current target token and current states

        # Define sampling models
        self.enc_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_state_input_h2 = Input(shape=(latent_dim,))
        decoder_state_input_c2 = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c,
                                 decoder_state_input_h2, decoder_state_input_c2]

        decoder_outputs, decoder_state_h1, decoder_state_c1 = \
            decoder_l1(decoder_inputs, initial_state=decoder_states_inputs[:2])
        decoder_outputs, decoder_state_h2, decoder_state_c2 = \
            decoder_l2(decoder_outputs, initial_state=decoder_states_inputs[-2:])
        decoder_states = [decoder_state_h1, decoder_state_c1,
                          decoder_state_h2, decoder_state_c2]
        decoder_outputs = decoder_dense(decoder_outputs)

        self.dec_model = Model([decoder_inputs] + decoder_states_inputs,
                               [decoder_outputs] + decoder_states)

    # Run training
    def fit(self, name, batch_sz=64, epochs=100):
        self.model.fit([self.data.encoder_input_data,
                        self.data.decoder_input_data],
                       self.data.decoder_target_data,
                       batch_size=batch_sz,
                       epochs=epochs,
                       validation_split=0.2)
        # Save model
        self.model.save(name)

# TODO: put the following into function / class defs
reverse_input_char_index = dict(
    (i, char) for char, i in dat.input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in dat.target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = s2s_char.enc_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, dat.decoder_input_data.shape[2]))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, dat.target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h1, c1, h2, c2 = s2s_char.dec_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, dat.decoder_input_data.shape[2]))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h1, c1, h2, c2]

    return decoded_sentence


correct = 0
for seq_index in range(640):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = dat.encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    if decoded_sentence == dat.target_texts[seq_index][1:]:
        correct += 1
#
#    print('-')
#    print('Input sentence:', input_texts[seq_index])
#    print('Decoded sentence:', decoded_sentence[:-2])
#    print('Correct sentence:', target_texts[seq_index][1:])

print('Accuracy:', correct/640)