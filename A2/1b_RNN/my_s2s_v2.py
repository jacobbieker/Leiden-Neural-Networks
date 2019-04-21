from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from gensim.models import KeyedVectors as w2v
from re import match, sub, DOTALL
from sklearn.metrics.pairwise import cosine_distances
from pickle import dump

'''
efr_wordvec_tanh_mean_squared_error: 29.559
rev_efr_wordvec_tanh_mean_squared_error: 27.473%
rev_ede_wordvec_tanh_mean_squared_error: 59.592%
'''


class data_prep():
    def __init__(self, txtfile, random_sample_flag, num_samples=10000):
        assert isinstance(random_sample_flag, bool), "flag should be boolean"
        self.data_path = txtfile
        self.num_samples = num_samples  # Number of samples to train on.
        self.random_sample = random_sample_flag
        self.input_texts = []
        self.target_texts = []
        # tokens are characters or word vectors depending on the tokeniser used
        self.input_token_index = 'undef; run tokeniser function'
        self.target_token_index = 'undef; run tokeniser function'
        self.encoder_input_data = 'undef; run tokeniser function'
        self.decoder_input_data = 'undef; run tokeniser function'
        self.decoder_target_data = 'undef; run tokeniser function'
        self.reverse_input_char_index = 'undef for word2vec'
        self.decode_index = 'undef for word2vec'
        self.en_stop = ['a', 'and', 'of', 'to']
        self.embed_type = None
        if txtfile == "fra.txt":
            self.target_lang = "fr"
        elif txtfile == "deu.txt":
            self.target_lang = "de"

    def as_char(self):
        np.random.seed(1)
        self.embed_type = "char"
        input_tokens = set()
        target_tokens = set()
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        sample_sz = min(self.num_samples, len(lines) - 1)
        if self.random_sample:
            sample = np.random.choice(lines, sample_sz, False)
        else:
            sample = lines[:sample_sz]
        for line in sample:
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
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.decode_index = dict(
            (i, char) for char, i in self.target_token_index.items())

    def load_w2v(self):
        en_path = './w2v_models/GoogleNews-vectors-negative300.bin'
        #en_path = './w2v_models/en_2e5.bin'
        fr_path = './w2v_models/fr_200_skip_cut100.bin'
        de_path = './w2v_models/german.model'
        if self.target_lang == "fr":
            target_path = fr_path
        elif self.target_lang == "de":
            target_path = de_path
        self.input_token_index = w2v.load_word2vec_format(
                en_path, binary=True, limit=int(2E5))
        self.target_token_index = w2v.load_word2vec_format(
                target_path, binary=True)

    def de_line_preproc(self, line):
        skip_line = 0  # boolean if true, then skip line
        input_text, target_text = line.split('\t')
        # simplifications
        if match('\$?[0-9]', line):  # skip numbers
            skip_line = 1
            return skip_line, None, None, None, None
        if "'" in input_text:  # skip abbrev english sentences
            skip_line = 1
            return skip_line, None, None, None, None
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
            if word not in self.input_token_index.vocab:
                word = word.lower()
            if word not in self.input_token_index.vocab:
                skip_line = 2
                unk_line_inputs.append((word, target_words))
            else:
                input_vecs = np.vstack(
                        (input_vecs, self.input_token_index.word_vec(word)))
        for word in target_words:
            if word == '':
                continue
            if word not in self.target_token_index.vocab:
                skip_line = 2
                unk_line_targets.append((word, target_words))
            else:
                vec = self.target_token_index.word_vec(word)
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

    def fr_line_preproc(self, line):
        skip_line = 0  # boolean if true, then skip line
        input_text, target_text = line.split('\t')
        # simplifications
        if match('\$?[0-9]', line):  # skip numbers
            skip_line = 1
            return skip_line, None, None, None, None
        if "'" in input_text:  # skip abbrev english sentences
            skip_line = 1
            return skip_line, None, None, None, None
        # cannot is not in en dict, whisky is spelt whiskey
        input_text = sub('cannot', 'can not', input_text)
        input_text = sub('whisky', 'whiskey', input_text)
        input_text = sub('[^\w ]', '', input_text)
        target_text = sub("[^\w \-']", '', target_text)
        # fr_mod is all lowercase and treats m' as a word
        target_text = sub("'", "' ", target_text).lower()
        target_text = sub("-", " ", target_text)
        target_text = sub("œ", "oe", target_text)
        target_text = sub(" *$", "", target_text)
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
            if word not in self.input_token_index.vocab:
                word = word.lower()
            if word not in self.input_token_index.vocab:
                skip_line = 2
                unk_line_inputs.append((word, target_words))
            else:
                input_vecs = np.vstack(
                        (input_vecs, self.input_token_index.word_vec(word)))
        for word in target_words:
            word = sub("[^\w']", '', word)  # remove punctuation
            if word == '':
                continue
            if word not in self.target_token_index.vocab:
                skip_line = 2
                unk_line_targets.append((word, target_words))
            else:
                vec = self.target_token_index.word_vec(word)
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
        np.random.seed(1)
        unk_input = []
        unk_target = []
        self.load_w2v()
        if self.target_lang == 'de':
            line_preproc = self.de_line_preproc
        elif self.target_lang == 'fr':
            line_preproc = self.fr_line_preproc

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        sample_sz = min(self.num_samples, len(lines) - 1)
        if self.random_sample:
            sample = np.random.choice(lines, sample_sz, False)
        else:
            sample = lines[:sample_sz]
        for line in sample:
            skip, input_vecs, target_vecs, unk_in, unk_out = line_preproc(line)
            if skip == 1:
                continue
            unk_input.extend(unk_in)
            unk_target.extend(unk_out)

        return unk_input, unk_target

    def as_wrdvec(self):
        np.random.seed(1)
        self.embed_type = "wordvec"
        self.load_w2v()
        if self.target_lang == 'de':
            line_preproc = self.de_line_preproc
        elif self.target_lang == 'fr':
            line_preproc = self.fr_line_preproc
        in_max_len = 0
        tar_max_len = 0
        in_vec_list = []
        tar_vec_list = []

        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        sample_sz = min(self.num_samples, len(lines) - 1)
        if self.random_sample:
            sample = np.random.choice(lines, sample_sz, False)
        else:
            sample = lines[:sample_sz]
        for line in sample:
            skip, input_vecs, target_vecs, unk_in, unk_out = line_preproc(line)
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
        self.act_func = None
        self.loss_func = None
        self.filename = None
        # self.enc_in_dat = data_prep.encoder_input_data
        # self.dec_in_dat = data_prep.decoder_input_data
        # self.dec_tar_dat = data_prep.decoder_target_data

    def twolayer(self, act_func, loss_func, latent_dim=256):
        # act_funcs: softmax, tanh, linear
        # loss_funcs: categorical_crossentropy, mean_squared_error
        # Define an input sequence and process it.
        self.act_func = act_func
        self.loss_func = loss_func
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

    def reverse(self, act_func, loss_func, latent_dim=256):
        zeros = np.all(self.data.encoder_input_data == 0, axis=2)
        for i, (sample, boo) in enumerate(
                zip(self.data.encoder_input_data, zeros)):
            self.data.encoder_input_data[i, boo, :] = sample[boo][::-1, :]
        self.twolayer(act_func, loss_func, latent_dim)

    def bidir(self, act_func, loss_func, latent_dim=256):
        zeros = np.all(self.data.encoder_input_data == 0, axis=2)
        for i, (sample, boo) in enumerate(
                zip(self.data.encoder_input_data, zeros)):
            self.data.encoder_input_data_rev[i, boo, :] = sample[boo][::-1, :]
        self.twolayer(act_func, loss_func, latent_dim)

    def twolang(self, act_func, loss_func, latent_dim=256):
        # act_funcs: softmax, tanh, linear
        # loss_funcs: categorical_crossentropy, mean_squared_error
        # Define an input sequence and process it.
        self.act_func = act_func
        self.loss_func = loss_func
        encoder_inputs = Input(shape=(None,
                                      self.data.encoder_input_data.shape[2]))
        encoder_l1 = LSTM(latent_dim, return_state=True, return_sequences=True)
        encoder_outputs, state_h1, state_c1 = encoder_l1(encoder_inputs)
        encoder_l2 = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h2, state_c2 = encoder_l2(encoder_outputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h1, state_c1, state_h2, state_c2]

        # Set up the decoder, using `encoder_states` as initial state.
        frdecoder_inputs = Input(shape=(None,
                                      self.data.decoder_input_data.shape[2]))
        frdecoder_l1 = LSTM(latent_dim, return_sequences=True, return_state=True)
        frdecoder_outputs, _, _ = frdecoder_l1(frdecoder_inputs,
                                           initial_state=[state_h1, state_c1])
        frdecoder_l2 = LSTM(latent_dim, return_sequences=True, return_state=True)
        frdecoder_outputs, _, _ = frdecoder_l2(frdecoder_outputs,
                                           initial_state=[state_h2, state_c2])
        frdecoder_dense = Dense(self.data.decoder_input_data.shape[2],
                              activation=act_func)
        frdecoder_outputs = frdecoder_dense(frdecoder_outputs)

        dedecoder_inputs = Input(shape=(None,
                                      self.data.decoder_input_data.shape[2]))
        dedecoder_l1 = LSTM(latent_dim, return_sequences=True, return_state=True)
        dedecoder_outputs, _, _ = dedecoder_l1(dedecoder_inputs,
                                           initial_state=[state_h1, state_c1])
        dedecoder_l2 = LSTM(latent_dim, return_sequences=True, return_state=True)
        dedecoder_outputs, _, _ = dedecoder_l2(dedecoder_outputs,
                                           initial_state=[state_h2, state_c2])
        dedecoder_dense = Dense(self.data.decoder_input_data.shape[2],
                              activation=act_func)
        dedecoder_outputs = dedecoder_dense(frdecoder_outputs)

        # Define the model that will turn `encoder_input_data` &
        # `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, frdecoder_inputs, dedecoder_inputs],
                           [frdecoder_outputs, dedecoder_outputs])
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
    def fit(self, pre, batch_sz=64, epochs=100):
        self.model.fit([self.data.encoder_input_data,
                        self.data.decoder_input_data],
                       self.data.decoder_target_data,
                       batch_size=batch_sz,
                       epochs=epochs,
                       validation_split=0.2)
        self.filename = pre + "_e" + self.data.target_lang + "_" + \
            self.data.embed_type + "_" + self.act_func + "_" + self.loss_func
        # Save model
        self.model.save_weights(self.filename + ".h5")

    def loadwts(self, filename):
        self.model.load_weights(filename)
        self.filename = sub("\.h5", "", filename)

    def decode_sequence(self, input_seq):
        assert len(input_seq.shape) == 3, "input sequence should be 3 dimensional"
        vec_sol = np.append([1, 0], np.zeros(200)).reshape((1, -1))
        vec_eol = np.append([0, 1], np.zeros(200)).reshape((1, -1))

        # Encode the input as state vectors.
        states_value = self.enc_model.predict(input_seq)

        # Generate empty target sequences of length 1.
        n = input_seq.shape[0]
        target_seq = np.zeros((n, 1, self.data.decoder_input_data.shape[2]))
        # Populate the first character of target sequence with the s.o.l. char.
        if self.data.embed_type == 'char':
            target_seq[:, 0, self.data.target_token_index['\t']] = 1.
        elif self.data.embed_type == 'wordvec':
            target_seq[:, 0, 0] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        # stop_condition = False
        decoded_sentences = ["" for i in range(n)]
        for i in range(self.data.decoder_input_data.shape[1]):
            output_tokens, h1, c1, h2, c2 = self.dec_model.predict(
                [target_seq] + states_value)
            target_seq.fill(0)
            states_value = [h1, c1, h2, c2]

            if self.data.embed_type == 'char':
                # Sample a token
                sampled_token_index = list(
                        np.argmax(output_tokens, 2).flatten())
                target_seq[list(range(n)), 0, sampled_token_index] = 1.
                decoded_sentences = [a + self.data.decode_index[b] for (a, b) in
                                    zip(decoded_sentences, sampled_token_index)]
            elif self.data.embed_type == 'wordvec':
                for j in range(n):
                    pred_vec = output_tokens[j, :, :]
                    sim_word = self.data.target_token_index.similar_by_vector(
                            pred_vec[0, 2:], 1)[0][0]
                    sim_vec = self.data.target_token_index.word_vec(sim_word)
                    sim_vec = np.pad(sim_vec, (2, 0), 'constant').reshape(1, -1)
                    vec_list = [sim_vec, vec_sol, vec_eol]
                    word_list = [" " + sim_word, "\t", "\n"]
                    arg = np.argmin(
                            [cosine_distances(pred_vec, v) for v in vec_list])
                    target_seq[j, :, :] = vec_list[arg]
                    decoded_sentences[j] += word_list[arg]
        for k, sen in enumerate(decoded_sentences):
            sen = sen[1:]
            decoded_sentences[k] = sub("\n.*", "", sen, flags=DOTALL)

        return decoded_sentences, n

    def train_acc(self, beg=0, end=np.inf):
        end = min(end, len(self.data.input_texts))
        correct = 0
        incorrects = []
        preds, n = self.decode_sequence(self.data.encoder_input_data[beg:end])
        for (truth, pred) in zip(self.data.target_texts, preds):
            if truth[1:-1] == pred:
                correct += 1
            else:
                incorrects.append((truth[1:-1], pred))
        print("Accuracy:", correct/n)
        with open(self.filename + '.list', 'wb') as f:
            dump([correct/n] + incorrects, f)

        return correct/n, incorrects
