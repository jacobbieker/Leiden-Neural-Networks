import tensorflow as tf
import my_s2s_v2 as my
from keras import backend as K

act_functions = ["tanh", "linear"]
losses = ["cosine", "mse", "mae", "logcosh"]
paths = ["fra.txt", "deu.txt"]

for act in act_functions:
    for loss in losses:
        for data in paths:
            dat = my.data_prep(data, random_sample_flag=True, num_samples=10000)
            dat.as_wrdvec()
            mod_rev_vec = my.seq2seq(dat)
            mod_rev_vec.reverse(act, loss, 900)
            mod_rev_vec.fit("rev")
            K.clear_session()
