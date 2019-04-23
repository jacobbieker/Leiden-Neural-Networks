import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import Twolang_s2s_v2 as my
from keras import backend as K
import os

act_functions = ["tanh", "linear"]
losses = ["cosine", "mse", "mae", "logcosh"]

dat = my.data_prep(txtfile="fra2lang4.5.txt", de_txtfile="deu2lang4.5.txt", random_sample_flag=True)
dat.as_wrdvec()
for act in act_functions:
    for loss in losses:
        mod_rev_vec = my.seq2seq(dat)
        mod_rev_vec.reverse(act, loss, 900)
        mod_rev_vec.fit("rev")
        K.clear_session()
