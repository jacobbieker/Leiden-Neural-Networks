import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import Twolang_s2s_v2 as my
from keras import backend as K
import sys
import os

'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''

act_functions = ["tanh", "linear"]
losses = ["mse", "mae", "logcosh"]
ways = ['for', 'rev']
dat = my.data_prep(txtfile="fra2lang4.5.txt", de_txtfile="deu2lang4.5.txt", random_sample_flag=True)
dat.as_wrdvec()
for act in act_functions:
    for loss in losses:
            for way in ways:
                if not os.path.isfile("{}_e{}_wordvec_{}_{}.list".format(way, 'frde', act, loss)):
                    mod = my.seq2seq(dat)
                    mod.twolang(act, loss, 900)
                    pre = "frde"
                    mod.loadwts("{}_e{}_wordvec_{}_{}.h5".format(way, pre, act, loss))
                    print("\n\n{}_e{}_wordvec_{}_{}.h5".format(way, pre, act, loss))
                    a, i = mod.train_acc()
                    print("{}_e{}_wordvec_{}_{}.h5\n\n".format(way, pre, act, loss))
                    K.clear_session()
                else:
                    print("Skipped")
