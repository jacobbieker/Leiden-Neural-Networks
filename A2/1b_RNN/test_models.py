import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import my_s2s_v2 as my
from keras import backend as K
import sys
import os


orig_stdout = sys.stdout
f = open('Both_fra_test.txt', 'a')
sys.stdout = f
'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''

act_functions = ["linear"]
losses = ["logcosh"]
paths = ["deu.txt"]
ways = ["for", "rev"]
dat = my.data_prep("deu.txt", True)
dat.as_wrdvec()
for act in act_functions:
    for loss in losses:
        for data in paths:
            for way in ways:
                if not os.path.isfile("{}_e{}_wordvec_{}_{}.list".format(way, data.split(".")[0][:-1], act, loss)):
                    mod = my.seq2seq(dat)
                    if way == "for":
                        mod.twolayer(act, loss, 900)
                    else:
                        mod.reverse(act, loss, 900)
                    if data == "fra.txt":
                        pre = "fr"
                    else:
                        pre = "de"
                    mod.loadwts("{}_e{}_wordvec_{}_{}.h5".format(way, pre, act, loss))
                    print("\n\n{}_e{}_wordvec_{}_{}.h5".format(way, pre, act, loss))
                    a, i = mod.train_acc()
                    print("{}_e{}_wordvec_{}_{}.h5\n\n".format(way, pre, act, loss))
                    K.clear_session()
                else:
                    print("Skipped")
