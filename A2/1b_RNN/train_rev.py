import tensorflow as tf
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import my_s2s_v2 as my
from keras import backend as K

act_functions = ["tanh", "linear"]
losses = ["cosine", "mse", "mae", "logcosh"]
paths = ["fra.txt", "deu.txt"]

for act in act_functions:
    for loss in losses:
        for data in paths:
            dat = my.data_prep(data, random_sample_flag=True)
            dat.as_wrdvec()
            mod_rev_vec = my.seq2seq(dat)
            mod_rev_vec.twolayer(act, loss, 900)
            mod_rev_vec.fit("for")
            K.clear_session()
