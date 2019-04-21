import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import my_s2s_v2 as my
'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''

act_functions = ["tanh", "linear"]
losses = ["cosine", "mse", "mae", "logcosh"]
paths = ["fra.txt", "deu.txt"]

for act in act_functions:
    for loss in losses:
        for data in paths:
            dat = my.data_prep(data, True)
            dat.as_wrdvec()
            mod = my.seq2seq(dat)
            mod.twolayer(act, loss, 900)
            if data == "fra.txt":
                pre = "fr"
            else:
                pre = "de"
            mod.loadwts("rev_e{}_wordvec_{}_{}.h5".format(pre, act, loss))
            print("\n\nrev_e{}_wordvec_{}_{}.h5".format(pre, act, loss))
            a, i = mod.train_acc()
            print("rev_e{}_wordvec_{}_{}.h5\n\n".format(pre, act, loss))
