import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
import my_s2s_v2 as my

dat = my.data_prep("deu.txt")
dat.as_wrdvec()
mod_rev_vec = my.seq2seq(dat)
mod_rev_vec.reverse("tanh", "mean_squared_error", 900)
mod_rev_vec.fit("rev")
