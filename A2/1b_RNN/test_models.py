import my_s2s_v2_noGenSim as my
'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''
dat = my.data_prep("fra.txt", True)
dat.as_char()
mod = my.seq2seq(dat)
mod.twolayer("softmax", "categorical_crossentropy", 512)
mod.loadwts("plain512_efr_char_softmax_categorical_crossentropy.h5")
a, i = mod.train_acc()