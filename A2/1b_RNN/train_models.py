import my_s2s_v2_noGenSim as my
'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''
dat = my.data_prep("fra.txt", True)
dat.as_char()
mod = my.seq2seq(dat)
mod.twolayer("softmax", "categorical_crossentropy", 256)
mod.loadwts("plain256_efr_char_softmax_categorical_crossentropy.h5")
a, i = mod.train_acc()

models=[["plain512", "softmax", "categorical_crossentropy", 512]]

# fr loop
for model in models:
    dat = my.data_prep("fra.txt", True)
    dat.as_char()
    mod = my.seq2seq(dat)
    mod.twolayer(model[1], model[2], model[3])
    mod.fit(model[0])
    a, i = mod.train_acc()
