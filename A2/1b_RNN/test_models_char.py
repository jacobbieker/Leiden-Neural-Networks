import my_s2s_v2_noGenSim as my
'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''
models = ['plain256_efr_char_softmax_categorical_crossentropy.h5',
          'plain512_efr_char_softmax_categorical_crossentropy.h5']
dim = [256, 512]

for mod, d in zip(models, dim):
    dat = my.data_prep("fra.txt", True)
    dat.as_char()
    mod = my.seq2seq(dat)
    mod.twolayer("softmax", "categorical_crossentropy", d)
    mod.loadwts(mod)
    a, i = mod.train_acc()

models = ['plain256_ede_char_softmax_categorical_crossentropy.h5',
          'plain512_ede_char_softmax_categorical_crossentropy.h5']

for mod, d in zip(models, dim):
    dat = my.data_prep("deu.txt", True)
    dat.as_char()
    mod = my.seq2seq(dat)
    mod.twolayer("softmax", "categorical_crossentropy", d)
    mod.loadwts(mod)
    a, i = mod.train_acc()
