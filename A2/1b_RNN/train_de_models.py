import my_s2s_v2_noGenSim as my
'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''

models=[["plain256", "softmax", "categorical_crossentropy", 256],
        ["plain512", "softmax", "categorical_crossentropy", 512]]

# de loop
for model in models:
    dat = my.data_prep("deu.txt", True)
    dat.as_char()
    mod = my.seq2seq(dat)
    mod.twolayer(model[1], model[2], model[3])
    mod.fit(model[0])
    a, i = mod.train_acc()