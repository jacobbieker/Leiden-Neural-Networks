import my_s2s_v2_noGenSim
'''
list models: [{plain, rev}latent_dim, act_func, loss_func, latent_dim]
'''

models=[["plain256", "softmax", "categorical_crossentropy", 256],
        ["plain512", "softmax", "categorical_crossentropy", 512]]

# fr loop
for model in models:
    dat = data_prep("fra.txt", True)
    dat.as_char()
    mod = seq2seq(frdat)
    mod.twolayer(model[1], model[2], model[3])
    mod.fit(model[0])
    a, i = mod.train_acc()

# de loop
for model in models:
    dat = data_prep("deu.txt", True)
    dat.as_char()
    mod = seq2seq(frdat)
    mod.twolayer(model[1], model[2], model[3])
    mod.fit(model[0])
    a, i = mod.train_acc()