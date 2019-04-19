import my_s2s_v2 as my

dat = my.data_prep("fra.txt")
dat.as_wrdvec()
mod_rev_vec = my.seq2seq(dat)
mod_rev_vec.reverse("tanh", "mean_squared_error", 900)
mod_rev_vec.fit("rev")
