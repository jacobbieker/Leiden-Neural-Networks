#import keras
import numpy as np

"""

Visualizations:

Parameters Space of the Activations based on the inputs (characters? or sentences?)
https://github.com/asap-report/lstm-visualisation/blob/master/lstm-visualisation.ipynb

PCA then TNSE maybe?
https://becominghuman.ai/visualizing-representations-bd9b62447e38

Mappings of embeddings to outputs?
https://medium.com/datalogue/attention-in-keras-1892773a4f22


Compare Activations of mistranslated vs not mistranslated inputs

Compare Activations of single language translator vs dual language translator

Single neuron activations
http://blog.echen.me/2017/05/30/exploring-lstms/

Visualize internal states of the network?
As in what is returned from return_state?

How the network is workng:
https://towardsdatascience.com/neural-machine-translation-using-seq2seq-with-keras-c23540453c74


"""
flines = None
findex = []
dlines = None
dindex = []
eng_words = []
de_eng_words = []

with open("fra.txt", "r") as french:
    flines = french.read().split('\n')
    for index, element in enumerate(flines):
        eng_words.append(element.split('\t')[0])
with open("deu.txt", "r") as german:
    dlines = german.read().split('\n')
    for index, element in enumerate(dlines):
        de_eng_words.append(element.split('\t')[0])

same_english = set(eng_words).intersection(de_eng_words)
print(len(same_english))

with open("fra2lang.txt", "w") as f:
    with open("deu2lang.txt", "w") as g:
        for line in same_english:
            for element in dlines:
                if element.split("\t")[0] in line:
                    g.write(element + "\n")
                    break
            for element in flines:
                if element.split("\t")[0] in line:
                    f.write(element + "\n")
                    break


