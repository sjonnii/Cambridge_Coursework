import gensim
#import load
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from random import shuffle
import os
import re
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from os import listdir
from os.path import isfile, join
from unidecode import unidecode
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# reviews1 = []
# reviews2 = []
# reviews3 = []
# reviews4 = []
# reviews5 = []
# reviews1 = [f for f in listdir("/Users/sigurjonisaksson/Documents/NLP/aclImdb/train/neg") if f.endswith('.txt')]
# reviews2 = [f for f in listdir("/Users/sigurjonisaksson/Documents/NLP/aclImdb/train/pos") if f.endswith('.txt')]
# reviews3 = [f for f in listdir("/Users/sigurjonisaksson/Documents/NLP/aclImdb/train/unsup") if f.endswith('.txt')]
# reviews4 = [f for f in listdir("/Users/sigurjonisaksson/Documents/NLP/aclImdb/test/neg") if f.endswith('.txt')]
# reviews5 = [f for f in listdir("/Users/sigurjonisaksson/Documents/NLP/aclImdb/test/pos") if f.endswith('.txt')]

# # train_pos = []
# # train_neg = []
# # test_pos = []
# # test_neg = []
# # unsup = []
# data = []

# for doc in reviews1:
# 	text_file = open('/Users/sigurjonisaksson/Documents/NLP/aclImdb/train/neg/' + doc, 'r').read()
# 	data.append(text_file)
# # #print len(data_pos)

# for doc in reviews2:
# 	text_file = open('/Users/sigurjonisaksson/Documents/NLP/aclImdb/train/pos/' + doc, 'r').read()
# 	data.append(text_file)

# for doc in reviews3:
#     text_file = open('/Users/sigurjonisaksson/Documents/NLP/aclImdb/train/unsup/' + doc, 'r').read()
#     data.append(text_file)

# for doc in reviews4:
#     text_file = open('/Users/sigurjonisaksson/Documents/NLP/aclImdb/test/neg/' + doc, 'r').read()
#     data.append(text_file)

# for doc in reviews5:
#     text_file = open('/Users/sigurjonisaksson/Documents/NLP/aclImdb/test/pos/' + doc, 'r').read()
#     data.append(text_file)


# outfile = open("imdb_big.txt", "w")
# print >> outfile, "\n".join(str(i) for i in data)
# outfile.close()

en_stop = get_stop_words('en')
p_stemmer = PorterStemmer()
taggeddoc = []
with utils.smart_open('imdb_big.txt') as fin:
    for item_no, line in enumerate(fin):
        #everything to lower
        line = line.lower()
        #make every string unicode
        line =  utils.to_unicode(line).split()
        #remove stop words
        line = [i for i in line if not i in en_stop]
        #remove numbers
        line = [p_stemmer.stem(i) for i in line]

        taggeddoc.append(TaggedDocument(line, ['%s' % item_no]))


# taggeddoc = []
# with utils.smart_open('imdb_big.txt') as fin:
#     for item_no, line in enumerate(fin):
#         taggeddoc.append(TaggedDocument(utils.to_unicode(line).split(), ['%s' % item_no]))

#model = Doc2Vec(alpha = 0.001, min_alpha = 0.001, size=100, hs = 1, dm = 0, dbow_words = 1) #good model
model = Doc2Vec(alpha = 0.001, min_alpha = 0.001, size=50, dm = 0,hs = 1, dbow_words = 1)
model.build_vocab(taggeddoc)

docvecs = model.docvecs
print (len(docvecs))

print '-start training'

for epoch in range(20):
  print epoch
  model.train(taggeddoc,total_examples=model.corpus_count,epochs=model.iter)  # decrease the learning rate
  model.min_alpha = model.alpha  # fix the learning rate, no decay


model.save('50_20_dbow_hier_words/trained.model')
model.save_word2vec_format('50_20_dbow_hier_words/trained.word2vec')





# print 'plotting'

# def plotWords(d2v):
#     #get model, we use w2v only

#     words_np = []
#     #a list of labels (words)
#     words_label = []

#     for i in range (25000):
# 		words_np.append(d2v[i])
# 		words_label.append(i)
 
#     pca = PCA(n_components=2)
#     pca.fit(words_np)
#     reduced= pca.transform(words_np)
 
#     # plt.plot(pca.explained_variance_ratio_)
#     for index,vec in enumerate(reduced):
#         # print ('%s %s'%(words_label[index],vec))
#         x,y=vec[0],vec[1]
#         plt.scatter(x,y)
#         plt.annotate(words_label[index],xy=(x,y))
#     plt.show()

# #plotWords(docvecs)





