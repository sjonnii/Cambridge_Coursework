from Corpora import MovieReviewCorpus
from Lexicon import SentimentLexicon
from Statistics import SignTest
from Classifiers import NaiveBayesText, SVMText
from Extensions import SVMDoc2Vec
from collections import Counter
import gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from subprocess import call
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# # retrieve corpus
corpus=MovieReviewCorpus(stemming=False,pos=False)
# # print corpus
# # # print corpus.train

# # # #use sign test for all significance testing
signTest=SignTest()

# # # # # #location of svmlight binaries 
svmlight_dir="/Users/sigurjonisaksson/Documents/NLP/svm_light/"

print "--- classifying reviews using sentiment lexicon  ---"

#read in lexicon
lexicon=SentimentLexicon()

# on average there are more positive than negative words per review (~7.13 more positive than negative per review)
# to take this bias into account will use threshold (roughly the bias itself) to make it harder to classify as positive
threshold=8
threshold_magn = 8
#------------------------------------------
#------------------------------------------
# question 0.1
lexicon.classify(corpus.reviews,threshold,magnitude=False)
token_preds=lexicon.predictions
print "token-only results: %.2f" % lexicon.getAccuracy()

lexicon.classify(corpus.reviews,threshold_magn,magnitude=True)
magnitude_preds=lexicon.predictions
print "magnitude results: %.2f" % lexicon.getAccuracy()

#------------------------------------------
#------------------------------------------
# question 0.2
p_value=signTest.getSignificance(token_preds,magnitude_preds)
print "magnitude lexicon results are",("significant" if p_value < 0.05 else "not significant"),"with respect to token-only","(p=%.8f)" % p_value

#------------------------------------------
#------------------------------------------
#question 1.0
print "--- classifying reviews using Naive Bayes on held-out test set ---"
NB=NaiveBayesText(smoothing=False,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
#store predictions from classifier
non_smoothed_preds=NB.predictions
print "Accuracy without smoothing: %.3f" % NB.getAccuracy()

#------------------------------------------
#------------------------------------------
# #question 2.0
# #use smoothing
NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
NB.test(corpus.test)
smoothed_preds=NB.predictions
# # saving this for use later
num_non_stemmed_features=len(NB.vocabulary)

print "Accuracy using smoothing: %.3f" % NB.getAccuracy()

#------------------------------------------
#------------------------------------------
#uestion 2.1
#see if smoothing significantly improves results
p_value=signTest.getSignificance(non_smoothed_preds,smoothed_preds)
print "results using smoothing are",("significant" if p_value < 0.05 else "not significant"),"with respect to no smoothing","(p=%.8f)" % p_value

#------------------------------------------
#------------------------------------------
#question 3.0
print "--- classifying reviews using 10-fold cross-evaluation ---"
#using previous instantiated object
#using cross-eval for smoothed predictions from now on
Accuracy = []
Std = []
total_notstemmed = []
for i in range(10):
	NB.crossValidate(corpus,i)
	smoothed_preds=NB.predictions
	total_notstemmed = total_notstemmed + smoothed_preds
	Accuracy.append('%.2f' %NB.getAccuracy())
	Std.append('%.2f' %NB.getStdDeviation())

print Accuracy
print Std
print sum(map(float,Accuracy))/len(Accuracy)


#------------------------------------------
#------------------------------------------
# question 4.0
print "--- stemming corpus ---"
# retrieve corpus with tokenized text and stemming (using porter)
stemmed_corpus=MovieReviewCorpus(stemming=True,pos=False)
NB2=NaiveBayesText(smoothing=True,bigrams=False,trigrams=False,discard_closed_class=False)
print "--- cross-validating NB using stemming ---"
Accuracy_stemm = []
Std_Stemm = []

total_stemmed = []
for i in range(10):
	NB2.crossValidate(stemmed_corpus,i)
	stemmed_preds=NB2.predictions
	total_stemmed = total_stemmed + stemmed_preds
	Accuracy_stemm.append('%.5f' %NB2.getAccuracy())
	Std_Stemm.append('%.5f' %NB2.getStdDeviation())

print Accuracy_stemm
print Std_Stemm
print sum(map(float,Accuracy_stemm))/len(Accuracy_stemm)

#------------------------------------------
#------------------------------------------
# # # # Q4.1
p_value=signTest.getSignificance(total_notstemmed,total_stemmed)
print len(total_notstemmed)
print len(total_stemmed)
print "results using non stemmed are",("significant" if p_value < 0.05 else "not significant"),"with respect to stemmed","(p=%.8f)" % p_value
#Q4.2
print "--- determining the number of features before/after stemming ---"
num_stemmed_features = len(NB2.vocabulary)
print 'num_non_stemmed_features:'
print num_non_stemmed_features
print 'num_stemmed_features'
print num_stemmed_features


#------------------------------------------
#------------------------------------------
# #question Q5.0
#cross-validate model using smoothing and bigrams
print "--- cross-validating naive bayes using smoothing and bigrams ---"
NB=NaiveBayesText(smoothing=True,bigrams=True,trigrams=False,discard_closed_class=False)
NB.train(corpus.train)
bigram_acc = []
bigram_std = []
total_bigram = []
for i in range(10):
	NB.crossValidate(corpus,i)
	smoothed_and_bigram_preds=NB.predictions
	total_bigram = total_bigram + smoothed_and_bigram_preds
	bigram_acc.append('%.2f' %NB.getAccuracy())
	bigram_std.append('%.2f' %NB.getStdDeviation())

print bigram_acc
print bigram_std
print sum(map(float,bigram_acc))/len(bigram_std)

# see if bigrams significantly improves results on smoothed NB only
p_value=signTest.getSignificance(total_notstemmed, total_bigram)
print "results using smoothing and bigrams are",("significant" if p_value < 0.05 else "not significant"),"with respect to smoothing only","(p=%.8f)" % p_value

#------------------------------------------
#------------------------------------------
# # # # Q5.1

print '## featurs bigram'

bigram_features = len(NB.vocabulary)
print 'num_non_stemmed_features:'
print num_non_stemmed_features
print 'num_bigram features'
print bigram_features

NB=NaiveBayesText(smoothing=True,bigrams=False,trigrams=True,discard_closed_class=False)
NB.train(corpus.train)
trigram_features = len(NB.vocabulary)
print 'num_trigram features'
print trigram_features


#------------------------------------------
#------------------------------------------
#Q6 and 6.1
print "--- classifying reviews using SVM 10-fold cross-eval ---"
SVM = SVMText(bigrams = False, trigrams = False, svmlight_dir = svmlight_dir, discard_closed_class = False)
##SVM2 = SVMText(bigrams = True, trigrams = False, svmlight_dir = svmlight_dir, discard_closed_class = False)

accr = []
stdv = []
total_svm = []
# svm_train = SVM.train(stemmed_corpus.train)
# svm_test = SVM.test(stemmed_corpus.test)
for i in range(10):
	SVM.crossValidate(stemmed_corpus,i)
	svm_train = SVM.predictions
	total_svm = total_svm + svm_train
	accr.append('%.2f' %SVM.getAccuracy())
	stdv.append('%.2f' %SVM.getStdDeviation())

print accr
print stdv
print sum(map(float,accr))/len(accr)

p_value=signTest.getSignificance(total_notstemmed,total_svm)
print "results using Naive Bayes are",("significant" if p_value < 0.05 else "not significant"),"with respect to SVM","(p=%.8f)" % p_value



#------------------------------------------
#------------------------------------------
# Q7
print "--- adding in POS information to corpus ---"
print "--- training svm on word+pos features ----"
tag_corpus=MovieReviewCorpus(stemming=True,pos=True)
SVM2 = SVMText(bigrams = False, trigrams = False, svmlight_dir = svmlight_dir, discard_closed_class = False)


accr_tag = []
stdv_tag = []
tag_total = []
for i in range(10):
	SVM2.crossValidate(tag_corpus,i)
	svm_train_tag = SVM2.predictions
	tag_total = tag_total + svm_train_tag
	accr_tag.append('%.2f' %SVM2.getAccuracy())
	stdv_tag.append('%.2f' %SVM2.getStdDeviation())

print accr_tag
print stdv_tag
print sum(map(float,accr_tag))/len(accr_tag)

print "--- training svm discarding closed-class words ---"

p_value=signTest.getSignificance(tag_total,total_svm)
print "results using Tags are",("significant" if p_value < 0.05 else "not significant"),"with respect to not tags","(p=%.8f)" % p_value

SVM3 = SVMText(bigrams = False, trigrams = False, svmlight_dir = svmlight_dir, discard_closed_class = True)
accr_tagdisc = []
stdv_tagdisc = []
tagdisc_total = []
for i in range(10):
	SVM3.crossValidate(tag_corpus,i)
	svm_train_tagdisc = SVM3.predictions
	tagdisc_total = tagdisc_total + svm_train_tagdisc
	accr_tagdisc.append('%.2f' %SVM3.getAccuracy())
	stdv_tagdisc.append('%.2f' %SVM3.getStdDeviation())

print accr_tagdisc
print stdv_tagdisc
print sum(map(float,accr_tagdisc))/len(accr_tagdisc)

p_value=signTest.getSignificance(tagdisc_total,total_svm)
print "results using Tags and discard are",("significant" if p_value < 0.05 else "not significant"),"with respect to no tags","(p=%.8f)" % p_value


#------------------------------------------
#------------------------------------------
#question 8.0

model = '50_20_dbow_hier/trained.model'
svm = SVMDoc2Vec(model, svmlight_dir = svmlight_dir)

total_accr = []
total_std = []
total_compare = []
for i in range(10):
	svm.crossValidate(stemmed_corpus,i)
	doc2_preds = svm.predictions
	total_compare = total_compare + doc2_preds
	total_accr.append('%.2f' %svm.getAccuracy())
	total_std.append('%.2f' %svm.getStdDeviation())

print total_accr
print total_std
print sum(map(float,total_accr))/len(total_accr)

p_value=signTest.getSignificance(total_notstemmed,total_compare)
print "results using Naive Bayes are",("significant" if p_value < 0.05 else "not significant"),"with respect to doc2vec","(p=%.8f)" % p_value

model = '50_20_dbow_hier/trained.model'
svm = SVMDoc2Vec(model, svmlight_dir = svmlight_dir)
model = svm.model
svm.crossValidate(stemmed_corpus,0)


words_np = [model.docvecs[1],model.docvecs[222],model.docvecs[333],model.docvecs[45003],model.docvecs[45103],model.docvecs[32000]]
words_label = ['NEG','NEG','NEG','POS','POS','POS']
pca = PCA(n_components=2)
pca.fit(words_np)
reduced = pca.transform(words_np)
for index,vec in enumerate(reduced):
    # print ('%s %s'%(words_label[index],vec))
    x,y=vec[0],vec[1]
    plt.scatter(x,y)
    plt.annotate(words_label[index],xy=(x,y))
plt.show()


