import os
from subprocess import call
from nltk.util import ngrams
from Analysis import Evaluation
from collections import Counter
import numpy as np

class NaiveBayesText(Evaluation):
    def __init__(self,smoothing,bigrams,trigrams,discard_closed_class):
        """
        initialisation of NaiveBayesText classifier.

        @param smoothing: use smoothing?
        @type smoothing: booleanp

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        # set of features for classifier
        self.vocabulary=set()
        # prior probability
        self.prior={}
        # conditional probablility
        self.condProb={}
        # use smoothing?
        self.smoothing=smoothing
        # add bigrams?
        self.bigrams=bigrams
        # add trigrams?
        self.trigrams=trigrams
        # restrict unigrams to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class
        # stored predictions from test instances
        self.predictions=[]

    def extractVocabulary(self,reviews):
        """
        extract features from training data and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review):
                self.vocabulary.add(token)

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for token in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(token)==2 and self.discard_closed_class:
                if token[1][0:2] in ["NN","JJ","RB","VB"]: text.append(token)
            else:
                text.append(token)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(bigram)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(trigram)
        return text

    def train(self,reviews):
        """
        train NaiveBayesText classifier.

        1. reset self.vocabulary, self.prior and self.condProb
        2. extract vocabulary (i.e. get features for training)
        3. get prior and conditional probability for each label ("POS","NEG") and store in self.prior and self.condProb
           note: to get conditional concatenate all text from reviews in each class and calculate token frequencies
                 to speed this up simply do one run of the movie reviews and count token frequencies if the token is in the vocabulary,
                 then iterate the vocabulary and calculate conditional probability (i.e. don't read the movie reviews in their entirety 
                 each time you need to calculate a probability for each token in the vocabulary)

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # TODO Q1    
        # TODO Q2 (use switch for smoothing from self.smoothing)
        self.extractVocabulary(reviews)
        self.prior = {}
        self.condProb = {}

        N_POS = 0
        N_NEG = 0
        #count Docs in each class
        for i in range(len(reviews)):
            if reviews[i][0] == 'POS':
                N_POS +=1
            elif reviews[i][0] == 'NEG':
                N_NEG += 1
            else:
                'skrytid'
        #calculate prior
        self.prior['POS'] = np.log(float(N_POS)/len(reviews))
        self.prior['NEG'] = np.log(float(N_NEG)/len(reviews))

        pos_corp_train = []
        neg_corp_train = []
        #concatenate docs in each class
        for i in range(len(reviews)):
            if reviews[i][0] == 'POS':
                for token in reviews[i][1]:
                    pos_corp_train.append(token)
            elif reviews[i][0] == 'NEG':
                for token in reviews[i][1]:
                    neg_corp_train.append(token)
            else:
                print 'weird'

        #Count how often tokens come up for both classes
        pos_counts = {}
        neg_counts = {}

        # for words in self.vocabulary:
        #     pos_counts[words] = pos_corp_train.count(words)
        #     neg_counts[words] = neg_corp_train.count(words) 

        pos_counts = Counter(pos_corp_train)
        neg_counts = Counter(neg_corp_train)

        if self.smoothing:
            for sentiment, review in reviews:
                for token in review:
                    self.condProb[token+'-POS'] = np.log(pos_counts[token]+1)-np.log(len(pos_corp_train)+len(self.vocabulary))
                    self.condProb[token+'-NEG'] = np.log(neg_counts[token]+1)-np.log(len(neg_corp_train)+len(self.vocabulary))
        else:
            for sentiment, review in reviews:
                for token in review:
                    self.condProb[token+'-POS'] = np.log(pos_counts[token])-np.log(len(pos_corp_train))
                    self.condProb[token+'-NEG'] = np.log(neg_counts[token])-np.log(len(neg_corp_train))



    def test(self,reviews):
        """
        test NaiveBayesText classifier and store predictions in self.predictions.
        self.predictions should contain a "+" if prediction was correct and "-" otherwise.
        
        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        #if self.smoothing:
        for sentiment, review in reviews:
            pos_prob = self.prior['POS']
            neg_prob = self.prior['NEG']
            predict = ''
            for token in review:
                try:
                    pos_prob += self.condProb[token+'-POS']
                    neg_prob += self.condProb[token+'-NEG']
                except:
                    neg_prob += 0
                    pos_prob += 0

            if pos_prob >= neg_prob:
                #self.predictions.append('+')
                predict = 'POS'
            elif pos_prob < neg_prob:
                #self.predictions.append('-')
                predict = 'NEG'
            else:
                print 'weird'
            if predict == sentiment:
                self.predictions.append('+')
            else:
                self.predictions.append('-')

class SVM(Evaluation):
    """
    general svm class to be extended by text-based classifiers.
    """
    def __init__(self,svmlight_dir):
        self.predictions=[]
        self.svmlight_dir=svmlight_dir

    def writeFeatureFile(self,data,filename):
        """
        write local file in svmlight data format.
        see http://svmlight.joachims.org/ for description of data format.

        @param data: input data
        @type data: list of (string, list) tuples where string is the label and list are features in (id, value) tuples

        @param filename: name of file to write
        @type filename: string
        """
        # TODO Q6.0

        with open(self.svmlight_dir+filename, 'w') as f:

            for label, values in data:
                f.write('%s ' % label)
                for tup in values:
                    f.write('%s'':''%s ' % tup)
                f.write('\n')




    def train(self,train_data):
        """
        train svm 

        @param train_data: training data 
        @type train_data: list of (string, list) tuples corresponding to (label, content)
        """
        # function to determine features in training set. to be implemented by child 
        self.getFeatures(train_data)
        # function to find vectors (feature, value pairs). to be implemented by child
        train_vectors=self.getVectors(train_data)
        self.writeFeatureFile(train_vectors,"train.data")
        # train SVM model
        #call([self.svmlight_dir+"svm_learn", self.svmlight_dir+"train.data","svm_model_2"],stdout=open(os.devnull,'wb'))
        call([self.svmlight_dir+"svm_learn",self.svmlight_dir+"train.data","svm_model_2"],stdout=open(os.devnull,'wb'))
        # "-t", "1", "-d","-c","40"
    def test(self,test_data):
        """
        test svm 

        @param test_data: test data 
        @type test_data: list of (string, list) tuples corresponding to (label, content)

        """
        # function to find vectors (feature, value pairs). to be implemented by child
        test_vectors=self.getVectors(test_data)
        self.writeFeatureFile(test_vectors,"test.data")
        call([self.svmlight_dir+"svm_classify",self.svmlight_dir+"test.data","svm_model_2", 'svm_predictions'],stdout=open(os.devnull,'wb'))


        index = 0;
        with open('svm_predictions') as f:
            for line in f:
                predict = ''
                line =line.strip('\n')
                if float(line) > 0:
                    predict = 'POS'
                elif float(line) <= 0:
                    predict = 'NEG'
                else:
                    print'OHHHHHH'
                if predict == test_data[index][0]:
                    self.predictions.append('+')
                else:
                    self.predictions.append('-')
                index +=1

            #print self.predictions



        

class SVMText(SVM):
    def __init__(self,bigrams,trigrams,svmlight_dir,discard_closed_class):
        """ 
        initialisation of SVMText object

        @param bigrams: add bigrams?
        @type bigrams: boolean

        @param trigrams: add trigrams?
        @type trigrams: boolean

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param svmlight_dir: location of smvlight binaries
        @type svmlight_dir: string

        @param discard_closed_class: restrict unigrams to nouns, adjectives, adverbs and verbs?
        @type discard_closed_class: boolean
        """
        SVM.__init__(self,svmlight_dir)
        self.vocabulary=set()
        # add in bigrams?
        self.bigrams=bigrams
        # add in trigrams?
        self.trigrams=trigrams
        # restrict to nouns, adjectives, adverbs and verbs?
        self.discard_closed_class=discard_closed_class

    def getFeatures(self,reviews):
        """
        determine features from training reviews and store in self.vocabulary.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)
        """
        # reset for each training iteration
        self.vocabulary=set()
        for sentiment,review in reviews:
            for token in self.extractReviewTokens(review): 
                self.vocabulary.add(token)
        # using dictionary of vocabulary:index for constant order
        # features for SVMLight are stored as: (feature id, feature value)
        # using index+1 as a feature id cannot be 0 for SVMLight
        self.vocabulary={token:index+1 for index,token in enumerate(self.vocabulary)}

    def extractReviewTokens(self,review):
        """
        extract tokens from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of strings
        """
        text=[]
        for term in review:
            # check if pos tags are included in review e.g. ("bad","JJ")
            if len(term)==2 and self.discard_closed_class:
                if term[1][0:2] in ["NN","JJ","RB","VB"]: text.append(term)
            else:
                text.append(term)
        if self.bigrams:
            for bigram in ngrams(review,2): text.append(term)
        if self.trigrams:
            for trigram in ngrams(review,3): text.append(term)
        return text

    def getVectors(self,reviews):
        """
        get vectors for svmlight from reviews.

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of (string, list) tuples where string is the label ("1"/"-1") and list
                 contains the features in svmlight format e.g. ("1",[(1, 0.04), (2, 4.0), ...])
                 svmlight feature format is: (id, value) and id must be > 0.
        """
        # TODO Q6.1
        counts= []
        for i in range(len(reviews)):
            counts.append(Counter(reviews[i][1]))

        data = []

        for i in range(len(reviews)):
            listinn = []
            label = ''
            if reviews[i][0] == 'POS':
                label = '1'
            elif reviews[i][0] == 'NEG':
                label = '-1'
            else:
                print 'weird'
            count = counts[i]
            for key, value in count.iteritems():
                try:
                    tup = (self.vocabulary[key], float(value)/len(reviews[i][1]))
                    listinn.append(tup)
                except:
                    continue
            listinn.sort(key=lambda x:x[0])
            tupp = (label, listinn)
            data.append(tupp)

        return data



