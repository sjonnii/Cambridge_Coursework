import os, codecs, sys
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
import re
import codecs

class MovieReviewCorpus():

    def __init__(self,stemming,pos):
        """
        initialisation of movie review corpus.

        @param stemming: use porter's stemming?
        @type stemming: boolean

        @param pos: use pos tagging?
        @type pos: boolean
        """
        # raw movie reviews
        self.reviews=[]
        # held-out train/test set
        self.train=[]
        self.test=[]
        # folds for cross-validation
        self.folds={}
        # porter stemmer
        self.stemmer=PorterStemmer() if stemming else None
        # part-of-speech tags
        self.pos=pos
        # import movie reviews
        self.get_reviews()

    def get_reviews(self):
        """
        processing of movie reviews.

        1. parse reviews in data/reviews and store in self.reviews.

           the format expected for reviews is: [(string,list), ...] e.g. [("POS",["a","good","movie"]), ("NEG",["a","bad","movie"])].
           in data/reviews there are .tag and .txt files. The .txt files contain the raw reviews and .tag files contain tokenized and pos-tagged reviews.

           to save effort, we recommend you use the .tag files. you can disregard the pos tags to begin with and include them later.
           when storing the pos tags, please use the format for each review: ("POS/NEG", [(token, pos-tag), ...]) e.g. [("POS",[("a","DT"), ("good","JJ"), ...])]

           to use the stemmer the command is: self.stemmer.stem(token)
           
        2. store training and held-out reviews in self.train/test. files beginning with cv9 go in self.test and others in self.train

        3. store reviews in self.folds. self.folds is a dictionary with the format: self.folds[fold_number] where fold_number is an int 0-9.
           you can get the fold number from the review file name.
        """
        # TODO Q0
        for i in range(10):
            self.folds[i] = []

        pospath = '/Users/sigurjonisaksson/Documents/NLP/scripts/data/reviews'  

        en_stop = get_stop_words('en')

        for (dirpath, dirnames, filenames) in os.walk(pospath):
            for filename in filenames:
                if filename.endswith('.tag'):
                    d = dirpath + '/' + filename
                    label = dirpath[-3:]
                    with open(d, 'r') as text:
                        temp = []
                        pos_tokens = []
                        for line in iter(text):
                            if self.pos:
                                head, sep, tail = line.partition('\t')
                                head = unicode(head, 'utf-8')
                                head = head.lower()
                                try:
                                    head = self.stemmer.stem(head)
                                except:
                                    head = head
                                if head in en_stop:
                                    continue
                                elif not head in en_stop:
                                    temp.append((head,tail))
                                else:
                                    print 'weird'
                                    
                            else:
                                head, sep, tail = line.partition('\t')
                                head = unicode(head, 'utf-8')
                                head = head.lower()
                                try:
                                    head = self.stemmer.stem(head)
                                except:
                                    head = head

                                if head in en_stop:
                                    continue
                                elif not head in en_stop:
                                    temp.append(head)
                                else:
                                    print 'weird'


                        #temp = [i for i in temp if not i in en_stop]

                        tup = (label, temp)

                        if self.pos:
                            self.reviews.append(tup)
                            if filename[2] == '9':
                                self.test.append(tup)
                            else:
                                self.train.append(tup)

                            self.folds[int(filename[2])].append(tup)


                        else:
                            self.reviews.append(tup)
                            if filename[2] == '9':
                                self.test.append(tup)
                            else:
                                self.train.append(tup)

                            self.folds[int(filename[2])].append(tup)




# stemmer=PorterStemmer()
# boolian = False
# reviews = []
# train = []
# test = []
# folds={}
# #initialize folds as emtpy lists
# for i in range(10):
#     folds[i] = []

# pospath = '/Users/sigurjonisaksson/Documents/NLP/scripts/data/reviews'
# for (dirpath,dirnames,filenames) in os.walk(pospath):
#     for filename in filenames:
#         if filename.endswith('.tag'):
#             d = dirpath + '/' + filename
#             label = dirpath[-3:]
#             with open(d, 'r') as text:
#                 temp = []
#                 temp2 = []
#                 #reviews.append(label)
#                 for line in iter(text):
#                     if boolian:
#                         pass
#                         # head, sep, tail = line.partition('\t')
#                         # tail = tail.strip('\n');
#                         # temp2.append(head)
#                         # temp2.append(tail)
#                         # temp.append(temp2)
#                     else:
#                         head, sep, tail = line.partition('\t')
#                         try:
#                             head = stemmer.stem(head)
#                         except:
#                             head = head
#                         temp.append(head)
#                 tup = (label, temp)
#                 reviews.append(tup)
#                 if filename[2] == '9':
#                     test.append(tup)
#                 else:
#                     train.append(tup)
#                 folds[int(filename[2])].append(tup)






