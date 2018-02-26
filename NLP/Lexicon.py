from Analysis import Evaluation
from nltk.stem.porter import PorterStemmer
import os, codecs, sys

class SentimentLexicon(Evaluation):
    def __init__(self):
        """
        read in lexicon database and store in self.lexicon
        """
        # if multiple entries take last entry by default
        self.lexicon=dict([[l.split()[2].split("=")[1],l.split()] for l in open("data/sent_lexicon","r")])

    def classify(self,reviews,threshold,magnitude):
        """
        classify movie reviews using self.lexicon.
        self.lexicon is a dictionary of word: [polarity_info, magnitude_info], e.g. "bad": ["priorpolarity=negative","type=strongsubj"].
        explore data/sent_lexicon to get a better understanding of the sentiment lexicon.
        store the predictions in self.predictions as a list of strings where "+" and "-" are correct/incorrect classifications respectively e.g. ["+","-","+",...]

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @param threshold: threshold to center decisions on. instead of using 0, there may be a bias in the reviews themselves which could be accounted for. 
                          experiment for good threshold values.
        @type threshold: integer
        
        @type magnitude: use magnitude information from self.lexicon?
        @param magnitude: boolean
        """
        # reset predictions
        #TODO Q0.1

        #PROBABLY A BUG HERE WHEN DOING WITH MAGNITUDE, GETTING SAME RESULTS. PRETTY SURE WITHOUT MAGNITUDE IS CORRECT.

        self.predictions=[]
        if magnitude:
            print 'yeah'
            for word in reviews:
                score = 0
                predict = ''
                content = word[1]
                for i in range(len(content)):
                     if content[i] in self.lexicon:
                        if self.lexicon[content[i]][0] == 'type=strongsubj':
                            if self.lexicon[content[i]][5] == 'priorpolarity=positive':
                                score +=1
                            elif self.lexicon[content[i]][5] == 'priorpolarity=negative':
                                score -=1
                            else:
                                continue
                        elif self.lexicon[content[i]][0] == 'type=weaksubj':
                            if self.lexicon[content[i]][5] == 'priorpolarity=positive':
                                score +=0.5
                            elif self.lexicon[content[i]][5] == 'priorpolarity=negative':
                                score -=0.5
                            else:
                                continue
                        else:
                            print 'what'
                     else:
                        continue
                if score >= threshold:
                    predict = 'POS'
                else:
                    predict = 'NEG'
                if predict == word[0]:
                    self.predictions.append('+')
                else:
                    self.predictions.append('-')


        else:
            print 'hmmm'
            for word in reviews:
                score = 0
                content = word[1]
                for i in range(len(content)):
                    if content[i] in self.lexicon:
                        if self.lexicon[content[i]][5] == 'priorpolarity=positive':
                            score +=1
                        elif self.lexicon[content[i]][5] == 'priorpolarity=negative':
                            score -=1
                        else:
                            continue
                    else:
                        continue

                if score > threshold:
                    predict = 'POS'
                    #self.predictions.append('+')
                else:
                    #self.predictions.append('-')
                    predict = 'NEG'  
                if predict == word[0]:
                    self.predictions.append('+')
                else:
                    self.predictions.append('-')


 






