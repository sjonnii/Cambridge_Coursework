import numpy as np, os
from subprocess import call
from gensim.models import Doc2Vec
from Classifiers import SVM
import gensim
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from subprocess import call


class SVMDoc2Vec(SVM):
    """ 
    class for baseline extension using SVM with Doc2Vec pre-trained vectors
    """
    def __init__(self,model,svmlight_dir):
        """
        initialisation of parent SVM object and self.model attribute
        to initialise SVM parent use: SVM.__init_(self,svmlight_dir)

        @param model: pre-trained doc2vec model to use
        @type model: string (e.g. random_model.model)

        @param svmlight_dir: location of local binaries for svmlight
        @type svmlight_dir: string
        """
        # TODO Q8

        SVM.__init__(self,svmlight_dir)
        self.model = gensim.models.Doc2Vec.load(model)
        docvec = self.model.docvecs[99]
        print len(docvec)



    def normalize(self,vector):
        """
        normalise vector between -1 and 1 inclusive.

        @param vector: vector inferred from doc2vec
        @type vector: numpy array

        @return: normalised vector
        """
        


    def getVectors(self,reviews):
        """
        infer document vector for each review. 

        @param reviews: movie reviews
        @type reviews: list of (string, list) tuples corresponding to (label, content)

        @return: list of (string, list) tuples where string is the label ("1"/"-1") and list
                 contains the features in svmlight format e.g. ("1",[(1, 0.04), (2, 4.0), ...])
                 svmlight feature format is: (id, value) and id must be > 0.
        """
        # TODO Q8
        # training_vectors = []

        vectors= []

        for i in range(len(reviews)):
            if reviews[i][0] == 'NEG':
                label = '-1'
            else:
                label = '1'
            vector = self.model.infer_vector(doc_words= reviews[i][1])
            norm = np.linalg.norm(vector,2)
            listi = []
            for values, elements in enumerate(vector):
                elements = float(elements)/norm
                tup = (values+1, elements)
                listi.append(tup)
            tup2 = (label,listi)
            vectors.append(tup2)

        return vectors

    # since using pre-trained vectors don't need to determine features 
    def getFeatures(self,reviews):
        pass
