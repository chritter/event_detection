import nltk,gensim
nltk.data.path=[]
nltk.data.path.append("C:\\Users\\rittchr\\nltk_data")
import re
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline



from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.grid_search import GridSearchCV


from nltk.data import load
all_pos_tags = list(load('nltk_data/help/tagsets/upenn_tagset.pickle').keys())
#all_pos_tags
NER_types = ['ORGANIZATION','PERSON','LOCATION','DATE','TIME','MONEY','PERCENT','FACILITY','FACILITY']

class extract_other_lexical_features(BaseEstimator, TransformerMixin):
    '''
    other lexical features such as time, special chars
    '''
    
    def fit(self, x, y=None):
        return self    

    def transform(self, sentences):
    
        def extract_other_lexical_features_int(sentence):
        
            tokentext = nltk.word_tokenize(sentence)

            ## Check if it is digit, could also use POS tag 'NUM'
            digits = np.any([token.isdigit() for token in tokentext])
            #digits = [any(char.isdigit() for char in token) for token in tokentext] #any char contains digit

            ## contains symbols (true), other characters
            symbols = np.any([not token.isalnum() for token in tokentext])

            ## contains time indicators ('yesterday','today')
            time_indicator_list = ['yesterday','today','tomorrow']
            times = np.any([True if token in time_indicator_list else False for token in tokentext])
            
            return [digits,symbols,times] #{'digits':digits,'symbols':symbols,'times':times}
        
        return [extract_other_lexical_features_int(sentence) for sentence in sentences]

class extract_syntactic_features(BaseEstimator, TransformerMixin):
    '''
    each sub-feature vector is of length all_pos_tags, fixed vector lengths!
    '''
    
    def fit(self, x, y=None):
        return self    

    def transform(self, sentences):
    
        def extract_syntactic_features(sentence):
            tokentext = nltk.word_tokenize(sentence)
            tags = [token[1] for token in nltk.pos_tag(tokentext)]

            # binary occurance of tags
            tag_occurance = [apt in tags for apt in all_pos_tags]

            count_dict = Counter(tags)

            # number of occurances
            tag_counts = [count_dict[apt] if apt in count_dict.keys() else 0 for apt in all_pos_tags]

            # occurance, 0, 1 or more
            tag_three_classes = [2 if tc>1 else tc for tc in tag_counts]

            # named entity recognition: person, organization, location, product, event,
            ner_found=[]
            for chunk in nltk.ne_chunk(nltk.pos_tag(tokentext)):
                if hasattr(chunk, 'label'):
                    ner_found.append(chunk.label())
            ners = [1 if ner in ner_found else 0 for ner in NER_types]

            return tag_occurance+tag_three_classes+tag_three_classes #{'tag_occurance':tag_occurance,'tag_three_classes':tag_three_classes,'ners':ners}

        return [extract_syntactic_features(sentence) for sentence in sentences]
        
        
def tokenize_lemmatize(sentence):
    
    #tokentext = nltk.word_tokenize(sentence)
    return [token.lemma_ for token in en_nlp(sentence)]

def tokenize_lemma_pos(sentence):
    '''
    Combine token name and pos label
    '''
    tokentext = nltk.word_tokenize(sentence)
    return [en_nlp(token[0])[0].lemma_+token[1] for token in nltk.pos_tag(tokentext)]
    
class Debug(BaseEstimator, TransformerMixin):

    def transform(self, X):
        self.shape = X.shape
        # what other output you want
        return X

    def fit(self, X, y=None, **fit_params):
        return self