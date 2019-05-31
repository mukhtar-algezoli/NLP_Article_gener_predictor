from SequenceVocabulary import *
from Vocabulary import *
import collections
import string
import numpy as np
from collections import Counter
class NewsVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""     
    def __init__(self,title_vocab, Category_vocab):       
        """         Args:             review_vocab (Vocabulary): maps words to integers     
        rating_vocab (Vocabulary): maps class labels to integers         """ 
        self.category_vocab = Category_vocab   
        self.title_vocab = title_vocab
    def vectorize(self, title , vector_length =-1): 
        indices = [self.title_vocab.begin_seq_index]
        indices.extend(self.title_vocab.lookup_token(token)
                       for token in title.split(" "))
        indices.append(self.title_vocab.end_seq_index)
        if vector_length < 0:
           vector_length = len(indices)
        out_vector = np.zeros(vector_length , dtype = np.int64)
        out_vector[:len(indices)] = indices
        out_vector[len(indices):] = self.title_vocab.mask_index
        
        return out_vector
    @classmethod    
    def from_dataframe(cls, news_df, cutoff=25):   
        category_vocab = Vocabulary()         
        for category in sorted(set(news_df.category)): 
            category_vocab.add_token(category)
        word_counts = Counter() 
        for title in news_df.title: 
            for token in title.split(" "): 
                if token  not in string.punctuation: 
                    word_counts[token] += 1          
        title_vocab = SequenceVocabulary() 
        for word, word_count in word_counts.items(): 
            if word_count >= cutoff: 
                title_vocab.add_token(word) 
        return cls(title_vocab, category_vocab)
    @classmethod    
    def from_serializable(cls, contents):    
        """Intantiate a ReviewVectorizer from a serializable dictionary      
        Args:             contents (dict): the serializable dictionary      
        Returns:             an instance of the ReviewVectorizer class         """    
        review_vocab = Vocabulary.from_serializable(contents['review_vocab'])       
        rating_vocab =  Vocabulary.from_serializable(contents['rating_vocab'])
        return cls(review_vocab=review_vocab, rating_vocab=rating_vocab)
    def to_serializable(self):        
        """Create the serializable dictionary for caching       
    Returns:             contents (dict): the serializable dictionary         """        
        return {'review_vocab': self.review_vocab.to_serializable(),                 'rating_vocab': self.rating_vocab.to_serializable()}
