﻿from NewsVectorizer import *
from torch.utils.data import Dataset
import pandas as pd
import torch

class NewsDataset(Dataset):  
    def __init__(self, news_df, vectorizer):    
        """
        Args:
            news_df (pandas.DataFrame): the dataset
            vectorizer (NewsVectorizer): vectorizer instatiated from dataset
        """
        self.news_df = news_df
        self._vectorizer = vectorizer

        # +1 if only using begin_seq, +2 if using both begin and end seq tokens
        measure_len = lambda context: len(context.split(" "))
        self._max_seq_length = max(map(measure_len, news_df.title)) + 2
        

        self.train_df = self.news_df[self.news_df.split=='train']
        self.train_size = len(self.train_df)

        self.val_df = self.news_df[self.news_df.split=='val']
        self.validation_size = len(self.val_df)

        self.test_df = self.news_df[self.news_df.split=='test']
        self.test_size = len(self.test_df)

        self._lookup_dict = {'train': (self.train_df, self.train_size),
                             'val': (self.val_df, self.validation_size),
                             'test': (self.test_df, self.test_size)}

        self.set_split('train')

        # Class weights
        class_counts = news_df.category.value_counts().to_dict()
        def sort_key(item):
            return self._vectorizer.category_vocab.lookup_token(item[0])
        sorted_counts = sorted(class_counts.items(), key=sort_key)
        frequencies = [count for _, count in sorted_counts]
        self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)
        
        
    @classmethod    
    def load_dataset_and_make_vectorizer(cls, news_csv): 
        """Load dataset and make a new vectorizer from scratch       
        Args:             review_csv (str): location of the dataset  
        Returns:             an instance of ReviewDataset         """   
        news_df = pd.read_csv(news_csv)
        train_news_df = news_df[news_df.split=='train']
        return cls(news_df, NewsVectorizer.from_dataframe(train_news_df))
    def get_vectorizer(self):      
        """ returns the vectorizer """     
        return self._vectorizer
    def set_split(self, split="train"): 
        """ selects the splits in the dataset using a column in the dataframe     
        Args:             split (str): one of "train", "val", or "test"         """    
        self._target_split = split      
        self._target_df, self._target_size = self._lookup_dict[split]
    def __len__(self):     
        return self._target_size
    def __getitem__(self, index):    
        """the primary entry point method for PyTorch datasets       
        Args:             index (int): the index to the data point    
        Returns:             a dict of the data point's features (x_data) and label (y_target)         """    
        row = self._target_df.iloc[index]
        title_vector =self._vectorizer.vectorize(row.title , self._max_seq_length)
        category_index =self._vectorizer.category_vocab.lookup_token(row.category)
        return {'x_data': title_vector,          
                'y_target': category_index}
    def get_num_batches(self, batch_size):       
        """Given a batch size, return the number of batches in the dataset           
        Args:             batch_size (int)         Returns:             number of batches in the dataset         """  
        return len(self) 