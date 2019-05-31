# Article genre predictor  
## Task
build a python/pytorch programme that captures the sequence of words from an article headline and predict its genre(bussinese,sci/tech,world,sport) using ag_news dataset.
## Summary 
this programme builds a Vocabulary of all the words present in the dataset where each word given an integer value. when a headline (input) is given to the programme a one hot encoding is produced for each word using its number in the vocabulary. this seq of one hot encodings (for each headline) is pushed into a CNN neural net, the output of the CNN is then pushed into a linear layer that update the weights of the whole neural net (during training) using backpropagation or predict the genre of the headline(when testing or predicting).
## Functionality
this project uses nlp embeddings and CNNs to predict the gener of an article from its headline , there is four geners : Bussinese,Sci/Tech,Sports,World. the programme also gives the percentage for each predition.
## Design 
a website will soon be released
## Run Locally
   * import torch,re
   * Savedmodel.tar has the pretrained weights
   * for new prediction run predict.py and insert the headline as instructed (will take few seconds to load the embeddings)
   * for training just run training.py
## Tech stack
   * pytorch
   * CNN
   * embeddings
   * vocabulary
   * vectorizor
   * Dataloader
