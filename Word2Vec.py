#after we know how to make skipgram from scratch , we will use here our libraries to make it easier 

import torch
import torch.nntorch.nn  as  nn
import torch.autogradtorch.aut  as autograd
import torch.optim as optim
import torch.nn.functional as F
import random, math
import numpy as np
import time
from numpy.random import multinomial
from collections import Counter
import re
import nltk 
nltk.download('brown')  # library for our corpus 
from nltk.corpus import brown


################################################################################## forward prop

class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.embeddings_target = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_context = nn.Embedding(vocab_size, embedding_size)

    def forward(self, target_word, context_word, negative_example):
        emb_target = self.embeddings_target(target_word)
        emb_context = self.embeddings_context(context_word)
        emb_product = torch.mul(emb_target, emb_context)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))
        emb_negative = self.embeddings_context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2))
        emb_product = torch.sum(emb_product, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product))
        return -out


##################################################################################  

################################################################################## our functions 

# fucnction for filter most frequent words from our vocab because they do not affect our learning well
def subsample_frequent_words(vocab , corpus):

# vocab is unrepeated list of all our words , corpus is list of words in our text . 
  new_tokens = []
# tokens is a list of word indexes from original text
  for word in vocab: 
  
    frac = vocab[word].count/len(tokens)
    prob = (np.sqrt(frac/0.001) + 1) * (0.001/frac)
    
    if np.random.random() < prob:
        new_tokens.append(word)  # our new filtered vocab 
        


# function for negative sampling our o/p so it will be less compututationally for our gradient
def sample_negative(sample_size):
    sample_probability = {}
    word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus))))
    normalizing_factor = sum([v**0.75 for v in word_counts.values()])
    for word in word_counts:
        sample_probability[word] = word_counts[word]**0.75 / normalizing_factor
    words = np.array(list(word_counts.keys()))
    while True:
        word_list = []
        sampled_index = np.array(multinomial(sample_size, list(sample_probability.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                 word_list.append(words[index])
        yield word_list
        


# return us the target word and i/p and the negative samples within specific batch size
def get_batches(context_tuple_list, batch_size=100):
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i in range(len(context_tuple_list)):
        batch_target.append(word_to_index[context_tuple_list[i][0]])
        batch_context.append(word_to_index[context_tuple_list[i][1]])
        batch_negative.append([word_to_index[w] for w in context_tuple_list[i][2]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = autograd.Variable(torch.from_numpy(np.array(batch_target)).long())
            tensor_context = autograd.Variable(torch.from_numpy(np.array(batch_context)).long())
            tensor_negative = autograd.Variable(torch.from_numpy(np.array(batch_negative)).long())
            batches.append((tensor_target, tensor_context, tensor_negative))
            batch_target, batch_context, batch_negative = [], [], []
    return batches
 

##################################################################################  

################################################################################## preparing data 
corpus = []

# get our corpus from our dataset 
for cat in ['news']:
    for text_id in brown.fileids(cat):
        raw_text = list(itertools.chain.from_iterable(brown.sents(text_id)))
        text = ' '.join(raw_text)
        text = text.lower()
        text.replace('\n', ' ')
        text = re.sub('[^a-z ]+', '', text)
        corpus.append([w for w in text.split() if w != '']) 

#subsample our vocab 
vocab = set(itertools.chain.from_iterable(corpus))
corpus = subsample_frequent_words(corpus)
vocabulary = set(itertools.chain.from_iterable(corpus))
vocabulary_size = len(vocabulary)

word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}



context_tuple_list = []
w = 4
negative_samples = sample_negative(8)

for text in corpus:
    for i, word in enumerate(text):
        first_context_word_index = max(0,i-w)
        last_context_word_index = min(i+w, len(text))
        for j in range(first_context_word_index, last_context_word_index):
            if i!=j:
                context_tuple_list.append((word, text[j], next(negative_samples)))
        
##################################################################################   

################################################################################## MODEL CREATION
loss_function = nn.CrossEntropyLoss()
net = Word2Vec(embedding_size=200, vocab_size=vocabulary_size)
optimizer = optim.Adam(net.parameters())
losses = []
num_epochs = 2000 
for epoch in range(num_epochs):
  losses = []
  context_tuple_batches = get_batches(context_tuple_list, batch_size=2000)
  for i in range(len(context_tuple_batches)):
    net.zero_grad()
    target_tensor, context_tensor, negative_tensor = context_tuple_batches[i]
    loss = net(target_tensor, context_tensor, negative_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.data)
