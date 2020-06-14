#Importing our libraries 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

######################################################################################## functions
#for tokenize the corpus we use .split function which Split a string into a list
def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


#for making one hot vector , we make all vector values zero except the word id we make it 1 . 
def get_input_layer(word_idx):
    x = torch.zeros(vocabulary_size).float()
    x[word_idx] = 1.0
    return x
########################################################################################
    


######################################################################################## preparing data 
corpus = [
    'he is a king',
    'she is a queen',
    'he is a man',
    'she is a woman',
    'warsaw is poland capital',
    'berlin is germany capital',
    'paris is france capital',
]


tokenized_corpus = tokenize_corpus(corpus)


# vocabulary list whice made of Non-repeating list of our tokenized-corpus to specify our one hot vector 
vocabulary = []
for sentence in tokenized_corpus:
    for token in sentence:
        if token not in vocabulary:
            vocabulary.append(token)

# making dict for our words to handle it easier later 
word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

vocabulary_size = len(vocabulary)

WINDOW_SIZE = 2   
idx_pairs = []   # will store center word and its context in it 
for sentence in tokenized_corpus:
    #store all word`s positions in ith sentence in ith iteration in indices list 
    indices = [word2idx[word] for word in sentence]
    for center_word_pos in range(len(indices)):
        #store center word and its context by window size 
        for w in range(-window_size, window_size + 1):
            context_word_pos = center_word_pos + w
            # make sure not jump out sentence
            if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
                continue
            context_word_idx = indices[context_word_pos]
            idx_pairs.append((indices[center_word_pos], context_word_idx))

idx_pairs = np.array(idx_pairs) # it will be useful to have this as numpy array

########################################################################################


######################################################################################## model creation 
embedding_dims = 5 # word embeddings for our model 

#W1 is matrix which consists of embeddings dims * vocab size 
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
#W2 same as weights between hidden layer and o/p layer 
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)

#define our hyperparameters
num_epochs = 100
learning_rate = 0.001

#loop epochs
for epoch in range(num_epochs):
    loss_val = 0
   
   #loop for our data and calculating loss and do optimization 
   for data, target in idx_pairs:  #to refresh , idx pairs is list which has center word and its context words
        x = Variable(get_input_layer(data)).float()
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(W1, x)
        z2 = torch.matmul(W2, z1)
    
        log_softmax = F.log_softmax(z2, dim=0)

        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.data[0]
        loss.backward()
        W1.data -= learning_rate * W1.grad.data
        W2.data -= learning_rate * W2.grad.data

        W1.grad.data.zero_()
        W2.grad.data.zero_()
        
    if epo % 10 == 0:    
        print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
        
