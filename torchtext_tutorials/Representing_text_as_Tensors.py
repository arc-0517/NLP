import warnings
warnings.filterwarnings('ignore')

''' Text classification task '''
## requires
import torch
import torchtext
import os
import collections
os.makedirs('./data',exist_ok=True)
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
classes = ['World', 'Sports', 'Business', 'Sci/Tech']

## trainset sample
next(iter(train_dataset))
for i,x in zip(range(5),train_dataset):
    print(f"**{classes[x[0]]}** -> {x[1]}")

## dataset convert to list
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
train_dataset = list(train_dataset)
test_dataset = list(test_dataset)


## text into numbers(tensors)
'''
If want word-level representation, wee need to do two things:
* use tokenizer to split text into tokens
* build a vocabulary of those tokens.
'''
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
tokenizer('He said: hello')

counter = collections.Counter()
for (label, line) in train_dataset:
    counter.update(tokenizer(line))
vocab = torchtext.vocab.Vocab(counter)

vocab_size = len(vocab)
print(f"Vocab size if {vocab_size}")

def encode(x):
    return [vocab.stoi[s] for s in tokenizer(x)]
tokenizer('I love to play with my words')

encode('I love to play with my words')

## BoW(Bag of Words) text representation
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = [
        'I like hot dogs.',
        'The dog ran fast.',
        'Its hot outside.',
    ]
vectorizer.fit_transform(corpus)
vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

vocab_size = len(vocab)

def to_bow(text,bow_vocab_size=vocab_size):
    res = torch.zeros(bow_vocab_size,dtype=torch.float32)
    for i in encode(text):
        if i<bow_vocab_size:
            res[i] += 1
    return res

print(to_bow(train_dataset[0][1]))

## Training BoW Classifier
from torch.utils.data import DataLoader
import numpy as np

# this collate function gets list of batch_size tuples, and needs to
# return a pair of label-feature tensors for the whole minibatch
def bowify(b):
    return (
            torch.LongTensor([t[0]-1 for t in b]),
            torch.stack([to_bow(t[1]) for t in b]))

train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=bowify, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=bowify, shuffle=True)

net = torch.nn.Sequential(torch.nn.Linear(vocab_size,4),
                          torch.nn.LogSoftmax(dim=1))

def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = torch.nn.NLLLoss(),epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    net.train()
    total_loss,acc,count,i = 0,0,0,0
    for labels,features in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = loss_fn(out,labels) #cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss
        _,predicted = torch.max(out,1)
        acc+=(predicted==labels).sum()
        count+=len(labels)
        i+=1
        if i%report_freq==0:
            print(f"{count}: acc={acc.item()/count}")
        if epoch_size and count>epoch_size:
            break
    return total_loss.item()/count, acc.item()/count

train_epoch(net,train_loader,epoch_size=15000)


## BiGrams
'''
On limitation of a BoW is that some words are part of multi word expressions,
for example, 'hot dog' - 'hot' and 'dog' in other contexts
'''
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
corpus = [
        'I like hot dogs.',
        'The dog ran fast.',
        'Its hot outside.',
    ]
bigram_vectorizer.fit_transform(corpus)
print("Vocabulary:\n",bigram_vectorizer.vocabulary_)
bigram_vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

## N grams
counter = collections.Counter()
for (label, line) in train_dataset:
    l = tokenizer(line)
    counter.update(torchtext.data.utils.ngrams_iterator(l, ngrams=2))

bi_vocab = torchtext.vocab.Vocab(counter, min_freq=1)

print("Bigram vocabulary length = ", len(bi_vocab))


## TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit_transform(corpus)
vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()