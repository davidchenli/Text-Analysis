#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pickle


# In[10]:


#pickle.dump( dic, open( f'dic.p', "wb" ) )
#pickle.dump( freqs, open( f'freqs.p', "wb" ) )
#pickle.dump( sentence, open( f'sentcnts.p', "wb" ) )
#pickle.dump( word, open( f'wordcnts.p', "wb" ) )
#pickle.dump( coocs, open( f'coocs.p', "wb" ) )
#os.getcwd()
#corpus(f"C:/Users/USER/Desktop/fb2015/%s"%os.listdir(file)[1])
#file="fb%d"%year[0]


# In[11]:


#dic=pickle.load(     open( f'dic.p', "rb"       ) )
#freqs = pickle.load( open( f'freqs.p', "rb"    ) )
#sen = pickle.load(   open( f'sentcnts.p', "rb" ) )
#word = pickle.load(  open( f'wordcnts.p', "rb" ) )
#coocs = pickle.load(open( f'coocs.p', "rb" ) )


# In[21]:


def cooccurrence_symmetric_window( sentlist, targets, weights ):
    m = len(weights)
    cooc = np.zeros((len(targets), len(targets)), np.float64)
    word2id = {w:i for (i,w) in enumerate(targets,0)}
    for sent in sentlist:
        words = sent.split()
        n = len(words)
        for i in range(n):
            end = min(n-1, i+m)
            for j in range(i+1, end+1):
                if(words[i] in word2id.keys() and words[j] in word2id.keys()):
                    cooc[word2id[words[i]], word2id[words[j]]] += weights[j-i-1]
    np.fill_diagonal(cooc, 0)
    return cooc+cooc.T
def textseries(month,path=path,year=year, target=target):
    sen=[]
    word=[]
    dic=[]
    for i in year:
        file="fb%d/"%i 
        p=path+file
        a=['fpost-%d-%d.csv'%(i,x) for x in month]    
        for j in range(len(a)):
            fh = open(p+a[j], 'r',encoding="utf8")
            s=fh.readlines()
            s=[re.sub("http[^\s]+|www\[^\s]+|@[^\s]+","URL",x.lower().strip()) for x in s]
            article=[re.sub("\\\\r\\\\n\\\\r\\\\n",' ',x) for x in s]
            article = [re.sub("[^a-zA-Z]", " ", x) for x in article]
            article=[re.sub("b Name|bName",'',x) for x in article]
            sentence=len(s)
            wordtot=sum([len(x.split()) for x in s])
            sen.append(sentence)
            word.append(wordtot)
            print(i,j+1,sentence,wordtot)
            article=pd.DataFrame({"line":article})
            word_lem= WordNetLemmatizer()
            dic.append( article["line"].apply(lambda x: ' '.join([word_lem.lemmatize(a,pos='n') for a in x.split() if a not in stop])))
            fh.close()
    freqs=[]
    coocs=[]
    for article in dic:
        freq=dict(zip(target,np.zeros(len(target),dtype=np.int)))
        for texts in article:
            for x in texts.split():
                 if x in freq.keys():
                        freq[x]+=1
        freqs.append(freq)
        coocs.append(cooccurrence_symmetric_window(article, target, [1,1/2,1/3]))
    return(sen,word,freqs,coocs)

def ppmi_cooc( cooc ):
    marginal0 = cooc.sum(0) # column sum
    marginal1 = cooc.sum(1) # row sum
    l=( np.array([marginal1]).T.dot( np.array([marginal0]) ) )
    l=l*(l>0)+(l<=0)/1000
    lift = cooc*sum(marginal0)/l
    ppmi = np.log2( lift*(lift>1) + (lift<=1) )
    return ppmi

def cosin(word1,word2,mat,target):
    dit=dict(zip(target,range(len(target))))
    return((np.dot(mat[dit[word1]],mat[dit[word2]])/(norm(mat[dit[word1]])*norm(mat[dit[word2]]))).round(3))


# In[19]:


path= "C:/Users/USER/Desktop/"
os.chdir(path)


# In[20]:


## Target list Read
f=open("Text/Food Ingredient.txt", 'r', encoding='utf8')
s=f.read()
f.close()
target = [x.lower() for x in re.sub('\n',' ',s).split()]
target=[re.sub("[^a-z]","@",x) for x in target if "@" not in re.sub("[^a-z]","@",x) ]
wordnet_lemmatizer= WordNetLemmatizer()
from autocorrect import Speller
spell = Speller(lang='en')
target=[spell(wordnet_lemmatizer.lemmatize(x,pos='n' ))for x in target]
target=set(target)
len(target)


# In[34]:


#### Clean and calculate frequency and cooc matrix
year=range(2011,2016)
mon=range(1,13)
stop = {"URL"}|set(stopwords.words("english"))
sentence,word,freqs,coocs=textseries(path=path,year=year, target=target,month=mon)


# In[41]:


#### Graph for word and sentence count
month=[]
for i in year:
    for k in mon:
        month.append("%s-%d"%(i,k))
plt.figure(figsize=(8,6))
plt.plot(month,sentence)
plt.xticks( [month[i] for i in [max(0,x*6-1) for x in range(len(year)*2+1)]])
plt.ylabel("Count")
plt.title("Number of Sentence")
plt.show()
plt.figure(figsize=(8,6))
plt.plot(month,word)
plt.xticks( [month[i] for i in [max(0,x*6-1) for x in range(len(year)*2+1)]])
plt.ylabel("Count")
plt.title("Number of Word")
plt.show()


# In[42]:


### Plot chosen word frequency with/without normalization with total word count
def plotrend1(year,mon,word,freqs,normal=True):
    tex=[x.lower() for x in word]
    month=[]
    for i in year:
        for k in mon:
            month.append("%s-%d"%(i,k))
    for word in tex:
        k=[x[word] for x in freqs]
        plt.figure(figsize=(8,6))
        a = np.ones(len(month))
        if normal:
            a = [sum(x.values()) for x in freqs]
    
        plt.plot(month,np.array(k)/np.array(a),label=word)
        plt.xticks([month[i] for i in [max(0,x*6-1) for x in range(len(year)*2+1)]])
        plt.suptitle("Frequency on %s"%word)
        plt.ylabel("Word Frequency")
        if normal:
            plt.title("with normalization")
            plt.ylabel("Percantage from total")
        plt.legend()
        plt.show()
    
plotrend1(year,mon,['cauliflower',"rice"],freqs,normal=True)


# In[43]:


plotrend1(year,mon,['cauliflower',"rice"],freqs,normal=False)


# In[44]:


# Plot ppmi and cosin similarity for chosen pair of word

def plotrend2(year,mon,word1,word2,targe,coocs,order1=True):
    word1=word1.lower()
    word2=word2.lower()
    month=[]
    dit=dict(zip(target,range(len(target))))
    for i in year:
        for k in mon:
            month.append("%s-%s"%(i,k))
    first=[]
    second = []
    
    for i in range(len(coocs)):
        coo=ppmi_cooc(coocs[i])
        first.append(coo[dit[word1],dit[word2]])
        second.append(cosin(word1,word2,coo,target))
    
    plt.figure(figsize=(8, 6))
    
    
    if order1:
        plt.plot(month,first,label=[word1,word2])
        plt.suptitle("Concurrence on %s and %s"%(word1,word2))
        plt.ylabel("PPMI")
    else:
        plt.plot(month,second,label=[word1,word2])
        plt.suptitle("Cosin Similarity of %s and %s"%(word1,word2))
        plt.ylabel("Similarity")
    plt.legend()
    plt.xticks( [month[i] for i in [max(0,x*6-1) for x in range(len(year)*2+1)]])
    plt.show()

plotrend2(year,mon,"cauliflower","rice",target,coocs,False)


# In[45]:


plotrend2(year,mon,"cauliflower","rice",target,coocs,True)

