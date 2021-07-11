#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sys
from sklearn.decomposition import LatentDirichletAllocation as LDA
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from numpy.linalg import svd


# In[10]:


def corpus( path, encoding='utf8', nonalphabetic=False, tolower=False ):
    
    files = [x for x in path] 
    doc8raw = []
    filenames = []
    for file in files:
        filenames.append( file )
        f=open(file, 'r', encoding=encoding)
        s=f.readlines()
        f.close()
        s = [x.strip() for x in s]
        doc8raw.append( ' '.join(s) )
    if tolower:
        doc8raw = [x.lower() for x in doc8raw]
    if nonalphabetic:
        doc8raw = [re.sub("[^a-zA-Z]", " ", x) for x in doc8raw]
    return doc8raw, filenames


# In[11]:


os.chdir(f"C:/Users/USER/Desktop/spritzer15/spritzer/15")
a = os.listdir(f"C:/Users/USER/Desktop/spritzer15/spritzer/15")
text,name=corpus(a,encoding = 'latin1')


# In[12]:


clean = []
for texts in text:
    texts=re.sub("@[^\s]+",'tag',texts)
    texts=re.sub("http[^\s]+|www\[^\s]+","url",texts)
    clean.append(texts)


# In[13]:


stop = {'tag','url','co', 'amp', 'via', 'don', 'dont'}|set(stopwords.words("english"))|set(stopwords.words("spanish"))|set(stopwords.words("portuguese"))


# In[14]:


normal=CountVectorizer(stop_words=stop)

dt= normal.fit_transform(clean).toarray()

terms = np.array(normal.get_feature_names() )

nonsparse = (dt>0).sum(0) > 0.001*dt.shape[0] # sparsity threshold 0.001
print("Drop terms with low weight: %d columns were dropped"%(len(nonsparse)-sum(nonsparse)))
dt = dt[:,nonsparse]
terms=terms[nonsparse]
print("Drop document with no element: %d rows were dropped"%sum(dt.sum(1)==0))
dt=dt[dt.sum(1)!=0,:]


# In[15]:


### NMF Method without tfidf
r = 10
model = NMF(n_components=r, init='random', random_state=0,max_iter=1000)
W = model.fit_transform(dt)
H = model.components_


# In[16]:


topic=[]
for k in range(10):
    topic = dict(zip(terms, H[k,:]))
    topic = {k: v for k, v in sorted(topic.items(), key=lambda item: item[1], reverse=True)} # discard stop words and sort by value
    topicwords = list(topic.keys())
    print("\nTopic ", k, ": \t", topicwords[0 : 10])


# In[17]:


### SVD without TFIDF
U,D,Vt = svd(dt, full_matrices=False)

dim=10
Uk = U[:,range(dim)]
Vk = Vt[range(dim),:]
Dk = np.diag(D[range(dim)])

docproj = Uk.dot(Dk)
termproj = Dk.dot(Vk).T



# In[18]:


topic=[]
for k in range(10):
    topic = dict(zip(terms, termproj[:,k]))
    topic = {k: v for k, v in sorted(topic.items(), key=lambda item: item[1], reverse=True)} # discard stop words and sort by value
    topicwords = list(topic.keys())
    print("\nTopic ", k, ": \t", topicwords[0 : 10])


# In[19]:


### LDA without TFIDF
T = 10
lda = LDA(n_components=T,random_state=10)
doctopicslda=lda.fit_transform(dt)
topicterms = lda.components_


for k in range(0, T):
    topic = dict(zip(terms, topicterms[k,:]))
    topic = {k: v for k, v in sorted(topic.items(), key=lambda item: item[1], reverse=True)} # discard stop words and sort by value
    topicwords = list(topic.keys())
    print("\nTopic ", k, ": \t", topicwords[0 : 10])


# In[20]:


# Final Graph
topic = dict(zip(terms, topicterms[4,:]))
topic = {k: v for k, v in sorted(topic.items(), key=lambda item: item[1], reverse=True)}
topc_10=(pd.DataFrame(topic.values(),index=topic.keys())).iloc[0:10,:]

wcloud = WordCloud().generate_from_frequencies( dict(zip(topc_10.index,topc_10.iloc[:,0])) )
plt.imshow(wcloud, interpolation='bilinear')


# In[21]:


############ Additional try: tfidf method

f=dt/(np.reshape(dt.sum(1), (-1,1)).dot( np.ones((1,dt.shape[1])) ))
idf=np.log2(len((dt>0)[1,])/((dt>0).sum(0)+0.1) )
weighted=f*idf


# In[22]:


######### NMF with tfidf method
r = 10
model = NMF(n_components=r, init='random', random_state=0,max_iter=1000)
W = model.fit_transform(weighted)
H = model.components_
topic=[]
for k in range(10):
    topic = dict(zip(terms, H[k,:]))
    topic = {k: v for k, v in sorted(topic.items(), key=lambda item: item[1], reverse=True)} # discard stop words and sort by value
    topicwords = list(topic.keys())
    print("\nTopic ", k, ": \t", topicwords[0 : 10])


# In[23]:


######### SVD with tfidf
U,D,Vt = svd(weighted, full_matrices=False)

dim=10
Uk = U[:,range(dim)]
Vk = Vt[range(dim),:]
Dk = np.diag(D[range(dim)])

docproj = Uk.dot(Dk)
termproj = Dk.dot(Vk).T
topic=[]
for k in range(10):
    topic = dict(zip(terms, termproj[:,k]))
    topic = {k: v for k, v in sorted(topic.items(), key=lambda item: item[1], reverse=True)} # discard stop words and sort by value
    topicwords = list(topic.keys())
    print("\nTopic ", k, ": \t", topicwords[0 : 10])


# In[24]:


############# LDA with tfidf
T = 10

lda = LDA(n_components=T)
doctopicslda=lda.fit_transform(weighted)
# term distribution of each topic, one row per topic, one column per term
topicterms = lda.components_


for k in range(0, T):
    topic = dict(zip(terms, topicterms[k,:]))
    topic = {k: v for k, v in sorted(topic.items(), key=lambda item: item[1], reverse=True)} # discard stop words and sort by value
    topicwords = list(topic.keys())
    print("\nTopic ", k, ": \t", topicwords[0 : 10])

