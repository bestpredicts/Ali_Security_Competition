
# coding: utf-8

# In[1]:


import gensim 
import pandas as pd

train_data=pd.read_csv("./Input/train_1.csv")
train_data.head()


# In[2]:


test_data=pd.read_csv("./Input/test.csv")
test_data.head()


# In[ ]:


data=pd.concat([train_data,test_data])
sen_list=data['api_text'].str.split().tolist() 


# In[ ]:


import gensim
w2v=gensim.models.Word2Vec(sentences=sen_list,size=50,min_count=1,window=5,iter=40,workers=32)


# In[11]:


w2v.wv.save_word2vec_format("./api_w2v_50.txt",binary=False) 


# In[7]:


sen=[]

for i in  sen_list:
    sen.extend(i)


# In[8]:


len(sen)


# In[9]:


len(set(sen))


# In[12]:



from keras.preprocessing.text import   Tokenizer
max_feature=311
token=Tokenizer(max_feature)
token.fit_on_texts(data['api_text'].values.tolist())

import numpy as np


EMBEDDING_FILE="./api_w2v_50.txt"
def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE))



embed_size=300
nb_words=max_feature

word_index = token.word_index
nb_words = min(max_feature, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_feature: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


# In[13]:


embedding_matrix.shape 


# In[14]:


np.save("./Embedding_Matrix_dim_50",embedding_matrix) 


# In[15]:


import pickle

with open("./Tokenizer.pkl",'wb') as f:
    pickle.dump(token,f)

