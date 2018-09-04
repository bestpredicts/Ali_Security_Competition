
# coding: utf-8

# In[ ]:


import pandas as pd
train_num=1
train_data=pd.read_csv("./Input/train_%d.csv"%train_num)
test_data=pd.read_csv("./Input/test.csv")


# In[ ]:


from sklearn.feature_extraction.text  import TfidfVectorizer
import os 
import pickle
n_range=(1,5)  #
max_feature=400000 #

if os.path.exists("API-Tfidf_%s.pkl"%str(n_range))==False:
    print("xx")
    api_tfidf=TfidfVectorizer(ngram_range=n_range,max_features=max_feature,min_df=2,max_df=0.97)
    api_tfidf.fit(train_data['api_text'].values)
    with open("API-Tfidf_%s.pkl"%str(n_range),'wb') as f:
        pickle.dump(api_tfidf,f)
        
else:
    with open("API-Tfidf_%s.pkl"%str(n_range),'rb')  as f:
        api_tfidf=pickle.load(f)


# In[ ]:



if os.path.exists("VALUE-Tfidf_%s.pkl"%str(n_range))==False:
    print("xx")
    value_tfidf=TfidfVectorizer(ngram_range=n_range,max_features=max_feature,min_df=2,max_df=0.97)
    value_tfidf.fit(train_data['value_text'].values)
    with open("VALUE-Tfidf_%s.pkl"%str(n_range),'wb') as f:
        pickle.dump(value_tfidf,f)
        
else:
    with open("VALUE-Tfidf_%s.pkl"%str(n_range),'rb')  as f:
        value_tfidf=pickle.load(f)


# In[ ]:


api_train_x=api_tfidf.transform(train_data['api_text'].values)
api_test_x=api_tfidf.transform(test_data['api_text'].values)


# In[ ]:


value_train_x=value_tfidf.transform(train_data['value_text'].values)
value_test_x=value_tfidf.transform(test_data['value_text'].values)


# In[ ]:


from scipy import  sparse

train_x_0=sparse.hstack((api_train_x,value_train_x))
test_x_0=sparse.hstack((api_test_x,value_test_x))


# In[ ]:


train_x=train_x_0
test_x= test_x_0


# In[ ]:


sparse.save_npz("./npy/data1_tfidf_train_x_10w.npz",train_x)


# In[ ]:


sparse.save_npz("./npy/data1_tfidf_test_x_10w.npz",test_x)


# In[ ]:


np.save("./npy/data1_train_y",train_y)

