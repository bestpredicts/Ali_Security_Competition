
# coding: utf-8

# In[ ]:


import pandas as pd
train_num=1
train_data=pd.read_csv("./Input2/train_%d.csv"%train_num)
test_data=pd.read_csv("./Input2/test.csv")


# In[ ]:


from sklearn.feature_extraction.text  import TfidfVectorizer
import os 
import pickle
n_range=(1,5)  #
max_feature=500000 #

if os.path.exists("API-Tfidf_%s.pkl"%str(n_range))==False:
    print("xx")
    api_tfidf=TfidfVectorizer(ngram_range=n_range,max_features=max_feature,min_df=2,max_df=0.98)
    api_tfidf.fit(train_data['api_text'].values)
    with open("API-Tfidf_%s.pkl"%str(n_range),'wb') as f:
        pickle.dump(api_tfidf,f)
        
else:
    with open("API-Tfidf_%s.pkl"%str(n_range),'rb')  as f:
        api_tfidf=pickle.load(f)


# In[ ]:



if os.path.exists("VALUE-Tfidf_%s.pkl"%str(n_range))==False:
    print("xx")
    value_tfidf=TfidfVectorizer(ngram_range=n_range,max_features=max_feature,min_df=2,max_df=0.98)
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

train_x=sparse.hstack((api_train_x,value_train_x))
test_x=sparse.hstack((api_test_x,value_test_x))


# In[ ]:


sparse.save_npz("./npz/train_apitext_valuetext.npz",train_x)
sparse.save_npz("./npz/test_apitext_valuetext.npz",test_x)


# In[ ]:


def  get_api_set(x):
    x=x.split()
    api_set=set(x)
    return " ".join(x)

train_data['api_set']=train_data['api_text'].map(lambda x:get_api_set(x))
test_data['api_set']=test_data['api_text'].map(lambda x:get_api_set(x))


# In[ ]:



if os.path.exists("API-SET-Tfidf_%s.pkl"%str(n_range))==False:
    print("xx")
    api_set_tfidf=TfidfVectorizer(ngram_range=n_range,max_features=311,min_df=2,max_df=0.98)
    api_set_tfidf.fit(train_data['api_set'].values)
    with open("API-SET-Tfidf_%s.pkl"%str(n_range),'wb') as f:
        pickle.dump(value_tfidf,f)
        
else:
    with open("API-SET-Tfidf_%s.pkl"%str(n_range),'rb')  as f:
        api_set_tfidf=pickle.load(f)


# In[ ]:


train_api_set_x=api_set_tfidf.transform(train_data['api_set'].values)
test_api_set_x=api_set_tfidf.transform(test_data['api_set'].values)


# In[ ]:


sparse.save_npz("./npz/train_api_set_x.npz",train_api_set_x)
sparse.save_npz("./npz/test_api_set_x.npz",test_api_set_x)


# In[ ]:


train_x=sparse.hstack((train_x,train_api_set_x))
test_x=sparse.hstack((test_x,test_api_set_x))


# In[ ]:


if os.path.exists("VALUE-SET-Tfidf_%s.pkl"%str(n_range))==False:
    print("xx")
    value_set_tfidf=TfidfVectorizer(ngram_range=n_range,min_df=2,max_df=0.98)
    value_set_tfidf.fit(train_data['value_set'].values)
    with open("VALUE-SET-Tfidf_%s.pkl"%str(n_range),'wb') as f:
        pickle.dump(value_tfidf,f)
        
else:
    with open("VALUE-SET-Tfidf_%s.pkl"%str(n_range),'rb')  as f:
        value_set_tfidf=pickle.load(f)


# In[ ]:


train_value_set_x=value_set_tfidf.transform(train_data['value_set'].values)
test_value_set_x=value_set_tfidf.transform(test_data['value_set'].values)


# In[ ]:


sparse.save_npz("./npz/train_value_set_x",train_value_set_x)
sparse.save_npz("./npz/test_value_set_x",test_value_set_x)


# In[ ]:


train_x=sparse.hstack((train_x,train_value_set_x))
test_x=sparse.hstack((test_x,test_value_set_x))


# In[ ]:


from sklearn.svm import  LinearSVC

train_y=train_data['label'].values


# In[ ]:



from sklearn.model_selection import  StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import  log_loss
nfold=10

kf=StratifiedKFold(nfold)
train_x=train_x.tocsr()
test_x=test_x.tocsr()

params = {
    'max_depth':-1,
    'boosting_type':'gbdt',
    'num_leaves': 31,
    'objective': 'multiclass',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'max_bin':255,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'multi_logloss',
    'num_class':6,
    'num_threads': -1,
    'is_unbalance':True,
}
import numpy as np
oof_train=np.zeros((train_x.shape[0],6))
oof_test=np.zeros((test_x.shape[0],6))
oof_test_skf=np.zeros((nfold,test_x.shape[0],6))
logloss_csv=[]

for i,(train,test) in  enumerate(kf.split(train_x,train_y)):
    print("*******第 %d 折********"%i)

    x_tr=train_x[train]
    x_te=train_x[test]
    y_tr=train_y[train]
    y_te=train_y[test]
    
    lgb_train = lgb.Dataset(x_tr, y_tr)
    lgb_eval = lgb.Dataset(x_te,y_te)
    lgb_train = lgb.Dataset(x_tr, y_tr)
    lgb_eval = lgb.Dataset(x_te,y_te,reference=lgb_train)
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)
    eva_pre=gbm.predict(x_te,num_iteration=gbm.best_iteration)
    oof_train[test]=eva_pre
    logloss=log_loss(y_te,eva_pre)
    print("valid logloss %s"%logloss)
    logloss_csv.append(logloss)
    pre=gbm.predict(test_x,num_iteration=gbm.best_iteration)
    oof_test_skf[i,:]=pre
    
print("mean loss  %f"%np.mean(logloss_csv))


# In[ ]:


oof_test=oof_test_skf.mean(axis=0)
score=np.mean(logloss_csv)
print(score)


# In[ ]:


#保存train_prob
train_prob=pd.DataFrame(oof_train)
train_prob.columns=['class_%s'%i for i  in   range(0,6)]
test_prob=pd.DataFrame(oof_test)
test_prob.columns=['class_%s'%i for i in range(0,6)]
train_prob.to_csv("./stacking_csv/cv10/train%d_lgb_tfidf_fea_train_prob_%f_cv5.csv"%(train_num,score),index=False)
test_prob.to_csv("./stacking_csv/cv10/train%d_lgb_tfidf_fea_test_prob_%f_cv5.csv"%(train_num,score),index=False)

