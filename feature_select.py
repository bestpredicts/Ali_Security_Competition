
# coding: utf-8

# In[2]:


import pandas as pd
train_num=1
train_data=pd.read_csv("./Input2/train_%d.csv"%train_num)


# In[5]:


import numpy as np
oth_fea=[i for i in train_data.columns  if 'num' in i or '_rate' in i  or 'exist' in i]
oth_fea_x=train_data[oth_fea].values


# In[6]:


oth_fea_x.shape


# In[7]:


from scipy import sparse

train_x=oth_fea_x


# In[8]:


print(train_x.shape)


# In[9]:


from sklearn.svm import  LinearSVC

train_y=train_data['label'].values


# In[10]:


from sklearn.model_selection import  train_test_split
x_tr,x_te,y_tr,y_te=train_test_split(train_x,train_y,test_size=0.2,shuffle=True)


# In[11]:



from sklearn.model_selection import  StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import  log_loss

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

    
lgb_train = lgb.Dataset(x_tr, y_tr)
lgb_eval = lgb.Dataset(x_te,y_te)

gbm = lgb.train(params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=lgb_eval,
            early_stopping_rounds=10)


# In[12]:


df=pd.DataFrame({"feature_name":oth_fea,"importance":gbm.feature_importance()})


# In[13]:


df.sort_index(by='importance',inplace=True,ascending=False)


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(1,figsize=(10, 5))
feature_list=df['importance'].values[:20]
feature=df['feature_name'].values[:20]
plt.bar(range(len(feature_list)),feature_list)
plt.xticks(range(len(feature)),feature, rotation='vertical')
plt.title("feature importace")
plt.savefig("feature_importace")
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df.head(20)


# In[ ]:


df.to_csv("lgb_feature_importance.csv",index=False)


# In[ ]:


unsed_feature=df[df.importance==0]
unsed_feature.head()


# In[ ]:


unsed_feature.info()


# In[ ]:


unsed_feature.to_csv("unused_fea.csv",index=False)


# In[ ]:


unsed_fea=unsed_feature['feature_name'].values.tolist()


# In[ ]:


unsed_fea


# In[ ]:


file=['./Input/test.csv','./Input/train_0.csv','./Input/train_1.csv','./Input/train_2.csv','./Input/train_3.csv','./Input/train_4.csv']


# In[ ]:


for i in file:
    data=pd.read_csv(i)
    data.drop(columns=unsed_fea,inplace=True)
    data.to_csv(i,index=False)


# In[ ]:


data=pd.read_csv("./Input/test.csv")


# In[ ]:


data.info()

