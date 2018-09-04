
# coding: utf-8

# In[ ]:


from scipy import sparse

train_tfidf=sparse.load_npz("./npy/data1_tfidf_train_x.npz")
train_count_fea=sparse.load_npz("./npy/data1_count_fea_train_x.npz")
train_x=sparse.hstack((train_tfidf,train_count_fea))


# In[ ]:


test_tfidf=sparse.load_npz("./npy/data1_tfidf_test_x.npz")
test_count_fea=sparse.load_npz("./npy/data1_count_fea_test_x.npz")
test_x=sparse.hstack((test_tfidf,test_count_fea))


# In[ ]:


import numpy as np
train_y=np.load("./npy/data1_train_y.npy")


# In[ ]:



from sklearn.model_selection import  StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import  log_loss
nfold=5

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

from sklearn.linear_model import LogisticRegression


for i,(train,test) in  enumerate(kf.split(train_x,train_y)):
    print("*******第 %d 折********"%i)

    x_tr=train_x[train]
    x_te=train_x[test]
    y_tr=train_y[train]
    y_te=train_y[test]
    
    lr=LogisticRegression()
    lr.fit(x_tr,y_tr)
    eva_pre=lr.predict_proba(x_te)
    oof_train[test]=eva_pre
    logloss=log_loss(y_te,eva_pre)
    print("valid logloss %s"%logloss)
    logloss_csv.append(logloss)
    pre=lr.predict_proba(test_x)
    oof_test_skf[i,:]=pre
    
print("mean loss  %f"%np.mean(logloss_csv))


# In[ ]:


oof_test=oof_test_skf.mean(axis=0)
score=np.mean(logloss_csv)
print(score)


# In[ ]:


import numpy as np
def check_ans(result):
    new_result=[]
    for i in result:
        m=i.sum()
        for j in range(0,6):
            i[j]=i[j]/m       
        if abs(i.sum()-1)>=1e-6:
            print(i.sum()-1)
        new_result.append(i)
    return new_result


# In[ ]:


import numpy as np

res=check_ans(oof_test)


# In[ ]:


#
test_prob=pd.DataFrame(oof_test)
test_prob.columns=['prob0','prob1','prob2','prob3','prob4','prob5']
test_prob['file_id']=test_data['file_id']


# In[ ]:


test_prob=test_prob[['file_id','prob0','prob1','prob2','prob3','prob4','prob5']]
test_prob.head(20)


# In[ ]:


test_prob.to_csv("lgb_submit_%f.csv"%np.mean(logloss_csv),index=False)


# In[ ]:


test_prob.info()


# In[ ]:


#保存train_prob
import pandas as pd
train_num=1
train_prob=pd.DataFrame(oof_train)
train_prob.columns=['class_%s'%i for i  in   range(0,6)]
test_prob=pd.DataFrame(oof_test)
test_prob.columns=['class_%s'%i for i in range(0,6)]
train_prob.to_csv("./stacking_csv/train%d_lr_train_prob_cv5_%f.csv"%(train_num,score),index=False)
test_prob.to_csv("./stacking_csv/train%d_lr_test_prob_cv5_%f.csv"%(train_num,score),index=False)


# In[ ]:


np.mean(logloss_csv)


# In[ ]:


from sklearn.metrics import  classification_report
y_pr=[np.argmax(i) for i in eva_pre]
print(classification_report(y_te,y_pr))

