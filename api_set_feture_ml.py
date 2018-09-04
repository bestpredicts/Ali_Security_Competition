
# coding: utf-8

# In[ ]:


from scipy import sparse
import numpy as np
train_x=sparse.load_npz("./npz/train_api_set_x.npz")
test_x=sparse.load_npz("./npz/test_api_set_x.npz")
train_y=np.load("./npy/data1_train_y.npy")


# In[ ]:



from sklearn.model_selection import  StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import  log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from  xgboost import XGBClassifier
nfold=10

kf=StratifiedKFold(nfold)
train_x=train_x.tocsr()
test_x=test_x.tocsr()


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
    
#     clf=RandomForestClassifier()
#     clf=LogisticRegression()
    clf=XGBClassifier()
    clf.fit(x_tr,y_tr)
    eva_pre=clf.predict_proba(x_te)
    oof_train[test]=eva_pre
    logloss=log_loss(y_te,eva_pre)
    print("valid logloss %s"%logloss)
    logloss_csv.append(logloss)
    pre=clf.predict_proba(test_x)
    oof_test_skf[i,:]=pre
    
print("mean loss  %f"%np.mean(logloss_csv))


# In[ ]:


oof_test=oof_test_skf.mean(axis=0)
score=np.mean(logloss_csv)
print(score)


# In[ ]:


#保存train_prob
import pandas as pd
train_num=1
train_prob=pd.DataFrame(oof_train)
train_prob.columns=['class_%s'%i for i  in   range(0,6)]
test_prob=pd.DataFrame(oof_test)
test_prob.columns=['class_%s'%i for i in range(0,6)]
train_prob.to_csv("./stacking_csv/cv10/train%d_xgb_train_prob_cv10_%f.csv"%(train_num,score),index=False)
test_prob.to_csv("./stacking_csv/cv10/train%d_xgb_test_prob_cv10_%f.csv"%(train_num,score),index=False)


# In[ ]:


np.mean(logloss_csv)


# In[ ]:


from sklearn.metrics import  classification_report
y_pr=[np.argmax(i) for i in eva_pre]
print(classification_report(y_te,y_pr))

