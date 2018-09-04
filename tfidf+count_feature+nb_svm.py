
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import sparse


train_x=sparse.load_npz("./npy/train_x.npz")
test_x=sparse.load_npz("./npy/test_x.npz")
train_y=np.load("./npy/train_y.npy")


# In[2]:


from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self


# In[ ]:



from sklearn.model_selection import  StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import  log_loss
nfold=5

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
    
    model=NbSvmClassifier()
    model.fit(x_tr,y_tr)
    
    eva_pre=model.predict_proba(x_te)
    oof_train[test]=eva_pre
    logloss=log_loss(y_te,eva_pre)
    print("valid logloss %s"%logloss)
    logloss_csv.append(logloss)
    pre=model.predict_proba(test_x)
    oof_test_skf[i,:]=pre
    
print("mean loss  %f"%np.mean(logloss_csv))


# In[4]:


oof_test=oof_test_skf.mean(axis=0)
score=np.mean(logloss_csv)
print(score)


# In[6]:


#保存train_prob
train_num=1
train_prob=pd.DataFrame(oof_train)
train_prob.columns=['class_%s'%i for i  in   range(0,6)]
test_prob=pd.DataFrame(oof_test)
test_prob.columns=['class_%s'%i for i in range(0,6)]
train_prob.to_csv("./stacking_csv/train%d_nbsvm_train_prob_cv5_%f.csv"%(train_num,score),index=False)
test_prob.to_csv("./stacking_csv/train%d_nbsvm_test_prob_cv5_%f.csv"%(train_num,score),index=False)

