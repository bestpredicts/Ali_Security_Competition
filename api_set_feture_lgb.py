
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
import pandas as pd
train_num=1
train_prob=pd.DataFrame(oof_train)
train_prob.columns=['class_%s'%i for i  in   range(0,6)]
test_prob=pd.DataFrame(oof_test)
test_prob.columns=['class_%s'%i for i in range(0,6)]
train_prob.to_csv("./stacking_csv/cv10/train%d_lgb_train_prob_cv10_%f.csv"%(train_num,score),index=False)
test_prob.to_csv("./stacking_csv/cv10/train%d_lgb_test_prob_cv10_%f.csv"%(train_num,score),index=False)

