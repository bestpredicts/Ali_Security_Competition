
# coding: utf-8

# In[ ]:


from scipy import sparse

train_tfidf=sparse.load_npz("./npy/data1_tfidf_train_x.npz")
train_count_fea=sparse.load_npz("./npy/data1_count_fea_train_x.npz")
train_x=sparse.hstack((train_tfidf,train_count_fea))

test_tfidf=sparse.load_npz("./npy/data1_tfidf_test_x.npz")
test_count_fea=sparse.load_npz("./npy/data1_count_fea_test_x.npz")
test_x=sparse.hstack((test_tfidf,test_count_fea))
                     
import numpy as np
train_y=np.load("./npy/data1_train_y.npy")


# In[ ]:



from sklearn.model_selection import  StratifiedKFold
import xgboost as xgb
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



 
params = {
    'booster': 'gbtree',
    # 'objective': 'multi:softmax',  # 多分类的问题、
    'objective': 'multi:softprob',   # 多分类概率
#     'objective': 'binary:logistic',
    'eval_metric': 'mlogloss',
    'num_class': 6,  # 类别数，与 multisoftmax 并用
    'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 8,  # 构建树的深度，越大越容易过拟合
    'alpha': 0,   # L1正则化系数
    'lambda': 10,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,  # 随机采样训练样本
    'colsample_bytree': 0.5,  # 生成树时进行的列采样
    'min_child_weight': 3,
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    'silent': 1,  # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.1,  # 如同学习率
    'seed': 1000,
    'nthread': -1,  # cpu 线程数
    'missing': 1,
#     'scale_pos_weight': (np.sum(y==0)/np.sum(y==1))  # 用来处理正负样本不均衡的问题,通常取：sum(negative cases) / sum(positive cases)
    # 'eval_metric': 'auc'
}

for i,(train,test) in  enumerate(kf.split(train_x,train_y)):
    print("*******第 %d 折********"%i)

    x_tr=train_x[train]
    x_te=train_x[test]
    y_tr=train_y[train]
    y_te=train_y[test]
    
    xgb_val = xgb.DMatrix(x_te, label=y_te)
    xgb_train = xgb.DMatrix(x_tr, label=y_tr)
    xgb_test = xgb.DMatrix(test_x)
    xgb_va=xgb.DMatrix(x_te)
    watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    
    model = xgb.train(params, xgb_train, 1000, watchlist, early_stopping_rounds=10)

    
    eva_pre=model.predict(xgb_va)
    oof_train[test]=eva_pre
    logloss=log_loss(y_te,eva_pre)
    print("valid logloss %s"%logloss)
    logloss_csv.append(logloss)
    pre=model.predict(xgb_test)
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
train_prob.to_csv("./stacking_csv/train%d_xgb_train_prob_cv5_%f.csv"%(train_num,score),index=False)
test_prob.to_csv("./stacking_csv/train%d_xgb_test_prob_cv5_%f.csv"%(train_num,score),index=False)

