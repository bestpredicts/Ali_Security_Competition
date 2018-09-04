
# coding: utf-8

# In[ ]:


from scipy import sparse
import numpy as np

train_text_tfidf=sparse.load_npz("./npz/train_apitext_valuetext.npz")
train_api_tfidf=sparse.load_npz("./npz/train_api_set_x.npz")
train_value_tfidf=sparse.load_npz("./npz/train_value_set_x.npz")
train_count_fea=np.load("./npy/train_count_fea_x.npy")

train_x=sparse.hstack((train_text_tfidf,train_api_tfidf))
train_x=sparse.hstack((train_x,train_value_tfidf))
train_x=sparse.hstack((train_x,train_count_fea))
print(train_x.shape)


# In[ ]:


test_text_tfidf=sparse.load_npz("./npz/test_apitext_valuetext.npz")
test_api_tfidf=sparse.load_npz("./npz/test_api_set_x.npz")
test_value_tfidf=sparse.load_npz("./npz/test_value_set_x.npz")
test_count_fea=np.load("./npy/test_count_fea_x.npy")

test_x=sparse.hstack((test_text_tfidf,test_api_tfidf))
test_x=sparse.hstack((test_x,test_value_tfidf))
test_x=sparse.hstack((test_x,test_count_fea))
print(test_x.shape)


# In[ ]:


train_y=np.load("./npy/data1_train_y.npy")


# In[ ]:


print(train_x.shape,test_x.shape,train_y.shape)


# In[ ]:


from sklearn.decomposition import  TruncatedSVD
svd = TruncatedSVD(250)
svd.fit(train_x)
train_x=svd.transform(train_x)
test_x=svd.transform(test_x)
print('ok')


# In[ ]:


train_x.shape


# In[ ]:



from sklearn.model_selection import  StratifiedKFold
import lightgbm as lgb
from sklearn.metrics import  log_loss
nfold=10

kf=StratifiedKFold(nfold)
train_x=train_x
test_x=test_x


import numpy as np
oof_train=np.zeros((train_x.shape[0],6))
oof_test=np.zeros((test_x.shape[0],6))
oof_test_skf=np.zeros((nfold,test_x.shape[0],6))
logloss_csv=[]

from keras.callbacks import  ModelCheckpoint
from keras.callbacks import  EarlyStopping
from keras.utils import  to_categorical

for i,(train,test) in  enumerate(kf.split(train_x,train_y)):
    print("*******第 %d 折********"%i)

    x_tr=train_x[train]
    x_te=train_x[test]
    y_tr=train_y[train]
    y_te=train_y[test]
    y_tr=to_categorical(y_tr)
    y_te=to_categorical(y_te)
    model=nn()
    weight_path="./nn_%d.h5"%i
    mc=ModelCheckpoint(weight_path,save_best_only=True)
    ep=EarlyStopping(patience=5)
    
    cb=[mc,ep]

    model.fit(x_tr,y_tr,validation_data=(x_te,y_te),callbacks=cb,epochs=100,batch_size=128)
    model.load_weights(weight_path)
    eva_pre=model.predict(x_te)
    oof_train[test]=eva_pre
    logloss=log_loss(y_te,eva_pre)
    print("valid logloss %s"%logloss)
    logloss_csv.append(logloss)
    pre=model.predict(test_x)
    oof_test_skf[i,:]=pre
    
print("mean loss  %f"%np.mean(logloss_csv))


# In[ ]:


oof_test=oof_test_skf.mean(axis=0)


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
train_prob.to_csv("./stacking_csv/cv10/train%d_nn_train_prob_cv5.csv"%train_num,index=False)
test_prob.to_csv("./stacking_csv/cv10/train%d_nn_test_prob_cv5.csv"%train_num,index=False)


# In[ ]:


np.mean(logloss_csv)


# In[ ]:


from sklearn.metrics import  classification_report
y_pr=[np.argmax(i) for i in eva_pre]
print(classification_report(y_te,y_pr))

