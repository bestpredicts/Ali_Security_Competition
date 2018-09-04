
# coding: utf-8

# In[ ]:


import pandas as pd

train_data=pd.read_csv("./proprecessed_train.csv")


# In[ ]:


gp=train_data.groupby("label")
class_dict={}
for i,(c,v) in enumerate(gp):
    class_dict[c]=v
    
for  i in  class_dict.keys():
    class_dict[i].to_csv("class_%s.csv"%i,index=False)


# In[ ]:


lenth=len(class_dict[0])
start=0
end=0
for i  in  range(5):
    end=int((i+1)*lenth/5)
    print(start,end)
    data=class_dict[0][start:end]
    data.to_csv("class_0_%d.csv"%(i+1),index=False)
    start=end
    


# In[ ]:


class_12345=pd.concat([class_dict[1],class_dict[2],class_dict[3],class_dict[4],class_dict[5]])
from sklearn.utils import  shuffle
class_12345=shuffle(class_12345)
class_12345.to_csv("class_12345.csv",index=False)


# In[ ]:


import pandas as pd

class_0_1=pd.read_csv("./class_0_1.csv")
class_0_2=pd.read_csv("./class_0_2.csv")
class_0_3=pd.read_csv("./class_0_3.csv")
class_0_4=pd.read_csv("./class_0_4.csv")
class_0_5=pd.read_csv("./class_0_5.csv")
class_12345=pd.read_csv("./class_12345.csv")
c_list=[class_0_1,class_0_2,class_0_3,class_0_4,class_0_5]
from sklearn.utils import shuffle

for i,d in enumerate(c_list):
    data=pd.concat([d,class_12345])
    data=shuffle(data)
    data.to_csv("./Input/train_%d.csv"%i,index=False)

