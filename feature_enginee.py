
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

def count_api(x,sub_x):
    num=x.count(sub_x)
    
    
    return num

def add_api_num_feature(data_path):
    data=pd.read_csv(data_path)
    data_path=data_path.replace("Input","Input2")
    print(data_path)
    api=[]
    with open("api_list.txt",'r') as f:
        for i in f:
            i=i.replace("\n","")
            api.append(i)
    api=list(set(api))
    for i in api:
        data["%s_num"%i]=data['api_text'].map(lambda x:count_api(x,i))
#         data["%s_num"%i]=np.log1p(data["%s_num"%i])  #取对数平滑

    data.to_csv(data_path,index=False)
    print("ok")


# In[ ]:


file_path=['./Input/test.csv','./Input/train_1.csv']

for p in file_path:
    print(p)
    add_api_num_feature(p)


# In[ ]:


import numpy  as np
import pandas as pd
def count_text_len(x):
    return len(x.split())

def add_text_len_feature(data_path):

    data=pd.read_csv(data_path)
    
    data['text_len_num']=data['api_text'].map(lambda x:count_text_len(x))
#     data['text_len_num']=np.log1p(data['text_len_num'])
    
    data.to_csv(data_path,index=False)
    print("ok")


# In[ ]:


file_path=['./Input2/test.csv','./Input2/train_1.csv']

for p in file_path:
    print(p)
    add_text_len_feature(p)


# In[ ]:


def add_rate_feature(data_path):
    data=pd.read_csv(data_path)

    api=[]
    with open("api_list.txt",'r') as f:
        for i in f:
            i=i.replace("\n","")
            api.append(i)
    apit=list(set(api))
    
    for i in api:
        data["%s_rate_num"%i]=data["%s_num"%i]/data['text_len_num']
        
    data.to_csv(data_path,index=False)
    print('ok')


# In[ ]:


file_path=['./Input/test.csv','./Input/train_0.csv','./Input/train_1.csv','./Input/train_2.csv','./Input/train_3.csv','./Input/train_4.csv']

for p in file_path:
    print(p)
    add_rate_feature(p)


# In[ ]:


def exist_api(x,i):
    if i in  x:
        return  1
    else:
        return 0

def add_api_exist_fature(data_path):
    data=pd.read_csv(data_path)

    api=[]
    with open("api_list.txt",'r') as f:
        for i in f:
            i=i.replace("\n","")
            api.append(i)
    api=list(set(api))
    for i in api:
        data["%s_api_exist"%i]=data["api_text"].map(lambda x:exist_api(x,i))
        
    data.to_csv(data_path,index=False)
    print('ok')  
    
    


# In[ ]:


file_path=['./Input/test.csv','./Input/train_0.csv','./Input/train_1.csv','./Input/train_2.csv','./Input/train_3.csv','./Input/train_4.csv']

for p in file_path:
    print(p)
    add_api_exist_fature(p)


# In[ ]:



import pandas as pd
test_data=pd.read_csv("/home/kawayi-4/xzl/notebook/kaggle/security/data/test.csv")


# In[ ]:


api=[]
with open("api_list.txt",'r') as f:
    for i in f:
        i=i.replace("\n","")
        api.append(i)
api=list(set(api))


# In[ ]:


test_return_value_df={}
test_return_value_df['file_id']=[]
for i in api:
    test_return_value_df["%s_api_return_value"%i]=[]


# In[ ]:


key_api_list=api
print(len(key_api_list))


# In[ ]:


gp=test_data.groupby("file_id")


# In[ ]:


count=0
for (i,v ) in gp:
    if count%1000==0:
        print(count)
    count+=1
    
    d=v.groupby("api")
    test_return_value_df['file_id'].append(i)
    allapi=set(v['api'].values.tolist())
    for api  in  key_api_list:
        if api not in allapi:
            test_return_value_df["%s_api_return_value"%api].append("None")  
    for j,dv in d:
        if  j  not in key_api_list:
            continue
        set_value=set(dv['return_value'].values.tolist())
        set_value_str=" ".join(str(m) for m in set_value)
        test_return_value_df["%s_api_return_value"%j].append(set_value_str)
    


# In[ ]:


new_df=pd.DataFrame(test_return_value_df)


# In[ ]:


new_df.to_csv("./Input/test_return_value.csv",index=False)

