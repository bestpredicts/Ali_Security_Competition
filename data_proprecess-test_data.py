
# coding: utf-8

# In[ ]:


import pandas as pd

train_data=pd.read_csv("../data/test.csv")


# In[ ]:


gp=train_data.groupby("file_id")


# In[ ]:


import time
import gc
proprecess_data = {}
proprecess_data['file_id'] = []
proprecess_data['tid_num']=[]
proprecess_data['api_text'] = []
proprecess_data['value_text'] = []

sum_file=len(gp)

start_time=time.time()

for i,(fid,f_df) in enumerate(gp):
    this_time=time.time()
    run_time=this_time-start_time
    if i%10000==0:
        print("proprecesed file %d  of %d   run  time  %d"%(i,sum_file,run_time))
        with open("test_data_propre.log" ,"a+") as f:
            f.write("proprecesed file %d  of %d   run  time  %d \n"%(i,sum_file,run_time))

    proprecess_data['file_id'].append(fid)
    tid_gp=f_df.groupby("tid")

    proprecess_data['tid_num'].append(len(tid_gp))
    
  
    content_list=[]
    value_content_list=[]
    for  tid,tid_df  in tid_gp:
        tid_df.sort_values(by=['index'],inplace=True)
        content_list.extend(tid_df['api'].values.tolist())
        content_list.append("。") 
        value_content_list.extend(tid_df['return_value'].values.astype(str).tolist())
        value_content_list.append("。")

    
    proprecess_data['api_text'].append(" ".join(content_list))
    proprecess_data['value_text'].append(" ".join(value_content_list))
    
    del content_list
    gc.collect()
    


# In[ ]:


new_df=pd.DataFrame(proprecess_data,columns=['file_id','tid_num','api_text','value_text'])
new_df.to_csv("proprecessed_test.csv",index=False)
new_df.head()

