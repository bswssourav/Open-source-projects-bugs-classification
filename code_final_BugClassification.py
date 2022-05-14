
# coding: utf-8

# In[ ]:

from __future__ import division
import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


import string
import math

tokenize = lambda doc: doc.lower().split(" ")
from time import sleep
from tqdm import tqdm


# In[ ]:


df1 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_crash.csv')
df2 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_regression.csv')
df3 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_security.csv')
df4 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_clean.csv')
df5 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_polish.csv')
df6 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_performance.csv')
df7 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_usability.csv')
df8 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_networking.csv')
df9 = pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_concurrency.csv')
df_nocategory=pd.read_csv('C:/Users/Sourav Biswas/Desktop/category wise data updated/mozilla_nocategory.csv')


# In[ ]:


number_traindata=3000

X_train1, X_test1= train_test_split(df1['Summary'].values.astype('U'),test_size=0.0005)
X_train2, X_test2= train_test_split(df2['Summary'].values.astype('U'),test_size=0.0005)
X_train3, X_test3= train_test_split(df3['Summary'].values.astype('U'),test_size=0.0005)
X_train4, X_test4= train_test_split(df4['Summary'].values.astype('U'),test_size=0.0005)
X_train5, X_test5= train_test_split(df5['Summary'].values.astype('U'),test_size=0.0005)
X_train6, X_test6= train_test_split(df6['Summary'].values.astype('U'),test_size=0.0005)
X_train7, X_test7= train_test_split(df7['Summary'].values.astype('U'),test_size=0.0005)
X_train8, X_test8= train_test_split(df8['Summary'].values.astype('U'),test_size=0.0005)
X_train9, X_test9= train_test_split(df9['Summary'].values.astype('U'),test_size=0.0005)
X_test=df_nocategory['Summary'].values.astype('U')
partition = int(X_test.shape[0]/3)
print("partition:",partition);
upto_partition1= partition+13189
upto_partition2=upto_partition1+10000
upto_partition3=upto_partition2+10000
upto_partition4=upto_partition3+10000
lo=60001
hi=80000
X_test=X_test[lo:hi]
print(X_test.shape[0])
print("train size",X_train1.shape[0])


# In[ ]:


list1 =['crash', 'fail', 'failure', 'stack', 'render', 'buffering', 'Shuttering', 'break', 'broke', 'broken', 'block', 'stop', 'intermittent-failure', 'freeze']
list2=[ 'regress', 'reproduce', 'reporoducilble', 'add-on', 'add', 'new', 'resize', 'change']
list3=['secure', 'corrupt', 'privacy', 'password', 'access', 'qablocker', 'audit', 'sandbox', 'bypass', 'vulnerable', 'safe', 'intruder', 'password', 'password-manager', 'protect', 'authenticate', 'credentials', 'hack', 'virus', 'permission']
list4=[ 'remove', 'move', 'clean', 'refactor', 'rename', 'content', 'rid', 'unit', 'base', 'api', 'webkit', 'content', 'resolve']
list5=['polish', 'text', 'tab', 'delete', 'bookmark', 'drop', 'manage', 'scroll', 'dropdown']
list6=['perf', 'time', 'slow', 'cpu', 'lag', 'long', 'CPU', 'hang', 'load', 'lack', 'memory', 'reduce', 'time', 'fast', 'vary', 'execution', 'run', 'support', 'test', 'test cases', 'allow', 'time-out', 'time out', 'lazy', 'uses', 'kernel', 'ux-consistency', 'ux-efficiency']
list7=[ 'window', 'menu', 'zoom', 'use', 'click', 'cursor', 'keyborad', 'select', 'mouse', 'feature', 'hitting', 'entering', 'bar', 'button', 'icon', 'engine']
list8=[ 'network', 'client', 'server', 'url', 'html', 'http', 'https']
list9=[ 'race', 'lock', 'deadlock', 'starvation', 'synchronization', 'synchronize', 'atomic', 'concurrency', 'concurrent', 'semaphore']


# In[ ]:


df_new1=pd.DataFrame()
df_new2=pd.DataFrame()
df_new3=pd.DataFrame()
df_new4=pd.DataFrame()
df_new5=pd.DataFrame()
df_new6=pd.DataFrame()
df_new7=pd.DataFrame()
df_new8=pd.DataFrame()
df_new9=pd.DataFrame()


# In[ ]:


len1=len(list1)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train1.shape[0])
for i in range(len(list1)):
    x.append([])

for i in tqdm(range(X_train1.shape[0])):
    tokens = nltk.word_tokenize(X_train1[i])
    flag=0
    for w in tokens:
        for j in range(len(list1)):
            if list1[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new1=df_new1.append(df1.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new1.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train1.shape[0])):
        if flag1[i]==0:
            df_new1=df_new1.append(df1.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list2)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train2.shape[0])
for i in range(len(list2)):
    x.append([])

for i in tqdm(range(X_train2.shape[0])):
    tokens = nltk.word_tokenize(X_train2[i])
    flag=0
    for w in tokens:
        for j in range(len(list2)):
            if list2[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new2=df_new2.append(df2.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new2.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train2.shape[0])):
        if flag1[i]==0:
            df_new2=df_new2.append(df2.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list3)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train3.shape[0])
for i in range(len(list3)):
    x.append([])

for i in tqdm(range(X_train3.shape[0])):
    tokens = nltk.word_tokenize(X_train3[i])
    flag=0
    for w in tokens:
        for j in range(len(list3)):
            if list3[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new3=df_new3.append(df3.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new3.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train3.shape[0])):
        if flag1[i]==0:
            df_new3=df_new3.append(df3.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list4)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train4.shape[0])
for i in range(len(list4)):
    x.append([])

for i in tqdm(range(X_train4.shape[0])):
    tokens = nltk.word_tokenize(X_train4[i])
    flag=0
    for w in tokens:
        for j in range(len(list4)):
            if list4[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new4=df_new4.append(df4.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new4.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train4.shape[0])):
        if flag1[i]==0:
            df_new4=df_new4.append(df4.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list5)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train5.shape[0])
for i in range(len(list5)):
    x.append([])

for i in tqdm(range(X_train5.shape[0])):
    tokens = nltk.word_tokenize(X_train5[i])
    flag=0
    for w in tokens:
        for j in range(len(list5)):
            if list5[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new5=df_new5.append(df5.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new5.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train5.shape[0])):
        if flag1[i]==0:
            df_new5=df_new5.append(df5.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list6)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train6.shape[0])
for i in range(len(list6)):
    x.append([])

for i in tqdm(range(X_train6.shape[0])):
    tokens = nltk.word_tokenize(X_train6[i])
    flag=0
    for w in tokens:
        for j in range(len(list6)):
            if list6[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new6=df_new6.append(df6.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new6.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train6.shape[0])):
        if flag1[i]==0:
            df_new6=df_new6.append(df6.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list7)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train7.shape[0])
for i in range(len(list7)):
    x.append([])

for i in tqdm(range(X_train7.shape[0])):
    tokens = nltk.word_tokenize(X_train7[i])
    flag=0
    for w in tokens:
        for j in range(len(list7)):
            if list7[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new7=df_new7.append(df7.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new7.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train7.shape[0])):
        if flag1[i]==0:
            df_new7=df_new7.append(df7.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list8)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train8.shape[0])
for i in range(len(list8)):
    x.append([])

for i in tqdm(range(X_train8.shape[0])):
    tokens = nltk.word_tokenize(X_train8[i])
    flag=0
    for w in tokens:
        for j in range(len(list8)):
            if list8[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new8=df_new8.append(df8.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new8.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train8.shape[0])):
        if flag1[i]==0:
            df_new8=df_new8.append(df8.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


len1=len(list9)
target = number_traindata/len1
x=[]
flag1=np.zeros(X_train9.shape[0])
for i in range(len(list9)):
    x.append([])

for i in tqdm(range(X_train9.shape[0])):
    tokens = nltk.word_tokenize(X_train9[i])
    flag=0
    for w in tokens:
        for j in range(len(list9)):
            if list9[j]==w:
                x[j].append(i)
                flag=1
                break
        if flag==1:
            break   
            
          
for i in tqdm(range(len(x))):
    count=0
    for j in x[i]:
        df_new9=df_new9.append(df9.iloc[[j]],ignore_index=True)
        flag1[j]=1
        count+=1
        if(count>=target):
            break
            
left_over=number_traindata-df_new9.shape[0]
if left_over>0:
    count=0
    for i in tqdm(range(X_train9.shape[0])):
        if flag1[i]==0:
            df_new9=df_new9.append(df9.iloc[[i]],ignore_index=True)
            count+=1
            if count>left_over:
                break
        
  


# In[ ]:


print(df_new1.shape[0])
print(df_new2.shape[0])
print(df_new3.shape[0])
print(df_new4.shape[0])
print(df_new5.shape[0])
print(df_new6.shape[0])
print(df_new7.shape[0])
print(df_new8.shape[0])
print(df_new9.shape[0])


# In[ ]:


X_train1=df_new1['Summary'].values.astype('U')
X_train2=df_new2['Summary'].values.astype('U')
X_train3=df_new3['Summary'].values.astype('U')
X_train4=df_new4['Summary'].values.astype('U')
X_train5=df_new5['Summary'].values.astype('U')
X_train6=df_new6['Summary'].values.astype('U')
X_train7=df_new7['Summary'].values.astype('U')
X_train8=df_new8['Summary'].values.astype('U')
X_train9=df_new9['Summary'].values.astype('U')


# In[ ]:


import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt') # if necessary...


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

#vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]


# In[ ]:


sum=0.0
arr=np.zeros((X_test.shape[0],9))
for i in tqdm(range(X_test.shape[0])):
    sum=0.0
    for j in range(X_train1.shape[0]):
        sum+=cosine_sim(X_test[i],X_train1[j])
    arr[i][0]=sum/X_train1.shape[0]
    
    sum=0.0
    for j in range(X_train2.shape[0]):
        sum+=cosine_sim(X_test[i],X_train2[j])
    arr[i][1]=sum/X_train2.shape[0]
    
    sum=0.0
    for j in range(X_train3.shape[0]):
        sum+=cosine_sim(X_test[i],X_train3[j])
    arr[i][2]=sum/X_train3.shape[0]
    
    sum=0.0
    for j in range(X_train4.shape[0]):
        sum+=cosine_sim(X_test[i],X_train4[j])
    arr[i][3]=sum/X_train4.shape[0]
    
    sum=0.0
    for j in range(X_train5.shape[0]):
        sum+=cosine_sim(X_test[i],X_train5[j])
    arr[i][4]=sum/X_train5.shape[0]
    
    sum=0.0
    for j in range(X_train6.shape[0]):
        sum+=cosine_sim(X_test[i],X_train6[j])
    arr[i][5]=sum/X_train6.shape[0]
    
    sum=0.0
    for j in range(X_train7.shape[0]):
        sum+=cosine_sim(X_test[i],X_train7[j])
    arr[i][6]=sum/X_train7.shape[0]
    
    sum=0.0
    for j in range(X_train8.shape[0]):
        sum+=cosine_sim(X_test[i],X_train8[j])
    arr[i][7]=sum/X_train8.shape[0]
    
    sum=0.0
    for j in range(X_train9.shape[0]):
        sum+=cosine_sim(X_test[i],X_train9[j])
    arr[i][8]=sum/X_train9.shape[0]


# In[ ]:


threshold=np.mean(arr)+0.75*np.std(arr)
print(threshold)


# In[ ]:


arr1=[]
for i in tqdm(range(arr.shape[0])):
    max=-1
    cat=-1
    for j in range(9):
        if arr[i][j]>max:
            max=arr[i][j]
            cat=j
    if max>=threshold:
        arr1.append(cat)
    else:
        arr1.append(-1)
        


# In[ ]:


crash_count=0
regression_count=0
security_count=0
clean_count=0
polish_count=0
performance_count=0
usability_count=0
networking_count=0
concurrency_count=0
none_count=0
for i in tqdm(range(len(arr1))):
    if(arr1[i]==0):
        df_nocategory['category'][i+lo]='CRASH'
        crash_count=crash_count+1
    if(arr1[i]==1):
        df_nocategory['category'][i+lo]='REGRESSION'
        regression_count = regression_count + 1
    if(arr1[i]==2):
        df_nocategory['category'][i+lo]='SECURITY'
        security_count = security_count + 1
    if(arr1[i]==3):
        df_nocategory['category'][i+lo]='CLEAN'
        clean_count = clean_count + 1
    if(arr1[i]==4):
        df_nocategory['category'][i+lo]='POLISH'
        polish_count = polish_count + 1
    if(arr1[i]==5):
        df_nocategory['category'][i+lo]='PERFORMANCE'
        performance_count = performance_count + 1
    if(arr1[i]==6):
        df_nocategory['category'][i+lo]='USABILITY'
        usability_count = usability_count + 1
    if(arr1[i]==7):
        df_nocategory['category'][i+lo]='NETWORKING'
        networking_count = networking_count + 1
    if(arr1[i]==8):
        df_nocategory['category'][i+lo]='CONCURRENCY'
        concurrency_count = concurrency_count + 1
    if arr1[i] == -1:
        df_nocategory['category'][i+lo] = 'None'
        none_count=none_count+1


# In[ ]:


df_nocategory.to_csv("output_uncaTocategorized_60001to80000.csv")
        

print("uncategorized to categorized: CRASH", crash_count)

print("uncategorized to categorized: Regression:", regression_count)

print("uncategorized to categorized: performance: ", performance_count)

print("uncategorized to categorized: polish: ", polish_count)

print("uncategorized to categorized: clean: ", clean_count)

print("uncategorized to categorized: networking: ", networking_count)

print("uncategorized to categorized: usability: ", usability_count)

print("uncategorized to categorized: Concurrency: ", concurrency_count)

print("uncategorized to categorized: Security: ", security_count)

print("NO category count: ", none_count)

