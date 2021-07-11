
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from numpy.linalg import svd
os.chdir("C:/Users/USER/Desktop/Text/")


#%%%% Function to deal with inconsistent date format

def day(x):
    for i in formats:
        try:
            return(datetime.strptime(x,i))
        except ValueError:
            print(x)


#%% Read gi.csv file 


file= "gi.csv"

def gi_load(file):
    df=pd.read_csv(file,index_col="Entry",dtype=object)
    df=pd.notnull(df)
    df.index=[re.sub("[^a-zA-Z]","", x.lower()) for x in df.index.values]
    unique={}
    duplicate=[]    
    for x in df.index.values:
        if x not in unique.keys():
            unique[x]=1
        else:
            duplicate.append(x)
    remove=set(duplicate)
    clean = {x:df.loc[x].sum(0)>0 for x in remove}
    clean.update({x: df.loc[x] for x in unique if x not in remove})
    df_clean=pd.DataFrame(clean)
    cat=df_clean.index.values
    word =df_clean.columns.values
    word2cat={x: list(cat[df_clean[x]]) for x in word}
    cat2word={x:dict(zip(word[df_clean.loc[x]],np.ones(len(word[df_clean.loc[x]]),np.int32))) for x in cat}
    return(word2cat,cat2word)


word2cat,cat2word = gi_load(file)

#%% Count number of words and categories in gi.csv


df=pd.read_csv(file,index_col="Entry",dtype=object)
print("number of category: %d"%len(cat2word.keys()))
print("number of word : %d" %len(df.index))
print("number of unique word : %d"%len(word2cat.keys()))


#%% Read and check date format


test = pd.read_csv("abreast_of_the_market.csv",encoding="utf8")
date= [re.sub(" ",'',x) for x in test["date"] ]
formats = ["%b%d,%Y"]
date = np.array([day(x) for x in date])

[x for x in date if pd.notnull(x) and x.year == 1998 and (x.month == 10 or x.month==11)]

# From code above we can see that there were 1 row in Nov 1998 without date information and during that year
# between Oct and Nov, 1998/11/26, which should have post on that day as it was work day. So we'll 
#input that date manually.  
#%%

date[pd.isnull(date)] =datetime(1998,11,26)
unique_date={}
duplicate_date=[]
for x in date:
    if x not in unique_date.keys():
        unique_date[x]=1
        
    else:
        duplicate_date.append(x)
        unique_date[x]+=1

test["date"]=date
test.loc[[x in [x for x in duplicate_date] for x in test["date"]]]

# We check wheter there's duplicated date in our data and if the date was 
# duplicated, check in the dataframe that whether the title and text content 
# were the same.


print("number of rows : %d" %len(test.index.values))
print("number of unique dates : %d"%len(unique_date))


# There are 18 dates have duplicated value in document. Those duplicated date
# had similar but not the same content and title. Thus, we'll combine the 
# content in duplicated date to form new row and then  divide the word count
# by number of duplicate rows(2) to calculate index for normalization.


#%% Sort the date and create list for date and categories


unique_date = {k: v for k, v in sorted(unique_date.items(), key=lambda item: item[0])}
date =  list(unique_date.keys())
cats= list(cat2word.keys())


#%% Join the content in same date 
# and clean the text to lower case and remove other special character

texts = {}
for i in date:
    raw = test.loc[test["date"] ==i]
    k=""
    l=''
    for j in range(len(raw)):
        k += raw.iloc[j,1]+' '
        l += raw.iloc[j,2]+' '
    texts[i] = re.sub("[^a-zA-Z]",' ',l.lower()) 


#%% Function for gi analysis and create dcm as dataframe


def gi_analysis(texts,unique_date,cat2word,word2cat):
    output={}
    for i in unique_date.keys():
        final =dict(zip(cats, np.zeros(len(cats))))
        text = texts[i]
        for x in text.split():
            for target in cats:
                if x in cat2word[target].keys():
                    final[target]+=1
        output[i]=dict(zip(final.keys(),np.array(list(final.values()))/unique_date[i]))
    return(pd.DataFrame(output,index=cats).T)


output_df = gi_analysis(texts,unique_date,cat2word,word2cat)


#%% Plot required scatter plot for Positiv and Negativ category


plt.figure(figsize=(8, 6))
plt.scatter(date,output_df["Positiv"],s=10)
plt.scatter(date,output_df["Negativ"],s=10)
plt.legend(['Positiv', 'Negative'])
plt.show()


#%% LSA analysis on dcm with highest explained variation


###NMF method: Would have reverse graph comparing to svd method
r = 1
model = NMF(n_components=r, init='random',max_iter=10000, random_state=0)
W = model.fit_transform(output_df)
H = model.components_
weight=H
index = output_df.apply(lambda x : x*H.reshape((-1,)),axis=1).sum(1)
plt.scatter(unique_date.keys(),index,s=10,c = 'red')
plt.legend(["Pessimism"])
plt.show()

#### SVD method
U,D,Vt = svd(np.array(output_df),compute_uv=True, full_matrices=False)

dim=1
Uk = U[:,range(dim)]
Vk = Vt[range(dim),:]
Dk = np.diag(D[range(dim)])

docproj = Uk.dot(Dk)
termproj = Vk
index1 = (termproj.reshape((-1,))*output_df).sum(1)
plt.scatter(date,index1,s=10,c="red")
plt.legend(["Pessimism"])
plt.show()

plt.scatter(date,docproj,s=10)
plt.legend(["Pessimism"])
plt.show()
