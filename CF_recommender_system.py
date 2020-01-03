#Recommender System

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv('IMDB-Movie-Data.csv',sep=',')
dataset.head()


# Buliding CF model
c=dataset['Rating'].mean()
m=dataset['Votes'].quantile(0.9)
qual_dataset=dataset[(dataset['Runtime (Minutes)']>90)&(dataset['Revenue (Millions)']>100)]
qual_dataset=qual_dataset[qual_dataset['Votes']>=m]
def wr(x,m=m,c=c):
    v=x['Votes']
    r=x['Rating']
    return (v/(v+m)*r)+(m/(m+v)*c)
qual_dataset['score']=qual_dataset.apply(wr,axis=1)


#Displaying scores in descending order
qual_dataset=qual_dataset.sort_values('score',ascending=False)

qual_dataset[['Title','Votes','Rating','score']].head(10)

#Plotting the result
#Plotting result for original dataset
#Can see long tail phenomena in result
dataset=dataset.sort_values('Votes',ascending=False)
plt.figure(figsize=(18,6))
plt.barh(dataset['Title'].head(15),dataset['Votes'].head(15),align='center',color='cornflowerblue')
plt.gca().invert_yaxis()
plt.xlabel("Votes")
plt.title("MOst voted Movies")


# Plotting result for modified dataset
# long tail phenoma is removed
plt.figure(figsize=(18,6))
plt.barh(qual_dataset['Title'].head(15),qual_dataset['score'].head(15),align='center',color='cornflowerblue')
plt.gca().invert_yaxis()
plt.xlabel("Score")
plt.title("MOst scored Movies")