#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import re
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


# load stopwords

# selected after observing results without stopwords
stopwords = [

  'the',

  'about',

  'an',

  'and',

  'are',

  'at',

  'be',

  'can',

  'for',

  'from',

  'if',

  'in',

  'is',

  'it',

  'of',

  'on',

  'or',

  'that',

  'this',

  'to',

  'you',

  'your',

  'with',

]


# In[4]:


def urlfinder():
    # find urls and twitter usernames within a string

    url_finder = re.compile(r"(?:\@|https?\://)\S+")
    return url_finder


# In[5]:


def filter_tweet(s):

  s = s.lower() # downcase

  s = url_finder.sub("", s) # remove urls and usernames

  return s


# In[6]:


### load data ###

def gettweets():
    df = pd.read_csv('C:/Users/Aashya.Khanduja/Documents/Aashya/Machine Learning/Unsupervised Learning/tweets.csv')

    count = df.count
    print (count)
#     text = df.text.tolist()

#     text = [filter_tweet(s) for s in text]

    return count


# In[7]:


def tfidf():
    # transform the text into a data matrix

    tfidf = TfidfVectorizer(max_features=100, stop_words=stopwords)

    X = tfidf.fit_transform(text).todense()
    return X


# In[8]:


def d(u,v):
    diff = u-v
    return diff.dot(diff)


# In[9]:


def cost(X,R,M):
    cost = 0
    for k in range(len(M)):
        for n in range(len(X)):
            cost +=R[n,k]*d(M[k],X[n])
    return cost
    


# In[42]:


def plot_k_means(X,K, max_iter=100, beta=0.1):
    N,D = X.shape
    M = np.zeros((K,D))
    R = np.zeros((N,K))
    
    #initialise M to random
    for k in range(K):

        M[k] = X[np.random.choice(N)]

    #adding a grid to see the changes in the scatter plots and ensuring that they arent overwritten
    grid_width = 5
    grid_height = max_iter/grid_width
    random_colors = np.random.random((K,3))
    plt.figure()
    
    costs = np.zeros(max_iter)

    #step 1 = determine assignments/ responsibilities
    for i in range(max_iter):
        # moving plots inside for loop
        colors = R.dot(random_colors)
#         matplotlib.rcParams['figure.figsize'] = [50, 50]
        plt.subplot(40, 40, i+1)
        plt.scatter(X[:,0], X[:,1], s=7, c=colors)
    plt.scatter()(X[:,0], X[:,1], marker = "*", c='g')
#         print ("iteration")
        
#         for k in range(K):

#             for n in range(N):

#                 R[n,k] = np.exp(-beta*d(M[k], X[n]))/ np.sum( np.exp(-beta*d(M[j],X[n])) for j in range(K))
                
# #         Recalculate Means
#         for k in range(K):
#             M[k] = R[:,k].dot(X) / R[:,k].sum()
            
#         costs[i] = cost(X,R,M)
#         if i>0:
#             if np.abs(costs[i] - costs[i-1])<0.1:
# #                 print ("end for")
#                 break
                
# #         matplotlib.rcParams['figure.figsize'] = [50, 50]
# #         plt.plot(costs)
# #         plt.title("Costs")
# #         plt.show()
        
# #       random_colors = np.random.random((K,3))
# #       colors = R.dot(random_colors)
# #       plt.scatter(X[:,0], X[:,1], c=colors)
# #       plt.show()


    matplotlib.rcParams['figure.figsize'] = [80, 80]
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.show()
       


# In[43]:


def main():
    
    urlclean = urlfinder
    filtertweet = filter_tweet
    df = pd.read_csv('C:/Users/Aashya.Khanduja/Documents/Aashya/Machine Learning/Unsupervised Learning/tweets.csv')

#     count = df["text"].count
#     print (count)
#     a = df.shape
#     print (a)
    
    count =len(df.index)
    print(count) 
#     X = tfidf
#     print (X)
    
    D=2 #"Visualisation"
    s=2 # How far apart are the means"

# Defining the means below*
    mu1= np.array([0,0]) #mean at Origin"
    mu2 = np.array ([s,s]) #""
    

    N = count #"Samples == 300 samples for each class"
    X= np.zeros((N,D))
    X[:600, :] = np.random.randn(600,D) +mu1 #random noise + first mean
    X[600:, :] = np.random.randn((count - 600),D) +mu2 #random noise + third mean

#     plt.scatter(X[:,0], X[:,1])
#     plt.show()
    
    K=2 #number of clusters
    plot_k_means(X,K)
       


# In[44]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:




