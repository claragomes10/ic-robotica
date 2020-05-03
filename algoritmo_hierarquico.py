#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:19:03 2020

@author: clara
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

saida_direita = pd.read_csv('saida_direita.csv')
corredor = pd.read_csv('corredor.csv')
dataset = pd.concat([saida_direita, corredor], axis=0)
dataset.head()

dataset['label'] = dataset['label'].replace('saida_direita',0)
dataset['label'] = dataset['label'].replace('corredor',1)

from sklearn.preprocessing import normalize
data = normalize(dataset)
data = pd.DataFrame(data, columns=dataset.columns)
data.head()

dend = shc.dendrogram(shc.linkage(data, method='ward'))
plt.axhline(y=8, color='r', linestyle='--')

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data)
print(cluster.labels_)
plt.figure(figsize=(10, 7))
plt.scatter(data.iloc[:,0],data.iloc[:,1], c=cluster.labels_, cmap='rainbow')

plt.title('clussters')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()




