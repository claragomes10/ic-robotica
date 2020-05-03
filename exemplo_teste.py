#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:03:30 2020

@author: clara
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

saida_direita = pd.read_csv('saida_direita.csv')
corredor = pd.read_csv('corredor.csv')
data = pd.concat([saida_direita, corredor], axis=0)
dataset = data.iloc[:,:].values

dendrogram = sch.dendrogram(sch.linkage(dataset, method = 'ward'))
plt.axhline(y=6, color='r', linestyle='--')

plt.scatter(dataset[:,:][:,0],dataset[:,:][:,1])

hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(dataset)

plt.scatter(dataset[y_hc == 0,0], dataset[y_hc == 0,1], s=100, c = 'red', label = 'Cluster 1')
plt.scatter(dataset[y_hc == 1,0], dataset[y_hc == 1,1], s=100, c = 'grey',label = 'Cluster 2')
plt.title('clussters')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
