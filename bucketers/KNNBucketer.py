import pandas as pd
import numpy as np
from time import time
import sys
from sklearn.neighbors import NearestNeighbors

class KNNBucketer(object):
    
    def __init__(self, encoder, n_neighbors, algorithm='auto'):
        self.encoder = encoder
        self.bucketer = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
        self.potential_neighbors_index = None
    
    def fit(self, X, y=None):
        
        dt_encoded = self.encoder.fit_transform(X)
        self.potential_neighbors_index = dt_encoded.index
        
        self.bucketer.fit(dt_encoded)
        
        return self
    
    
    def predict(self, X, y=None):
        
        dt_encoded = self.encoder.transform(X)
        
        _, indices = self.bucketer.kneighbors(dt_encoded)
        
        return [self.potential_neighbors_index.iloc[knn_idxs] for knn_idxs in indices]
    
    
    def fit_predict(self, X, y=None):
        
        self.fit(X)
        return self.predict(X)