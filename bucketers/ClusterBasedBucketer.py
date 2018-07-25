import pandas as pd
import numpy as np
from time import time
import sys

class ClusterBasedBucketer(object):
    
    def __init__(self, encoder, clustering):
        self.encoder = encoder
        self.clustering = clustering
        
    
    def fit(self, X, preencoded=False):
        
        if not preencoded:
            X = self.encoder.fit_transform(X)
        
        self.clustering.fit(X)
        
        return self
    
    
    def predict(self, X, preencoded=False):
        
        if not preencoded:
            X = self.encoder.transform(X)
        
        return self.clustering.predict(X)
    
    
    def fit_predict(self, X, preencoded=False):
        
        self.fit(X, preencoded)
        return self.predict(X, preencoded)