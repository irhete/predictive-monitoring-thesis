import pandas as pd
import numpy as np
from time import time
import sys

class StateBasedBucketer(object):
    
    def __init__(self, encoder):
        self.encoder = encoder
        
        self.dt_states = None
        self.n_states = 0
        
    
    def fit(self, X, preencoded=False):
        
        if not preencoded:
            X = self.encoder.fit_transform(X)
        
        self.dt_states = X.drop_duplicates()
        self.dt_states = self.dt_states.assign(state = range(len(self.dt_states)))
        
        self.n_states = len(self.dt_states)
        
        return self
    
    
    def predict(self, X, preencoded=False):
        
        if not preencoded:
            X = self.encoder.transform(X)
        
        dt_transformed = pd.merge(X, self.dt_states, how='left')
        dt_transformed.fillna(-1, inplace=True)
        
        return dt_transformed["state"].astype(int).as_matrix()
    
    
    def fit_predict(self, X, preencoded=False):
        
        self.fit(X, preencoded)
        return self.predict(X, preencoded)