# g24ai1046 data_utils.py

import pandas as pd
import numpy as np

def get_boston_housing():

    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
    target = raw.values[1::2, 2]
    
    features = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
                'DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=features)
    df['MEDV'] = target
    return df
    