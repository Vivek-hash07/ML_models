from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
import pandas as pd
import numpy as np

data = {
    "Speed": [10,20,30,40,50],
    "Distance":[1000,2000,3000,4000,5000]
}

print(data)

df=pd.DataFrame(data)
print(df)

ss = StandardScaler()
x1 = ss.fit_transform(df)
print(x1)

mm = MinMaxScaler()
x2 = mm.fit_transform(df)
print(x2)

rs = RobustScaler()
x3 = rs.fit_transform(df)
print(x3)

n = Normalizer()
x4 = n.fit_transform(df)            
print(x4)