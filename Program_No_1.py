import pandas as pd
import numpy as np

print("Version of numpy is : ", np.__version__)
print("Version of Pandas is : ",pd.__version__)


#numpy operations

a = [1,2,3,4]
print(a)
print("type of a is : ",type(a))

b = np.array([1,2,3,4])
print(b)
print("type of b is ",type(b))
print("Dimension of b is ",b.ndim)

c = np.array([[1,2,3,4],[5,6,7,8]])
print(c)
print("type of c is : ",type(c))
print("Dimension of c is : ",c.ndim)


d = np.array([[[1,2,3,4],[5,6,7,8]],[[11,22,33,44],[55,66,77,88]]])
print(d)
print("type of d is : ",type(d))
print("Dimension of d is : ",d.ndim)

#pandas operations
data = {'data':['alice','bob','charlie'],'sec':['A','B','C']}
df = pd.DataFrame(data)
print(df)
print("Columns of DataFrame : ",df.columns)