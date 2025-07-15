import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

data = {
    'Data1' : [10,20,30,40,50],
    'Data2' : [1200,1500,3500,4000,8000]
}

print("Data is : ",data)

df = pd.DataFrame(data)
print(df)

plt.figure(1)
plt.bar(df['Data1'],df['Data2'])
plt.title('data before scaler with plt')
plt.xlabel('Data1')
plt.ylabel('Data2')

plt.figure(2)
plt.scatter(data['Data1'],data['Data2'],color='blue',alpha=0.7,edgecolors='Black')
plt.xlabel('Data1')
plt.ylabel('Data2')
plt.title("Scaltter plot before scalar using matplotlib")

rs = RobustScaler()
rdata = rs.fit_transform(df)
dfr = pd.DataFrame(rdata, columns=df.columns)
print(dfr)

plt.figure(3)
sns.set_theme(style='darkgrid')
sns.barplot(x='Data1',y='Data2',data=dfr)
plt.title("Data after Scaler with sns")

plt.figure(4)
sns.scatterplot(data=dfr,x='Data1',y='Data2',color='red')
plt.title("Scatterplot after scaler using seaborn")



plt.show()