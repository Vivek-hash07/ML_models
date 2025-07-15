import sklearn as sk
from sklearn import datasets
import pandas as pd

print("Scikit-learn version:", sk.__version__)
iris = datasets.load_iris()

df = pd.DataFrame(iris.data,columns=iris.feature_names)

print(df)

print(df.columns)

print(df.describe())

print(df.info())

print(df.head())

print(df.tail())
