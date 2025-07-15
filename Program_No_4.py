import pandas as pd


df=pd.read_csv(r"cars.csv")

#install excel package à python -m pip install openpyxl

pd.set_option("display.expand_frame_repr", True)

print(df)

print("columns of df")

print(df.columns)

print("describe of df")

print(df.describe())

print("head of df")

print(df.head)