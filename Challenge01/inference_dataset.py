import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./2023_train.csv")
df = df.iloc[-48:, :12]
df = df.fillna(0)

df.to_csv("inference.csv", index=False)