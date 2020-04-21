##
import pandas as pd
import numpy as np

df = pd.read_csv("../notebooks/titanic/train.csv")
##
df.to_parquet("titanic.pq")

##
n = 1000000
rep = int(np.ceil(n / len(df)))

df = pd.concat([df] * rep)

df = df.sample(frac=1).reset_index(drop=True)[:n]

df.to_parquet("titanic_1m.parquet")


##
df.memory_usage(deep=True).sum()

##
