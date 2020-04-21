import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from time import time


df = pd.read_parquet("automobile_2500.pq")
print("DataFrame shape:", df.shape)

then = time()

profile = ProfileReport(df, title="Pandas Profiling Report")
elapsed = time() - then

print(f"Pandas Profiling Elapsed: {elapsed}s")

profile.to_file(output_file="your_report.html")
