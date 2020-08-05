##
import pandas as pd
import altair as alt
from json import loads

G = 1024 * 1024 * 1024
##
df = pd.read_json("results/adult_dask_1.json", lines=True)
df["DVM"] = df["mem_size"] / df["memory"]
##
alt.Chart(df[df["DVM"] < 1], title="Plot(df) Comparison").mark_line(point=True).encode(
    y=alt.Y("elapsed", title="Elapsed (s)"),
    x=alt.X("DVM", title="Dataset Size / Memory Size"),
    color="name:N",
    tooltip="elapsed",
    column=alt.Column("partition:O", title="Data Loading Mode"),
)

##
alt.Chart(
    df[df.memory == 1 * G], title="Plot Comparison: 8G Mem/8 CPU/16 Data Partition"
).mark_bar().encode(
    y="name:N",
    x=alt.X("elapsed", title="Elapsed (s)"),
    color="name",
    tooltip="elapsed",
    row="nrow:Q",
    column=alt.Column("partition:O", title="Data Loading Mode"),
).resolve_scale(
    x="independent"
)
##
pdf = df.pivot_table(
    index=["Mem", "CPU", "Dataset", "Partition", "Row", "Col", "Mode"],
    columns="Func",
    values="Elapsed",
).reset_index()


##
