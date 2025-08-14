# scripts/split_erlangen.py (optionnel)
import pandas as pd

erl = {"erlangen2011_2","erlangen2012_1","erlangen2012_2",
       "erlangen2013_1","erlangen2013_2","erlangen2014_1"}

df = pd.read_csv("data/real/real.csv")
df.columns = df.columns.str.strip()
out_e = df[df["name"].isin(erl)].copy()
out_n = df[~df["name"].isin(erl)].copy()
out_e.to_csv("data/real/real_only_erlangen.csv", index=False)
out_n.to_csv("data/real/real_no_erlangen.csv", index=False)
print("OK.")
