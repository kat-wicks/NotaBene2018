import pandas as pd
import numpy as np
import datetime 





df = pd.read_csv("merged_2.csv")
df.fillna("A2")
def mapping(x):
	dict_map = {"A1": -1, "A2": -1, "C1": 0, "C2": 0, "I":1}
	if x in dict_map:
		return dict_map[x]
	else:
		return 0
df["label"] = df.word_tag.apply(lambda x: mapping(x))
def ce_avg(df,row):
	print(row)
	author = row["author_id"]
	time = row["ctime"]
	selected =df.loc[(df['author_id'] == author) & (df['ctime']< time)]
	if selected.shape[0] >0:
		return sum(selected["label"].values)/(selected.shape[0])
	else:
		return -0.5

df.fillna(0)
df["ce_avg"] = df.apply(lambda row: ce_avg(df, row),axis=1, raw = True)
df.to_csv("merged_2.csv")