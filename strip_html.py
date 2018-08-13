from bs4 import BeautifulSoup 
import pandas as pd 


def htmlstripper(inp):
	return BeautifulSoup(inp,"html5lib").text

for i in range(11,47):
	df = pd.read_table("C:\\Users\\ktwic\\Desktop\\BIS2A 2016\\nb data\\lines" +str(i)+".csv", sep = ",", error_bad_lines = False)
	df["body"] = df[df.columns.values[-1]].apply(lambda x: htmlstripper(str(x)))
	print(i)
	df.to_csv("C:\\Users\\ktwic\\Desktop\\BIS2A 2016\\nb data\\lines_cleaned" +str(i) +".csv", mode = "w+")



