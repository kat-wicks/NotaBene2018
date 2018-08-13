counter = 0
lines = []
i = 0
with open("C:\\Users\\ktwic\\Desktop\\BIS2A 2018\\bis2a.txt", "r", encoding = "utf-8") as f:
	for line in f:
		if counter%10000 == 0:
			with open("lines"+str(i)+".txt", "w+", encoding = "utf-8") as w:
				w.writelines(lines)
			lines=[]
			w.close()
			i+=1
		lines.append( ' '.join(line.split())+ '\n')

		counter +=1
f.close()

