import math
PATH="../CIKM2017_train/"
testPath="../CIKM2017_testB/"

fr = open(PATH + "train.txt", 'r')
fw = open(PATH + "train_pre_v2.csv", 'w')
index=0

fw.writelines("label,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15"+"\n")
for line in open(PATH + "train.txt", 'r'):
	index+=1
	print(index)
	train_datas=fr.readline()
	label=float(train_datas.split(",")[1])
	feature=[]
	radar_maps=train_datas.split(",")[2]
	radar_map=radar_maps.split(" ")
	features=[label]
	for T in range(15):
		weights=[]
		for H in range(4):
			for Y in range(101):
				for X in range(101):
					i=T*4*101*101+H*101*101+Y*101+X
					height_weight = (4-H)/4
					distance_weight = float(1 / ((math.sqrt((50-X) ** 2 + (50-Y) ** 2) / 2) + 1))
					weight = float(float(radar_map[i]) * height_weight * distance_weight)
					if(int(radar_map[i])!=-1):
						weights.append(weight)
		feature=float(sum(weights)/len(weights))
		features.append(str(feature))
	out_line=','.join(features)
	fw.writelines(out_line+"\n")
fr.close()
fw.close()


fr = open(testPath + "testB.txt", 'r')
fw = open(testPath + "testB_pre_v2.csv", 'w')
index=0

fw.writelines("T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15"+"\n")
for line in open(testPath + "testB.txt", 'r'):
	index+=1
	print(index)
	train_datas=fr.readline()
	label=float(train_datas.split(",")[1])
	feature=[]
	radar_maps=train_datas.split(",")[2]
	radar_map=radar_maps.split(" ")
	features=[]
	for T in range(15):
		weights=[]
		for H in range(4):
			for Y in range(101):
				for X in range(101):
					i=T*4*101*101+H*101*101+Y*101+X
					height_weight = (4-H)/4
					distance_weight = float(1 / ((math.sqrt((50-X) ** 2 + (50-Y) ** 2) / 2) + 1))
					weight = float(float(radar_map[i]) * height_weight * distance_weight)
					if(int(radar_map[i])!=-1):
						weights.append(weight)
		feature=float(sum(weights)/len(weights))
		features.append(str(feature))
	out_line=','.join(features)
	fw.writelines(out_line+"\n")
fr.close()
fw.close()
