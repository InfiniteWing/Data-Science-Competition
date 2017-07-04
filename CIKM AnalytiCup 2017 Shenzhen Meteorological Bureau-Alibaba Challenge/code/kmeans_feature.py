import math
from sklearn.cluster import KMeans
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
trainPATH="../CIKM2017_train/"
testPath="../CIKM2017_testB/"

fr = open(trainPATH + "train.txt", 'r')
train_features=[]
index=0
for line in open(trainPATH + "train.txt", 'r'):
	index+=1
	print(index)
	train_datas=fr.readline()
	label=float(train_datas.split(",")[1])
	radar_maps=train_datas.split(",")[2]
	radar_map=radar_maps.split(" ")
	maxdBZ=0
	for T in range(15):
		if(T != 0 ):
			continue
		for H in range(4):
			if(H!=0):
				continue
			radar_map_features=[]
			for Y in range(101):
				for X in range(101):
					i=T*4*101*101+H*101*101+Y*101+X
					dBZ = float(radar_map[i])
					radar_map_features.append(dBZ)
			train_features.append(radar_map_features)
train_features=np.array(train_features)
kmeans = KMeans(n_clusters=10, random_state=0).fit(train_features)
train = pd.read_csv(trainPATH+'train_pre_v2.csv')
train["kmeans"]=kmeans.labels_
train.to_csv(trainPATH+'train_pre_v2_kmeans.csv', index=False)


fr = open(testPath + "testB.txt", 'r')
test_features=[]
index=0
for line in open(testPath + "testB.txt", 'r'):
	index+=1
	print(index)
	train_datas=fr.readline()
	label=float(train_datas.split(",")[1])
	radar_maps=train_datas.split(",")[2]
	radar_map=radar_maps.split(" ")
	maxdBZ=0
	for T in range(15):
		if(T != 0 ):
			continue
		for H in range(4):
			if(H!=0):
				continue
			radar_map_features=[]
			for Y in range(101):
				for X in range(101):
					i=T*4*101*101+H*101*101+Y*101+X
					dBZ = float(radar_map[i])
					radar_map_features.append(dBZ)
			test_features.append(radar_map_features)
test_features=np.array(test_features)
pred=kmeans.predict(test_features)
test = pd.read_csv(testPath+'testB_pre_v2.csv')
test["kmeans"]=kmeans.labels_
test.to_csv(testPath+'testB_pre_v2_kmeans.csv', index=False)