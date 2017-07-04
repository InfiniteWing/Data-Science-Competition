import math
from sklearn.cluster import KMeans
import numpy as np
trainPATH="../CIKM2017_train/"
testPath="../CIKM2017_testA/"

fr = open(trainPATH + "testA.txt", 'r')
train_features=[]
index=0
for line in open(trainPATH + "train.txt", 'r'):
	index+=1
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
kmeans = KMeans(n_clusters=6, random_state=0).fit(features_np)

fr = open(testPath + "testA.txt", 'r')
predict_features=[]
index=0
for line in open(testPath + "testA.txt", 'r'):
	index+=1
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
			predict_features.append(radar_map_features)
predict_features=np.array(predict_features)
kmeans = KMeans(n_clusters=6, random_state=0).fit(features_np)
for i,lb in enumerate(kmeans.labels_):
	print("{},{}".format(i,lb))

fr.close()