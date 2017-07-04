import math
from PIL import Image,ImageDraw,ImageFont
PATH="../CIKM2017_train/"
testPath="../CIKM2017_testA/"
fr = open(PATH + "train.txt", 'r')

index=0
dbZ_colors=[
			(255,255,255),#-1
			(0,255,255),#dbZ 0~5...and so on
			(0,151,255),
			(0,0,255),
			(0,255,0),
			(0,201,0),
			(0,151,0),
			(255,255,0),
			(255,201,0),
			(255,121,0),
			(255,0,0),
			(201,0,0),
			(151,0,0),
			(255,0,255),
			(151,0,255)
			]
font = ImageFont.truetype("arial.ttf", 46)
for line in open(PATH + "train.txt", 'r'):
	index+=1
	print(index)
	if(index<10):
		continue
	train_datas=fr.readline()
	label=(train_datas.split(",")[1])
	radar_maps=train_datas.split(",")[2]
	radar_map=radar_maps.split(" ")
	maxdBZ=0
	for T in range(15):
		if(T != 14 ):
			continue
		for H in range(4):
			im = Image.new( "RGB", (101,101) )
			for Y in range(101):
				for X in range(101):
					i=T*4*101*101+H*101*101+Y*101+X
					dBZ = float(radar_map[i])
					if(int(radar_map[i])==-1):
						color=(0,0,0)
					elif(dBZ==0):
						color=(255,255,255)
					else:
						color_index=int(dBZ/5)+1
						if(color_index>=len(dbZ_colors)):
							color_index=len(dbZ_colors)-1
						color=dbZ_colors[color_index]
					im.putpixel( (X,Y), color)
			draw = ImageDraw.Draw(im)
			draw.text((21, 27),label,(0,0,0),font=font)
			im.save( "vis_train/I{}T{}H{}.jpg".format(index,T+1,H+1) )
		
fr.close()