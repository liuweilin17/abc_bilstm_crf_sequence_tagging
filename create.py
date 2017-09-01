#encoding=utf-8
import numpy as np 

#a = ['中国','利润率','腾讯','旅游业','农业产品']
#b = [[1,1], [2,2,2], [3,3], [4,4,4], [5,5,5,5]]

input_file = "/home/wlliu/entity_recognition/data/word_2col_data/train_file"
tmp = ""
label = []
data_list = []
label_list = []
count = 1
with open(input_file) as f:
	for line in f:
		if(line!="\n" and len(line.split("\t")) == 2):
			line_arr = line.split("\t")
			tmp += line_arr[0]
			print("count:" + str(count) + "--" + line)
			#print(line_arr[1].strip())
			label.append(int(line_arr[1].strip()))
		else:
			data_list.append(tmp)
			label_list.append(label)
			tmp = ""
			label = []
		count += 1
	if(tmp != ""):
			data_list.append(tmp)
			label_list.append(label)

np.save('data.npy', data_list)
np.save('label.npy', label_list)

