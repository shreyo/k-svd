import numpy as np
import math
import sys

sites = 2 #total number of sites
n = 2	#number of rows of Y
s = 5	#number of columns of Y, or total number of samples
k = 3	#number of atoms in teh dictionary
td = int(sys.argv[1])
tp = int(sys.argv[2])
tc = int(sys.argv[3])
outFile = 'rmse_td_'+ str(td)+ '_tp_'+ str(tp) + '_tc_' + str(tc)
path = '/Users/romi/Documents/cloud_fall15/code/tiny/output/td_' + str(td) + '_tp_' + str(tp) + '_tc_' + str(tc) + '/'
rmse = []

for i in range(sites):
	y_file = path + 'Site_' + str(i) + '_Y.txt'
	x_file = path + 'Site_' + str(i) + '_X.txt'
	d_file = path + 'Site_' + str(i) + '_D.txt'
	y_arr = np.zeros((n,s))
	indx = 0
	with open(y_file, 'r') as fin1:
		for line in fin1:
			words = line.rstrip().split(" ")
			arr =  np.array([float(x) for x in words[1].split(',')])
			y_arr[:,indx] = arr
			indx = indx + 1

	x_arr = np.zeros((k,s))
	indx = 0
	with open(x_file, 'r') as fin2:
	 	for line in fin2:
	 		arr =  np.array([float(x) for x in line.split(',')])
	 		x_arr[indx, :] = arr
	 		indx = indx + 1

	d_arr = np.zeros((n,k))
	indx = 0
	with open(d_file, 'r') as fin3:
		for line in fin3:
			arr =  np.array([float(x) for x in line.split(',')])
			d_arr[indx, :] = arr
			indx = indx +1

	d_dot_x = np.dot(d_arr, x_arr)
	diff = np.zeros((n,s))
	for a in range(s):
		if len(np.nonzero(d_dot_x[:,a])[0]) ==0:
			print "error in col " + str(a)
			#continue
		diff[:,a] = y_arr[:,a] - d_dot_x[:,a]
	#print diff


	# tot = 0
	# for d1 in range(n):
	# 	for d2 in range(s):
	# 		tot = tot + diff[d1, d2] ** 2
	# avg_e = tot/(n*s)


	rmse = []
	for j in range(n):
		tot = 0
		for x in range(s):
			tot = tot + diff[j,x]**2
		rmse.append(math.sqrt(tot/s))

with open(outFile, 'wb') as fout:
    for r in rmse:
        fout.write(str(r) + "\n")
