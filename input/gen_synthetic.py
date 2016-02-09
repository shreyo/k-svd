import numpy
'''
this function generates a synthetic dataset similar to that used by cloud-ksvd
'''
n = 2
k = 5
k_i = 4
s_i=5
sites = 2

s = s_i * sites #each site has s_i samples

path = '/Users/romi/Documents/cloud_fall15/code/tiny/input'
out_file = path + '/Y_' + str(n) + '_' + str(s) + '.txt'
dict = numpy.zeros((n,k))
#generate dictionary atoms unifrmly distributed in the unit sphere (values from 0 and 1)
for i in range(k):
    dict[:, i] = numpy.random.uniform(0, 1, n)

with open(out_file, 'wb') as fout:
    for i in range(sites): #for each site, generate a random input (y_i)
        sub_dict = numpy.zeros((n,k_i))
        rand_indices = numpy.random.randint(0, k, k_i) #choose 45 random sub cloums of dict
        for j in range(len(rand_indices)):
            sub_dict[:,j] = dict[:, rand_indices[j]]
        y_i = numpy.zeros((n,s_i))

        for j in range(s_i):
            #create corresponding x_i[:,j]
            x_i = numpy.zeros(k_i)
            #choose 3 random indices and put non zero values in them to form the final x_i
            rand_indices = numpy.random.randint(0, k_i, 3)
            for x in  range(3):
                x_i[rand_indices[x]] = numpy.random.uniform(0, 1)
            noise = numpy.random.normal(0,0.01, n) #gaussian noise
            y_col = numpy.dot(sub_dict, x_i) + noise
            print y_col.shape
            out_str = ""
            for indx in range(len(y_col) -1):
                out_str = out_str + str(y_col[indx]) + ","
            out_str = out_str + str(y_col[len(y_col)-1])+ "\n"
            print out_str
            fout.write(str(i) + " " + out_str)



