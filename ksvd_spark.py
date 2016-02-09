from __future__ import print_function
import sys
from pyspark import SparkContext, SparkConf
import time
import numpy as np
from sklearn.decomposition import SparseCoder
N = 2 # dimension of each sample. Dimension of Y (NxS)
K = 3  # #of atoms in the dictionary. Dimension of D (NxK)
site_no = 2 #total #of sites
alpha = 0.2 #weights of edges to neighbors, same parameter is used to design the doubly stochastic matrix W
t_d = 1 #iterations of K-SVD
t_p =  1 #iterations of distributed power
t_c = 3 #iterations of consensus averaging

def parseVector(line):
    words = line.rstrip().split(" ")
    arr =  np.array([float(x) for x in words[1].split(',')])
    arrSz = arr.size
    arr_mod = np.reshape(arr, (arrSz,1))
    return (int(words[0]), arr_mod) #word[0] is the site index where this data is stored


def parseStochMat(line):
    words = line.rstrip().split(" ")
    fin_arr =np.zeros((site_no, site_no))
    for w in range(len(words)):
        row = words[w].split(",")
        for r in range(len(row)):
            fin_arr[w, r] = float(row[r])
    #multiply fin_array t_c number of times..to be used later
    W_tc = np.zeros(fin_arr.shape)
    for t in range(t_c):
        W_tc = np.dot(fin_arr, fin_arr)

    #generate N copies of it, one for every site
    vals = []
    for n in range(site_no):
        vals = vals + [(n, W_tc[1,:])]
    return vals


def genD_X(p):

    Site = p[0]
    Y = p[1][0]
    D = p[1][1]

    n_nonzero = 1
    algo = 'omp'
    #Y_mod=np.reshape(Y, (1,N))
    Y_mod = Y.T
    D_mod = D.T
    #D = np.random.randn(K,N)
    coder = SparseCoder(dictionary=D_mod, transform_n_nonzero_coefs=n_nonzero, transform_alpha=None, transform_algorithm=algo)
    X = coder.transform(Y_mod)
    X_mod = X.T
    #X-returned has shape (k,s)
    #shape of Y is (n,s) while D is (n,k)
    return (Site, (Y,D,X_mod))

def genM((Site, (Y,D,X,k))):
    W = np.nonzero(X[k,:])[0] #array of indices of non zero elements of X at column k
    w_sz = W.size

    S = Y.shape[1]
    Sigma = np.zeros((S, w_sz))

    for w in range(w_sz):
        Sigma[W[w], w] = 1

    pre_sum = 0
    post_sum = 0

    for j in range(0, k-1):
        X_row = X[j,:]
        X_reshape = np.reshape(X_row, (1,X_row.size))
        D_col = D[:,j]
        D_reshape = np.reshape(D_col, (D_col.size,1))
        pre_sum = pre_sum + np.dot(D_reshape, X_reshape)
    for j in range(k+1, K):
        X_row = X[j,:]
        X_reshape = np.reshape(X_row, (1,X_row.size))
        D_col = D[:,j]
        D_reshape = np.reshape(D_col, (D_col.size,1))
        post_sum = post_sum + np.dot(D_reshape, X_reshape)
    E = np.dot((Y - (pre_sum + post_sum)), Sigma) #reduced error matrix

    M = np.dot(E, E.T)

    return (Site, (Y,D,X,k,E,M))

def genQ((Site, (Y,D,X,k,E,M))):
    #set the seed to be the same, so that every site generates the same initial eigen vector estimate
    np.random.seed(1)
    q_int = np.random.uniform(0,1, (M.shape[0],1))
    return (Site, (Y,D,X,k,E,M,q_int))



def stencil((Site, (z,neigh))):
    vals = []
    for n in np.nditer(neigh):
        vals = vals + [(int(n), np.multiply(alpha,z))] #send your value multiplied by weight to your neighbors
        #Note: weight to all your neighbirs is considered constant = alpha as specified in the doubly stocahstic matrix W.
        #If the matrix W was generated such that these weights are not constant, you need to refer to the complete matrix to figure out the corrsponding weight
    return vals

def genV(p):

    Site = p[0]
    V = p[1][1]
    Y = p[1][0][0]
    E = p[1][0][4]

    D_red = np.reshape(V, (V.size,1))
    if(E.shape[1]) == 0:
       E_mod = np.zeros((V.shape[0], 1))
    else:
        E_mod = E

    v_transp = np.reshape(V,(1,V.size))
    X_red = np.dot(v_transp, E_mod)

    return((Site, (D_red,X_red)))


def updateQ(p):
    Site = p[0]
    W_tc = p[1][1]
    Z=p[1][0]

    V = Z/W_tc[Site]
    arg = 0
    for v in range(V.size):
        arg = arg + V[v]*V[v]

    if arg !=0:
        V = V/np.sqrt(arg)

    return((Site, V)) #normalized V is the new estimate of eigenvector


def updateD_X(p):
    Site = p[0]
    Y =p[1][0][0]
    D = p[1][0][1]
    X = p[1][0][2]
    k = p[1][0][3]

    D_mod = p[1][1][0]
    X_mod = p[1][1][1]

    D[:,k] = D_mod[:,0]
    X[k,:]  = np.zeros((1,Y.shape[1]))
    if(X_mod.size != 0):
        for x in range(len(X_mod[0,:])):
            X[k,x] = X_mod[0,x]

    return((Site, (Y,D,X)))


if __name__ == "__main__":
    startTime = time.time()
    conf = (SparkConf().setAppName("KSVD").set("spark.executor.memory", "6g").set("spark.storage.memoryFraction", "0.4"))

    # Initialize the spark context.
    sc = SparkContext(conf = conf)

    dataNh = sc.textFile(sys.argv[2]).map(parseVector).cache() #used repeatedly in consensus averaging loop, makes sense to cache
    dataW = sc.textFile(sys.argv[3]).flatMap(parseStochMat).reduceByKey(lambda arrA, arrB: np.concatenate((arrA, arrB), axis=1)).cache() #used repeatedly in distributed power iteration
    #dataW is the complete doubly stochastic matrix and is avialable at every site (key-Site, val-matrix).
    #Note, instead of storing the matrix, we store the W^(t_c) as that is what is required for compuattion and we do not want to do matrix multiplication evrytime in the lopp

    D = np.random.randn(N,K) #initialize D
    dataD = sc.textFile(sys.argv[1]).map(parseVector).reduceByKey(lambda arrA, arrB: np.concatenate((arrA, arrB), axis=1)).map(lambda (Site,Y): (Site,(Y,D)))
    #reduce by key would bring all data belonging to 1 site at one place, key = site#
    #associate a D and a X with each data (each data is a column)

    for d in range(t_d):
        dataDX = dataD.map(genD_X) #do sparse coding

        #update dict
        for k in range(K):
            dataM = dataDX.map(lambda (Site, (Y,D,X)): (Site, (Y,D,X,k))).map(genM).map(genQ)
            #genM returns reduced error and M, genQ returns initial estimate of eigenvector(q_int)

            #initialize power distribution
            dataCons = dataM.map(lambda (Site, (Y,D,X,k,E,M,q_int)): (Site, (np.dot(M, q_int))))
            dataRest = dataM.map(lambda (Site, (Y,D,X,k,E,M,q_int)): (Site, (Y,D,X,k,E))).cache() #this is used repeatedly in loop and makes sense to cache

            for p in range(t_p):
               #everything done locally upto now
                #do consensus averaging
                for c in range(t_c):
                    stencilParts =  dataCons.join(dataNh).flatMap(stencil) #add the neighbor info and do stencil operation to neighbors
                    dataCons = stencilParts.reduceByKey(lambda x,y:x+y) #add all values you received from your neighbors


                dataCons = dataCons.join(dataW).map(updateQ)
                #end of distributed power

            #update k'th col fo D and kth col of X
            dataDX = dataRest.join(dataCons).map(genV) #note, onl teh reduced error matrix E is required from dataRest in this step.
            #will it be faster to dp a map to only extract E, before merging with dataCons?
            #returns D_reduced kth col and X_th red kth row

            dataDX= dataRest.join(dataDX).map(updateD_X)
            #update the kth col of D and kth row of X

        #end of update of all cols of D
        dataD = dataDX.map(lambda (Site, (Y,D,X)): (Site, (Y,D)))


    #collect and save output
    dataOut = dataDX.map(lambda (Site, (Y,D,X)): (Site, (D,X))).collect()

    for (site, (output)) in dataOut:

        D_final = output[0]
        X_final = output[1]
        x_file = 'Site_' + str(site) + '_X.txt'
        d_file = 'Site_' + str(site) + '_D.txt'
        np.savetxt(x_file, X_final, delimiter=",")
        np.savetxt(d_file, D_final, delimiter=",")


    sc.stop()
    print (time.time() - startTime)


