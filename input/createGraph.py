from igraph import*
from pickle import*
import sys
import numpy as np
'''
this function creates an Erdos Renyi graph given the number of sites/nodes.
The output format is a lisy of neighbors for each node
It also creates a doubly stochastic matrix conforming to the topology
'''
nodes = int(sys.argv[1]) #total number of sites
#create graph
g = Graph.Erdos_Renyi(nodes, p=0.9)
print summary(g)
print g.is_connected()

#create a doubly stochastic matrix
alpha = 0.2
W = np.zeros((nodes, nodes))

with open('edgeList.txt', 'wb') as fout:
    for v in range(len(g.vs)):
        neighbors = g.neighborhood(g.vs[v])
        #print neighbors
        outStr = str(v) + " "
        W[v, v] = 1 - (len(neighbors)-1)*alpha # (1- node_degree*alpha)
        for n in range(1, len(neighbors)-1):
            outStr = outStr + str(neighbors[n]) + ","
            W[v, neighbors[n]] = alpha
        outStr = outStr + str(neighbors[len(neighbors)-1]) + "\n"
        W[v, neighbors[len(neighbors)-1]] = alpha
        fout.write(outStr)
print W
with open('stochasticMat.txt', 'wb') as fout2:
	outStr = ""
	for n in range(nodes):
		for n1 in range(nodes-1):
			outStr = outStr + str(W[n,n1]) +","
		outStr = outStr + str(W[n, nodes-1]) +" "
	fout2.write(outStr)


