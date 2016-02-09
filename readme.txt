cloud k-svd is written in Python as a standalone application (ksvd_spark.py)
This distributed version of k-svd is based on the paper "Cloud K-SVD: A Collaborative Dictionary Learning Algorithm for Big, Distributed Data" by Haroon Raja and Waheed U. Bajwa (http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7219480)

#note on input files:
The input folder contains the requisite input files to run
1. raw data file, Y
2. topology file indicating neigbors for each site
3. stochsatic matrix file, representing the complete topology 

It also contains codes for generating these inputs namely:
1. gen_synthetic.py: This function creates a synthetic data file, following the procedure highlighted by the authors of the cloud-ksvd paper
2. createGraph.py: Creates an Erdos-Renyi random graph and takes the number of nodes as an input argument. It outputs the topology file (edgeList.txt) and the doubly stochastic matric (stochasticMat.txt)
3. gen_stochastic.py: This code creates a standalone stochastic matrix, given the graph (edgeList.txt)

#to run type:

/<path-to-spark>/bin/spark-submit ksvd_spark.py input/Y_2_10.txt input/edgeList.txt input/stochasticMat.txt --num-executors 6  --executor-memory 6G --executor-cores 6 --driver-memory 6G --master local[*]

#Note: the requisite parameters are the 3 input files. The rest of the parameters are optional and depend on the memory available on the machine the code is being run on
#Also the iteartions for dictionary learning (td), power iteration (tp) and consensus averaging (tc) are defined at the start of the file.


#Parsing the output: ksvd_spark.py creates 2 output file for each site, (local dictionary + local sparse representation). These files are saved in the output folder.  

In order to run the parsing file, you need to create a new directory name td_<td-val>_tp_<tp-val>_tc_<tc-val>, move the output files to that directory. In addition, it needs the original local data files (Y's) at each site, which are not saved from the output of ksvd_spark.py in order to save on time and memory.
For convenience, an example of Y's D's and X's are provided in for td=2, tp=2,tc=5.
The parsing file parseOut.py, takes in 3 command line inputs--> the valus of td, tp and tc respectively and produces the corresponding rmse error.
