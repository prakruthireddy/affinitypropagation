import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_random_state
import csv
from sklearn.metrics.pairwise import euclidean_distances
import math
'''
convergence_iter: number of iterations that the estimated exemplars have to stay fixed for the algorithm to terminate.
'''

class AffinityPropagation():
    def __init__(
            self,
            damping = 0.8,
            affinity="negative squared euclidean distance",
            convergence_iter=10,
            max_iter=2000,
            preference = None,
            random_state = 0,
            nodeData_fileName = "nodeData.txt",
            edgeList_fileName = "edgeList.txt",
            dis = 3):
        self.damping = damping
        self.affinity = affinity
        self.convergence_iter= convergence_iter
        self.max_iter = max_iter
        self.preference = preference
        self.random_state = random_state
        self.nodeData_fileName = nodeData_fileName
        self.edgeList_fileName = edgeList_fileName
        self.dis = dis
        
    def _similarity_matrix(self):
        """ Computing the distance from every edge to every other edge in the dataset"""
        with open(self.nodeData_fileName, newline='') as csvfile:
            nodeData = np.array(list(csv.reader(csvfile, delimiter=' ')), float)
        with open(self.edgeList_fileName, newline='') as csvfile:
            edgeData = np.array(list(csv.reader(csvfile, delimiter=' ')), int)
            
        nodeIDs = np.array(nodeData[:,0], int)
        number_of_nodes = len(nodeIDs)
        number_of_edges = len(edgeData)
        nodeMap = dict(zip(nodeIDs, np.arange(number_of_nodes)))
        
        distanceMatrix = np.zeros((number_of_edges, number_of_edges))
        similarityMatrix = np.zeros((number_of_edges, number_of_edges))

        for i in range(number_of_edges-1):
            edge1_v1 = edgeData[i,0]
            edge1_v2 = edgeData[i,1]
            nodeValues_v1 = nodeMap[edge1_v1]
            v1_xy = [nodeData[nodeValues_v1, 1], nodeData[nodeValues_v1, 2]]
            nodeValues_v2 = nodeMap[edge1_v2]
            v2_xy = [nodeData[nodeValues_v2, 1], nodeData[nodeValues_v2, 2]]
            for j in range(i + 1, number_of_edges):
                edge2_v1 = edgeData[j,0]
                edge2_v2 = edgeData[j,1]
                nodeValues_v3 = nodeMap[edge2_v1]
                v3_xy = [nodeData[nodeValues_v3, 1], nodeData[nodeValues_v3, 2]]
                nodeValues_v4 = nodeMap[edge2_v2]
                v4_xy = [nodeData[nodeValues_v4, 1], nodeData[nodeValues_v4, 2]]
                minDist = (v1_xy[0] - v3_xy[0])**2 + (v1_xy[1] - v3_xy[1])**2 
                newDist = (v1_xy[0] - v4_xy[0])**2 + (v1_xy[1] - v4_xy[1])**2 
                if newDist < minDist:
                    minDist = newDist
                newDist = (v2_xy[0] - v3_xy[0])**2 + (v2_xy[1] - v3_xy[1])**2 
                if newDist < minDist:
                    minDist = newDist
                newDist = (v2_xy[0] - v4_xy[0])**2 + (v2_xy[1] - v4_xy[1])**2 
                if newDist < minDist:
                    minDist = newDist

                distanceMatrix[i,j] = minDist
                distanceMatrix[j,i] = minDist
                similarityMatrix[i,j] = minDist
                similarityMatrix[j,i] = minDist
                
                if minDist > self.dis * self.dis:
                    val = 1e10
                    similarityMatrix[i,j] = val
                    similarityMatrix[j,i] = val
                
        return distanceMatrix, similarityMatrix
        
    def _affinity_propagation(self, S):
        n_samples = S.shape[0]
        # Setting preferences. 
        if self.preference == None:
            self.preference = np.min(S)
            print("pref", self.preference)
        # Avoiding a small amount of noise to avoid degenerate solutions
        self.random_state = check_random_state(self.random_state)
        
        # Adding preferences to the diagonal of S
        
        S.flat[::(n_samples + 1)] = self.preference
        S += ((np.finfo(np.double).eps * S + np.finfo(np.double).tiny * 100) * self.random_state.randn(n_samples, n_samples))


        f = open("file.txt", "w")
        
        #print("Similarity Matrix: ", S)
        #print("Max", np.max(S))
        #print("Min:", np.min(S))
        #f.write("\nSimilarity Matrix: ")
        #f.write(np.array_str(S))
        # Initializing Availability and Responsibility Matrices
        A = np.zeros((n_samples, n_samples))
        R = np.zeros((n_samples, n_samples))
        
        e = np.zeros((n_samples, self.convergence_iter))
        
        indices = np.arange(n_samples) #[0,1,2,...,n_samples-1]

        E = None
        for i in range(self.max_iter):
            #print("\nIteration round number: ", i)
            #f.write("\n\nIteration round number: ")
            #f.write(str(i))
            # Computing Responsibilities
            Rold = R
            # Adding the Availability and Similarity matrix
            AS = A + S
    
            I = np.argmax(AS, axis=1) # Take the index of max value across each row 
            Y1 = np.amax(AS, axis=1) # Take the max value of each row
            #print("y1", Y1)
            
            AS[indices, I] = -np.inf # Replace the max value element with -inf
            Y2 = np.amax(AS, axis=1) # Take the max value of each row again (To get the second max value)
            R = S - Y1.reshape((n_samples,1)) 
            R[indices, I] = S[indices, I] - Y2
            R =  (1-self.damping)*R+(self.damping)*Rold # Dampening Responsibilities.
            #print("\nResponsibility matrix: \n", R)
            #f.write("\nResponsibility matrix: ")
            #f.write(np.array_str(R))
            # Computing Availabilities
            Aold = A
            Rp = np.maximum(R, 0) # Storing all +ve values of R in Rp
            Rp.flat[::n_samples + 1] = R.flat[::n_samples + 1] # Replacing the diagonal elements
            A = np.sum(Rp, axis=0) - Rp
            dA = A.flat[::n_samples + 1]
            A = np.minimum(A,0)
            A.flat[::n_samples + 1] = dA
            A = (1-self.damping)*A+self.damping*Aold # Dampening Availabilities
            #print("\nAvailibility matrix: \n", A)
            #f.write("\nAvailability matrix: ")
            #f.write(np.array_str(A))
            #Check for convergence
            E = (np.diag(A) + np.diag(R)) > 1e-10 # Check if the sum of diagonal elements are positive. 
            F = np.diag(A) + np.diag(R)   
            print("positive diagonal A+R", F[E])
            e[:, i % self.convergence_iter] = E # Check if the exemplars have stayed unchanged for the convergence_iter number of iterations
            K = np.sum(E) # Number of exemplars computed.
            # Check if the algorithm converged i.e., the exemplars have remained unchanged or the max number of iterations have been reached
            if i >= self.convergence_iter:     
                se = np.sum(e, axis=1)
                unconverged = (np.sum((se == self.convergence_iter) + (se == 0)) != n_samples)
                if (not unconverged and (K > 0)):
                    never_converged = False
                    break
        else:
            never_converged = True
        I = np.flatnonzero(E) # Get the indices of the non-zero elements of E. 
        K = I.size # Get number of clusters.
        Q = A + R
        print("AR on 714]", Q[714])
        print("max", np.max(Q[714]))
        print("arg", np.argmax(Q[714]))
        print("AR[714,188]", Q[714,188])
        # If the algorithm has convereged then assign every element to a cluster
        if K > 0 and not never_converged:  
            c = np.argmax(S[:, I], axis=1) # Assign each element to a 
            
            c[I] = np.arange(K)  # Identify clusters
            
            for k in range(K):
                cluster = np.where(c == k)[0]
                new_max = np.argmax(np.sum(S[cluster[:, np.newaxis], cluster], axis=0))
                I[k] = cluster[new_max]
            
            c = np.argmax(S[:, I], axis=1)
            c[I] = np.arange(K)
            labels = I[c]
            cluster_centers_indices = np.unique(labels)
            labels = np.searchsorted(cluster_centers_indices, labels)
        else:
            print("The algorithm has not converged")
            labels = np.array([-1] * n_samples)
            cluster_centers_indices = []
        f.close()
        return cluster_centers_indices, labels, i + 1, K, never_converged
            
        
    def fit(self):
        #X = np.array([[50,-150],[-50,-150],[-150,150],[-150,200],[150,150],[    150,200]])
        #Sm = -euclidean_distances(X, squared=True)
        self.sm, self.similarity_matrix_ = self._similarity_matrix()
        self.similarity_matrix_ = -self.similarity_matrix_
        self.cluster_centers_indices_, self.labels_, self.n_iter_, self.number_of_clusters_, self.converged_ = self._affinity_propagation(self.similarity_matrix_)
        return self 
    
################################################################

nodefile = input("enter node file name: ")
edgefile = input("enter edge file name: ")
dis = int(input("enter max distance: "))


#damping_values = np.array([.5, 0.52, 0.54, 0.56, 0.58, 0.6,  0.62, 0.64, 0.66, 0.68, 0.7,  0.72, 0.74, 0.76, 0.78, 0.8,  0.82, 0.84, 0.86    , 0.88, 0.9, 0.92, 0.94, 0.96, 0.98])
#for i in damping_values:
ap = AffinityPropagation(nodeData_fileName = nodefile, edgeList_fileName = edgefile, dis = dis)
ap.fit()
    #print(ap.labels_)
print("iter", ap.n_iter_)


print("labels", ap.labels_)
#print("iteration", ap.n_iter_)
#print("number of clusters", ap.number_of_clusters_)

with open(nodefile, newline='') as csvfile:
    nodeData = np.array(list(csv.reader(csvfile, delimiter=' ')), float)
with open(edgefile, newline='') as csvfile:
    edgeData = np.array(list(csv.reader(csvfile, delimiter=' ')), int)

print("xy of 714", edgeData[714])
print("n 1 714", nodeData[np.where(nodeData[:,0] == edgeData[714,0])])
print("xy of 490", edgeData[490])
print("n1 490", nodeData[np.where(nodeData[:,0] == edgeData[490,0])])
"""
#Label max distance
for g in range(ap.number_of_clusters_):
    h = np.where(ap.labels_ == g)[0]
    x = ap.sm[ap.cluster_centers_indices_[g],h]
    if g == 6:
        print("Distance for cluster 6", x)
    print("***")
    print("Cluster number: ", g)
    print("Cluster center: ", ap.cluster_centers_indices_[g])
    print("Index of max edge", h[np.argmax(x)])
    print("Max edge similarity", ap.similarity_matrix_[h[np.argmax(x)], ap.cluster_centers_indices_[g]])
    print("Max edge distance squared", ap.sm[h[np.argmax(x)], ap.cluster_centers_indices_[g]])
    print("The distance", math.sqrt(np.max(x)))
    edge_one = np.where(nodeData[:,0] == edgeData[h[np.argmax(x)],0])[0][0]
    x1 = nodeData[edge_one,1]
    y1 = nodeData[edge_one,2]
    plt.scatter(x1, y1, c = 'black', marker='*', s=100)
    plt.text(x1+17,y1+17, str(g), fontsize=12, color="black")
"""
col = {0: 'grey', 1: 'blue', 2: 'orange', 3: 'green', 4: 'hotpink', 5: 'purple', 6: 'darkkhaki', 7: 'red', 8: 'black', 9: 'rosybrown', 10: 'plum', 11: 'wheat', 12:'lightskyblue', 13: 'salmon', 14: 'olive', 15: 'lawngreen', 16: 'lightskyblue', 17: 'paleturquoise', 18: 'tan', 19: 'plum', 20: 'hotpink', 21: 'firebrick', 22: 'yellow', 23: 'powderblue', 24: 'salmon'}

# Plotting graph
for i in range(ap.labels_.shape[0]):
     one = np.where(nodeData[:,0] == edgeData[i,0])[0][0]
     two = np.where(nodeData[:,0] == edgeData[i,1])[0][0]
     x1 = nodeData[one, 1]
     y1 = nodeData[one, 2]
     x2 = nodeData[two, 1]
     y2 = nodeData[two, 2]
     plt.plot([x1, x2], [y1,y2], color=col[ap.labels_[i]])
hh = 0
# Highlighting Exemplars
for j in range(ap.cluster_centers_indices_.size):
    cc_one = np.where(nodeData[:,0] == edgeData[ap.cluster_centers_indices_[j],0])[0][0]
    x1 = nodeData[cc_one, 1]
    y1 = nodeData[cc_one, 2]
    hh+=1
    plt.scatter(x1, y1, c = col[j], marker='*', s=200)
    plt.text(x1+25,y1+25, str(hh), fontsize=14, color=col[j])

print(ap.sm)
print(ap.similarity_matrix_)
print("MAX VALUE: ", np.max(ap.sm))

plt.xlabel("X axis (m)")
plt.ylabel("Y axis (m)")
#plt.xticks(np.arange(0, 2000, 200))
axs = plt.gca()
axs.set_aspect('equal', 'box')
axs.axis('equal')
plt.show()

