from random import sample
from functions import functions, np

class TimeSeriesKmeans():
    
    def __init__ (self, X, y, metric = 'euclidean', method = 'interpolate'):
        self.X = X # 2 dimensional data - can be ndarray iff instances are of the same length
        self.keys = y # list of input labels ordered with self.X
        self.metric = metric # 'euclidean', 'dtw', 'fdtw'
        self.method = method # 'interpolate', 'pad'
        self.clusters = [] # ordered with self.X
        self.centroids = {} # keys by cluster 0 - k_clusters
        
    # assigns each instance to its geometrically closest centroid

    def assign_clusters(self):
        clusters = [] 
        for i in range(len(self.X)): 
            distances = {}
            for k in self.centroids:
                distances[functions[self.metric](self.X[i], self.centroids[k])] = k
            clusters.append(distances[min(distances)])
    
    # sets the centroids to the mean length and the mean value of each index, in each cluster

    def update_centroids(self): # 'pad'
        for k in range(len(self.centroids)):  
            cluster = []
            lengths = []
            for i in range(len(self.X)):
                if self.clusters[i] == k:
                    cluster.append(self.X[i])
                    lengths.append(len(self.X[i]))
            length = functions['avg'](lengths) if self.method == 'interpolate' else max(lengths)
            cluster = np.array([functions[self.method](arr, length) for arr in cluster])
            self.centroids[self.keys[i].append(k)] = np.mean(cluster, axis = 0)
    
    # solves the local cluster problem with randomly assigned centroids

    def local_kmeans(self, k_clusters, iter = 10):
        if len(self.centroids) == 0:
            self.centroids = sample(self.X, k_clusters)  
        self.assign_clusters()       
        for i in range(iter):           
            self.assign_clusters()
            self.update_centroids()
        
    # solves the k_clusters = 1 problem, and iterativeley samples & solves locally for  additional centroids up to k_clusters = k

    def global_kmeans(self, k_clusters, n_sample = 10, iter = 10):
        global_costs = {}
        for k in range(k_clusters):
            costs = {}
            for n in range(n_sample):
                self.centroids = self.centroids + sample(self.X, 1)
                for m in range(iter):             
                    self.assign_clusters()
                    self.update_centroids()    
                    cost = 0
                    for i in range(len(self.X)):
                        cost += functions[self.metric](self.X[i], self.centroids[self.clusters[i]])
                    costs[cost] = [self.centroids, self.clusters]
            global_costs[min(costs)] = costs[min(costs)]
        self.centroids[self.keys[i]] = global_costs[min(global_costs)][0]
        self.clusters = global_costs[min(global_costs)][1]   