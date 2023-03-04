from random import sample
from functions import functions, np
from tslearn.barycenters import dtw_barycenter_averaging

class TimeSeriesKmeans():
    
    def __init__ (self, k_clusters, max_iter = 100, centroids = [],metric = 'dtw', method = 'barycenter'):
        self.k_clusters = k_clusters
        self.max_iter = max_iter
        self.centroids = centroids
        self.metric = metric # 'euclidean', 'dtw', 'fdtw'
        self.method = method

    def _interpolated_average_shapelet(self, X) -> np.array:
        shapelet = []
        mean_length = functions['int_avg']([len(arr) for arr in X])
        interpolated_candidates = [functions['interpolate'](arr, mean_length) for arr in X]
        for i in range(mean_length):
            ith = []
            for n in range(len(X)):
                ith.append(interpolated_candidates[n][i])
            shapelet.append(functions['avg'](ith))
        return np.array(shapelet)
    
    def _barycenter_average_shapelet(self, X, barycenter_size = None, init_barycenter = None, max_iter = 30, tol=1e-05, weights=None, metric_params=None, verbose=False,n_init=1) -> np.array:
        return dtw_barycenter_averaging(X, barycenter_size=barycenter_size, init_barycenter=init_barycenter, max_iter=max_iter, tol=tol, weights=weights, metric_params=metric_params, verbose=verbose,n_init=n_init)
        
    # assigns each instance to its geometrically closest centroid
    def _assign_clusters(self, X) -> list:
        return [np.argmin(np.array([functions[self.metric](instance, centroid) for centroid in self.centroids])) for instance in X]
    
    # randomly initializes k centroids to an instance of x
    def _initialize_centroids(self, X, k_centroids) -> list:
        return sample(X, k_centroids)

    # checks to make sure there are no duplicate centroids
    def _check_centroid_assignment(self) -> bool:
        if len(self.centroids) == 0:
            return False
        for k1 in range(len(self.centroids)):
            for k2 in range(len(self.centroids)):
                if k1 != k2 and np.array_equal(self.centroids[k1], self.centroids[k2]):
                    return False
        return True 

    # sets the centroids to the mean length and the mean value of each index, in each cluster
    def _update_centroids(self, X) -> list:
        new_centroids = []
        for k in range(len(self.centroids)):  
            cluster = []
            for i in range(len(X)):
                if self.clusters[i] == k:
                    cluster.append(X[i])
            if self.method == 'barycenter':
               new_centroids.append(self._barycenter_average_shapelet(cluster))
            if self.method == 'interpolate':
                new_centroids.append(self._interpolated_average_shapelet(cluster))
        return new_centroids

    # returns True if each centroid has not changed upon centroid update
    def _check_solution(self, new_centroids) -> bool:
        return all([np.array_equal(self.centroids[i], new_centroids[i]) for i in range(len(self.centroids))])
    

    def _get_inertia(self, X) -> float:
        return sum([functions[self.metric](X[i], self.centroids[self.clusters[i]]) for i in range(len(X))])
            
    # solves the local cluster problem with randomly assigned centroids
    def local_kmeans(self, X):
        if len(self.centroids) < self.k_clusters:
            self.centroids = self.centroids + self._initialize_centroids(X, self.k_clusters - len(self.centroids))
        for i in range(self.max_iter):
            self.clusters = self._assign_clusters(X)
            new_centroids = self._update_centroids(X)
            if self._check_solution(new_centroids):
                break
            else:
                self.centroids = new_centroids
        self.inertia = self._get_inertia(X)

    # solves the local cluster problem n_init times and saves the result with the lowest inertia
    def sample_kmeans(self, X, n_init = 5):
        costs = {}
        for n in range(n_init):
            self.local_kmeans(X)
            costs[self.inertia]  = self.clusters, self.centroids
        self.inertia = min(costs)
        self.clusters = costs[self.inertia][0]
        self.centroids = costs[self.inertia][1]

    # solves the k_clusters = 1 problem, and iterativeley samples & solves locally for  additional centroids up to k = k_clusters
    def global_kmeans(self, X, n_init = 5):
        global_costs = {}
        for k in range(1, self.k_clusters+1):
            costs = {}
            for n in range(n_init):
                self.centroids = global_costs[min(global_costs)][1] + self._initialize_centroids(X, 1) if k > 1 else self._initialize_centroids(X, k)
                for i in range(self.max_iter):
                    self.clusters = self._assign_clusters(X)
                    new_centroids = self._update_centroids(X)
                    if self._check_solution(new_centroids):
                        break
                    else:
                        self.centroids = new_centroids
                inertia = self._get_inertia(X)
                costs[inertia]  = self.clusters, self.centroids
            global_costs[min(costs)] = costs[min(costs)][0], costs[min(costs)][1]
        self.inertia = min(global_costs)
        self.clusters = global_costs[self.inertia][0]
        self.centroids = global_costs[self.inertia][1]

