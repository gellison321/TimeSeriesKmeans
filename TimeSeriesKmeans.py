from random import sample
from functions import functions, np
from tslearn.barycenters import dtw_barycenter_averaging

class TimeSeriesKmeans():
    
    def __init__ (self, X, y = [], metric = 'dtw', method = 'barycenter'):
        self.X = X # 2 dimensional data - can be ndarray iff instances are of the same length
        self.keys = y # list of input labels ordered with self.X
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
    def _assign_clusters(self, centroids) -> list:
        return [np.argmin(np.array([functions[self.metric](instance, centroid) for centroid in centroids])) for instance in self.X]
    
    # randomly initializes k centroids to an instance of x
    def _initialize_centroids(self, k_centroids) -> list:
        return sample(self.X, k_centroids)

    # checks to make sure there are no duplicate centroids
    def _check_centroid_assignment(self, centroids) -> bool:
        if len(centroids) == 0:
            return False
        for k1 in range(len(centroids)):
            for k2 in range(len(centroids)):
                if k1 != k2 and np.array_equal(centroids[k1], centroids[k2]):
                    return False
        return True 

    # sets the centroids to the mean length and the mean value of each index, in each cluster
    def _update_centroids(self, centroids, clusters) -> list:
        new_centroids = []
        for k in range(len(centroids)):  
            cluster = []
            for i in range(len(self.X)):
                if clusters[i] == k:
                    cluster.append(self.X[i])
            if self.method == 'barycenter':
               new_centroids.append(self._barycenter_average_shapelet(cluster))
            if self.method == 'interpolate':
                new_centroids.append(self._interpolated_average_shapelet(cluster))
        return new_centroids

    # returns True if each centroid has not changed upon centroid update
    def _check_solution(self, centroids, new_centroids) -> bool:
        return all([np.array_equal(centroids[i], new_centroids[i]) for i in range(len(centroids))])
    

    def _get_intertia(self, centroids, clusters) -> float:
        pass
            
    # solves the local cluster problem with randomly assigned centroids
    def local_kmeans(self, k_clusters, centroids = [], max_iter = 100):
        if len(centroids) < k_clusters:
            centroids = centroids + self._initialize_centroids(k_clusters - len(centroids))
        for i in range(max_iter):
            clusters = self._assign_clusters(centroids)
            new_centroids = self._update_centroids(centroids, clusters)
            if self._check_solution(centroids, new_centroids):
                break
            else:
                centroids = new_centroids
        return clusters, centroids

    # solves the local cluster problem with randomly assigned centroids
    def sample_kmeans(self, k_clusters, n_init = 5,  centroids = [], max_iter = 100):
        pass

    # solves the k_clusters = 1 problem, and iterativeley samples & solves locally for  additional centroids up to k = k_clusters
    def global_kmeans(self, k_clusters, n_sample = 4, iter = 10):
        pass