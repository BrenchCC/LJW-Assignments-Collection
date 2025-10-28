import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score


class KMeans:
    """K-means clustering implementation

    Attributes:
        K (int): Number of clusters
        max_iters (int): Maximum iterations
        tol (float): Convergence tolerance
        centroids (np.array): Cluster centroids
        labels (np.array): Cluster assignments
    """

    def __init__(self, K=3, max_iters=100, tol=1e-4):
        """Initialize K-means parameters

        Args:
            K (int): Number of clusters
            max_iters (int): Maximum iterations
            tol (float): Convergence tolerance
        """
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def initialize_centroids(self, X):
        """Initialize centroids using k-means++ strategy

        Args:
            X (np.array): Input data (n_samples, n_features)

        Returns:
            np.array: Initial centroids (K, n_features)
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.K, n_features))

        # First centroid: random point
        idx = np.random.choice(n_samples)
        centroids[0] = X[idx]

        # Subsequent centroids: proportional to squared distance
        for k in range(1, self.K):
            dists = np.array([np.min([np.linalg.norm(x - c)**2 for c in centroids[:k]])
                             for x in X])
            probs = dists / dists.sum()
            idx = np.random.choice(n_samples, p=probs)
            centroids[k] = X[idx]

        return centroids

    def fit(self, X):
        """Fit K-means model to data

        Args:
            X (np.array): Input data (n_samples, n_features)
        """
        n_samples = X.shape[0]

        # Initialize centroids
        self.centroids = self.initialize_centroids(X)
        prev_centroids = self.centroids.copy()
        self.labels = np.zeros(n_samples, dtype=int)

        # Iteration loop
        for it in range(self.max_iters):
            # Assignment step: assign each point to nearest centroid
            dists = np.zeros((n_samples, self.K))
            for k in range(self.K):
                dists[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
            self.labels = np.argmin(dists, axis=1)

            # Update step: recompute centroids
            for k in range(self.K):
                cluster_points = X[self.labels == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = cluster_points.mean(axis=0)

            # Check convergence
            centroid_shift = np.linalg.norm(self.centroids - prev_centroids)
            if centroid_shift < self.tol:
                break

            prev_centroids = self.centroids.copy()

    def predict(self, X):
        """Predict cluster labels for new data

        Args:
            X (np.array): Input data (n_samples, n_features)

        Returns:
            np.array: Cluster labels
        """
        dists = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            dists[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
        return np.argmin(dists, axis=1)

class EMGMM:
    """Gaussian Mixture Model with EM algorithm

    Attributes:
        K (int): Number of clusters
        max_iters (int): Maximum iterations
        tol (float): Convergence tolerance
        reg_covar (float): Regularization for covariance
        weights (np.array): Mixture weights
        means (np.array): Cluster means
        covariances (np.array): Cluster covariances
        responsibilities (np.array): Cluster responsibilities
    """

    def __init__(self, K=3, max_iters=200, tol=1e-4, reg_covar=1e-6):
        """Initialize GMM parameters

        Args:
            K (int): Number of clusters
            max_iters (int): Maximum iterations
            tol (float): Convergence tolerance
            reg_covar (float): Covariance regularization
        """
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.weights = None
        self.means = None
        self.covariances = None
        self.responsibilities = None

    def initialize_parameters(self, X):
        """Initialize GMM parameters using K-means

        Args:
            X (np.array): Input data (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Use K-means for initialization
        kmeans = KMeans(K=self.K)
        kmeans.fit(X)
        labels = kmeans.labels

        self.weights = np.zeros(self.K)
        self.means = np.zeros((self.K, n_features))
        self.covariances = np.zeros((self.K, n_features, n_features))

        for k in range(self.K):
            cluster_points = X[labels == k]
            self.weights[k] = len(cluster_points) / n_samples
            self.means[k] = cluster_points.mean(axis=0)
            self.covariances[k] = np.cov(cluster_points.T) + self.reg_covar * np.eye(n_features)

    def _compute_log_prob(self, X):
        """Compute log probabilities for each component

        Args:
            X (np.array): Input data (n_samples, n_features)

        Returns:
            np.array: Log probabilities (n_samples, K)
        """
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, self.K))

        for k in range(self.K):
            log_probs[:, k] = multivariate_normal.logpdf(
                X, mean=self.means[k], cov=self.covariances[k])

        return log_probs + np.log(self.weights)

    def _e_step(self, X):
        """E-step: compute responsibilities with stable logsumexp

        Args:
            X (np.array): Input data (n_samples, n_features)

        Returns:
            tuple: (log_likelihood, responsibilities)
        """
        log_probs = self._compute_log_prob(X)  # shape: (n_samples, K)

        # Stable computation of log_sum_exp (per sample)
        max_log = np.max(log_probs, axis=1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(log_probs - max_log), axis=1, keepdims=True)) + max_log

        # Compute responsibilities
        log_resp = log_probs - log_sum_exp
        resp = np.exp(log_resp)

        # Total log likelihood is sum of log_sum_exp
        total_log_likelihood = np.sum(log_sum_exp)

        return total_log_likelihood, resp

    def _m_step(self, X, resp):
        """M-step: update parameters

        Args:
            X (np.array): Input data
            resp (np.array): Responsibilities
        """
        n_samples, n_features = X.shape

        # Effective number of points per component
        Nk = resp.sum(axis=0)

        # Update weights
        self.weights = Nk / n_samples

        # Update means
        self.means = resp.T @ X / Nk[:, np.newaxis]

        # Update covariances
        for k in range(self.K):
            diff = X - self.means[k]
            self.covariances[k] = (resp[:, k] * diff.T) @ diff / Nk[k]
            self.covariances[k] += self.reg_covar * np.eye(n_features)

    def fit(self, X):
        """Fit GMM model to data

        Args:
            X (np.array): Input data (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.initialize_parameters(X)

        prev_log_likelihood = -np.inf
        self.responsibilities = np.zeros((n_samples, self.K))

        # EM iterations
        for it in range(self.max_iters):
            # E-step
            log_likelihood, self.responsibilities = self._e_step(X)

            # Check convergence
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

            prev_log_likelihood = log_likelihood

            # M-step
            self._m_step(X, self.responsibilities)

    def predict(self, X):
        """Predict cluster labels

        Args:
            X (np.array): Input data

        Returns:
            np.array: Cluster labels
        """
        log_probs = self._compute_log_prob(X)
        return np.argmax(log_probs, axis=1)

class MeanShift:
    """Mean Shift clustering implementation

    Attributes:
        bandwidth (float): Kernel bandwidth
        max_iters (int): Maximum iterations
        tol (float): Convergence tolerance
        cluster_centers (np.array): Cluster centers
        labels (np.array): Cluster labels
    """

    def __init__(self, bandwidth=1.0, max_iters=200, tol=1e-4):
        """Initialize Mean Shift parameters

        Args:
            bandwidth (float): Kernel bandwidth
            max_iters (int): Maximum iterations
            tol (float): Convergence tolerance
        """
        self.bandwidth = bandwidth
        self.max_iters = max_iters
        self.tol = tol
        self.cluster_centers = None
        self.labels = None

    def _gaussian_kernel(self, x, center):
        """Compute Gaussian kernel density

        Args:
            x (np.array): Data point
            center (np.array): Kernel center

        Returns:
            float: Kernel density
        """
        dist = np.linalg.norm(x - center)
        return np.exp(-0.5 * (dist**2) / (self.bandwidth**2))

    def _shift_point(self, x, X):
        """Perform mean shift for a single point

        Args:
            x (np.array): Initial point
            X (np.array): All data points

        Returns:
            np.array: Shifted point
        """
        shifted = np.zeros_like(x)
        total_weight = 0.0

        for point in X:
            weight = self._gaussian_kernel(point, x)
            shifted += weight * point
            total_weight += weight

        return shifted / total_weight if total_weight > 0 else x

    def fit(self, X):
        """Fit Mean Shift model to data

        Args:
            X (np.array): Input data (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        shifted_points = np.zeros_like(X)

        # Perform mean shift for each point
        for i in range(n_samples):
            point = X[i].copy()
            for it in range(self.max_iters):
                new_point = self._shift_point(point, X)
                if np.linalg.norm(new_point - point) < self.tol:
                    break
                point = new_point
            shifted_points[i] = point

        # Cluster shifted points
        self.cluster_centers = []
        self.labels = np.zeros(n_samples, dtype=int) - 1
        cluster_id = 0

        for i in range(n_samples):
            if self.labels[i] != -1:
                continue

            # New cluster
            self.cluster_centers.append(shifted_points[i])
            self.labels[i] = cluster_id

            # Find all points in this cluster
            for j in range(i+1, n_samples):
                if np.linalg.norm(shifted_points[i] - shifted_points[j]) < self.bandwidth/2:
                    self.labels[j] = cluster_id

            cluster_id += 1

        self.cluster_centers = np.array(self.cluster_centers)

    def predict(self, X):
        """Predict cluster labels for new data

        Args:
            X (np.array): Input data

        Returns:
            np.array: Cluster labels
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            min_dist = np.inf
            for j, center in enumerate(self.cluster_centers):
                dist = np.linalg.norm(X[i] - center)
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j

        return labels