# import necessary packages
import os
import random as rd
import numpy as np

def polynomial_features(x, degree):
    """
    Convert input `x` into presentation-style features
    Phi: Feature Matrix (degree+1, n)
    """
    n = len(x)
    Phi = np.zeros((degree+1, n))
    for d in range(degree+1):
        Phi[d, :] = x ** d
    return Phi

# Least-Squares(LS)
def least_squares(Phi, y):
    """
    Least-Squares(LS)

    func parameter:
    Phi: feature matrix (D, n) (dim, num of samples)
    y: target value (n,)

    theta: parameter estimation(D,)
    """
    # formula: theta = (Phi * Phi^T)^(-1) * Phi * y
    n = Phi.shape[1]
    Phi_matrix = Phi  # D x n

    # calculation (Phi * Phi^T)
    A = np.dot(Phi_matrix, Phi_matrix.T)

    # calculation (Phi * y)
    b = np.dot(Phi_matrix, y)

    # solve theta = A^(-1) * b
    theta = np.linalg.solve(A, b)

    return theta

def regularized_least_squares(Phi, y, lambda_val):
    """
    Regularized_Least_Squares(RLS)

    func parameter:
    Phi: feature matrix (D, n)
    y: target value (n,)
    lambda_val: regularization parameter

    return:
    theta: parameter estimation (D,)
    """
    D, n = Phi.shape

    # formula: theta = (Phi * Phi^T + lambda * I)^(-1) * Phi * y
    A = np.dot(Phi, Phi.T) + lambda_val * np.eye(D)
    b = np.dot(Phi, y)

    theta = np.linalg.solve(A, b)

    return theta

def lasso_regression(Phi, y, lambda_val, max_iter = 1000, tol = 1e-4):
    """
    LASSO Regression(using coordinate descent)

    func parameter:
    Phi: feature matrix (D, n)
    y: target value (n,)
    lambda_val:  regularization parameter
    max_iter: maximum number of iterations
    tol: Convergence limit

    return:
    theta: parameter estimation (D,)
    """
    D, n = Phi.shape
    theta = np.zeros(D)  # initialize paramter

    # pre calculation
    Phi_squared = np.sum(Phi**2, axis=1)

    for iteration in range(max_iter):
        theta_old = theta.copy()

        # coordinate descent: update one parameter of one iteration
        for j in range(D):
            # calculate residuals
            r = y - np.dot(Phi.T, theta) + Phi[j, :] * theta[j]

            # update parameter
            if Phi_squared[j] > 0:
                theta_j = np.dot(Phi[j, :], r) / Phi_squared[j]

                # using soft threshold
                if theta_j > lambda_val / Phi_squared[j]:
                    theta[j] = theta_j - lambda_val / Phi_squared[j]
                elif theta_j < -lambda_val / Phi_squared[j]:
                    theta[j] = theta_j + lambda_val / Phi_squared[j]
                else:
                    theta[j] = 0

        # check the convergence
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return theta
def robust_regression(Phi, y, max_iter=100, tol=1e-4):
    """
    Robust regression (using iteratively reweighted least squares)

    func parameters:
    Phi: feature matrix (D, n)
    y: target value (n,)
    max_iter: Maximum number of iterations
    tol: Convergence tolerance

    Returns:
    theta: parameter estimation (D,)
    """

    # initialize variable
    D, n = Phi.shape
    theta = np.zeros(D)
    weights = np.ones(n)

    for iteration in range(max_iter):
        theta_old = theta.copy()

        #  reweighted least squares
        # W = diag(weights)
        # theta = (Phi * W * Phi^T)^(-1) * Phi * W * y
        W_sqrt = np.sqrt(weights)
        Phi_weighted = Phi * W_sqrt
        y_weighted = y * W_sqrt

        A = np.dot(Phi_weighted, Phi_weighted.T)
        b = np.dot(Phi_weighted, y_weighted)

        theta = np.linalg.solve(A, b)

        # update weight by residual
        residuals = np.abs(y - np.dot(Phi.T, theta))
        weights = 1 / np.maximum(residuals, 1e-10)  # avoid division by zero

        # check convergence
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return theta

def bayesian_regression(Phi, y, alpha=1.0, sigma2=5.0):
    """
    Byesian Regression

    fun parameters:
    Phi: feature matrix (D, n)
    y: target value (n,)
    alpha: prior accuracy
    sigma2: noise variance

    return:
    theta_mean: posterior mean (D,)
    theta_cov: posterior covariance (D, D)
    """
    D, n = Phi.shape

    # Calculate the posterior distribution parameters
    # Sigma_theta = (alpha * I + (1/sigma2) * Phi * Phi^T)^(-1)
    # mu_theta = (1/sigma2) * Sigma_theta * Phi * y

    A = alpha * np.eye(D) + (1/sigma2) * np.dot(Phi, Phi.T)
    Sigma_theta = np.linalg.inv(A)
    mu_theta = (1/sigma2) * np.dot(Sigma_theta, np.dot(Phi, y))

    return mu_theta, Sigma_theta

# Prediction function
def predict(Phi, theta):
    """
    Make predictions using the learned parameters

    Parameters:
    Phi: feature matrix (D, n)
    theta: parameter vector (D,)

    Returns:
    predictions: predicted values (n,)
    """
    return np.dot(Phi.T, theta)

if __name__ == "__main__":
    # Test feature transformation
    print("Test feature transformation")
    x_test = np.array([1, 2, 3])
    degree = 3
    Phi_test = polynomial_features(x_test, degree)
    print("input x:", x_test)
    print("Polynomial characteristic matrix (3 degree):")
    print(Phi_test)
    print("-"*50)

    # simple test of LS
    print("Least squares simple test")
    D = rd.randint(1, 5)
    n = rd.randint(D, 10)
    print(f"D: {D}, n: {n}")

    Phi = np.random.randn(D, n)
    print(f"random feature matrix:\n{Phi}")

    y = np.random.randn(n)
    print(f"random target value: {y}")

    theta_hat = least_squares(Phi, y)
    print(f"theta_hat: {theta_hat}")
    print("-"*50)

    # simple test of RLS
    print("regularized least squares test")
    lambda_val = rd.uniform(0.01, 0.7)
    print(f"regularization parameter: {lambda_val}")
    theta_hat = regularized_least_squares(Phi, y, lambda_val)
    print(f"theta_hat: {theta_hat}")
    print("-"*50)

    # simple test of lasso
    print("lasso regression test")
    theta_hat = lasso_regression(Phi, y, lambda_val)
    print(f"theta_hat: {theta_hat}")

    # simple test of robust regression
    print("robust regression test")
    theta_hat = robust_regression(Phi, y)
    print(f"theta_hat: {theta_hat}")
    print("-"*50)

    # simple test of bayesian regression
    print("bayesian regression test")
    theta_hat = bayesian_regression(Phi, y)
    print(f"theta_hat: {theta_hat}")
    print("-"*50)

