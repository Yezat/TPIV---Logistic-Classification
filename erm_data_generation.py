import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

def compute_solution(ys, xs, Sigmadelta, eps, reg_param):
    n, d = xs.shape
    xs_norm = xs / np.sqrt(d)
    w = cp.Variable(d)


    objective = cp.Minimize(
        cp.logistic(cp.multiply(-ys, xs_norm @ w)) + reg_param * cp.norm(w,2)**2
    )

    return w.value



d = 200
tau = 0.05
Sigmax = np.eye(d,d)
Sigmatheta = np.eye(d,d)
Sigmadelta = np.eye(d,d)

reg_param = 0.1
eps = 0.1

alphas = np.logspace(np.log10(0.2), 1, 10)

for i, alpha in enumerate(alphas):
    n = int(alpha * d)

    xs = np.random.multivariate_normal(np.zeros(d), Sigmax, shape=(d,1))
    wstar = np.random.multivariate_normal(np.zeros(d), Sigmatheta, shape=(d,1))

    ys = np.sign(xs @ wstar / np.sqrt(d) + tau * np.random.randn(n,1))

    w = compute_solution(ys, xs, Sigmadelta, eps, reg_param)