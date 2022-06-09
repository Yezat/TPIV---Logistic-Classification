""" 
This module does Logistic Regression using sklearn
"""

from sklearn.linear_model import LogisticRegression
from data import *


# plot imports
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def get_logistic_regressor(X, y, lam):
    """
    Return a Logistic Regression object from sklearn.linear_model.LogisticRegression

    parameters:
    X: training data (n,d)
    y: training labels (n,)
    lam: regularization parameter
    """
    max_iter = 1000
    tol      = 1e-16
    if lam == 0:
        clf =  LogisticRegression(penalty='none',solver='lbfgs',fit_intercept=False, max_iter=max_iter, tol=tol, verbose=0).fit(X,y)
    else:
        clf =  LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, max_iter=max_iter, tol=tol, verbose=0,C=1./lam).fit(X,y)
    return clf

def predict_on_logistic_regressor(clf, X):
    """
    Uses a Logistic Regression object from sklearn.linear_model.LogisticRegression to predict labels for test data X

    parameters:
    clf: LogisticRegression object from sklearn.linear_model.LogisticRegression
    X: test data (n,d)
    """
    return clf.predict(X)

def predict_proba_on_logistic_regressor(clf, X):
    """
    Uses a Logistic Regression object from sklearn.linear_model.LogisticRegression to predict probabilities for test data X

    parameters:
    clf: LogisticRegression object from sklearn.linear_model.LogisticRegression
    X: test data (n,d)
    """
    prob = clf.predict_proba(X)
    return prob[:,1]


if __name__ == "__main__":
    d = 300
    n = 3000
    n_test = 100000
    w = sample_weights(d)
    p = 0.75
    dp = 0.1
    tau = 2
    Xtrain, y = sample_training_data(w,d,n,tau)
    Xtest = sample_test_data(d,n_test)  
    clf = get_logistic_regressor(Xtrain,y)

    f_erm = predict_proba_on_logistic_regressor(clf,Xtest)
    
    fig,ax = plt.subplots()
    plt_erm, = ax.plot(f_erm, label="$f_{erm}$")
    ax.legend(handles=[plt_erm])
    plt.title("$f_{erm}$")
    plt.xlabel("Samples")
    plt.ylabel("$f_{erm}$")
    plt.show()

