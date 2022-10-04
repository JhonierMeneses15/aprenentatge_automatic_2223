import numpy as np
from sklearn import preprocessing as sc
class Adaline:
    """ADAptive LInear NEuron classifier.
       Gradient Descent

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Error in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def normalize(self, X, y=None):
        return sc.RobustScaler().fit_transform(X , y)


    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        errors = 0
        for _ in range(self.n_iter):
            errs = (y - self.predict(X))
            self.w_[1:] += self.eta * X.T.dot(errs)  # actualitzacio dels pesos
            self.w_[0] += self.eta * errs.sum()  # actualitzacio del bias
            # Extra: calculam els errors de classificacio a cada iteració
            cost = (errs ** 2).sum() / 2.0
            self.cost_.append(cost)





    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
