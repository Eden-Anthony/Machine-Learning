from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
import numpy as np


class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE

    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """

    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        # Get unique classes
        self.classes_ = np.sort(np.unique(y))
        # Split X by class
        training_sets = [X[y == yi] for yi in self.classes_]
        # Create a KDE for each class
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        # Calculate the prior probabilites of each class
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self

    def predict_proba(self, X):
        # Calculate p(x|class)
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        # Calculate p(class|x)*p(x) , the unnormalised version of p(class|x)
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        # Return the index of the most dominant class
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
