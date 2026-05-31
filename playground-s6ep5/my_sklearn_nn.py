import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, _fit_context
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class PitNN(nn.Module):
    
    def __init__(self, in_features, width, depth):
        super(PitNN, self).__init__()
        self.l1 = nn.Linear(in_features, width)
        self.lr = nn.Linear(width, width)
        #self.l2 = nn.Linear(width, in_features)
        self.relu = nn.ReLU()
        self.lf = nn.Linear(width, 1)
        self.sigmoid = nn.Sigmoid()
        self.depth = depth
    def forward(self, x):
        x = self.relu(self.l1(x))
        # x = self.relu(self.l1(x))
        for i in range(self.depth):
            x = self.relu(self.lr(x))
        #x = self.relu(self.l2(x))
        # x = self.relu(self.l1(x))
        # x = self.relu(self.l1(x))
        # x = self.relu(self.l1(x))
        # x = self.relu(self.l1(x))
        x = self.sigmoid(self.lf(x))
        return x

# Note that the mixin class should always be on the left of `BaseEstimator` to ensure
# the MRO works as expected.
class MyNNClassifier(ClassifierMixin, BaseEstimator):
    """An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from skltemplate import TemplateClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = TemplateClassifier().fit(X, y)
    >>> clf.predict(X)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(self, in_features, epochs=20, batch_size=256, learning_rate=0.001, weight_decay=0.0001,
                 width=None, depth=1, device='cpu'):
        self.in_features = in_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.width = width if width is not None else in_features
        self.depth = depth
        self.device = device
        self.model = PitNN(in_features, self.width, self.depth).to(device)


    def _train(self, X, y):
        """A reference implementation of a training function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        loss_fn = nn.BCELoss()  
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        torch.manual_seed(1)
        y = y.reshape(-1, 1)
        train_ds = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            for x_batch, y_batch in train_dl:
                pred = self.model(x_batch.to(self.device))
                loss = loss_fn(pred, y_batch.to(self.device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        return self 
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # `_validate_data` is defined in the `BaseEstimator` class.
        # It allows to:
        # - run different checks on the input data;
        # - define some attributes associated to the input data: `n_features_in_` and
        #   `feature_names_in_`.
        # X, y = self._validate_data(X, y)
        # We need to make sure that we have a classification task
        check_classification_targets(y)

        # classifier should always store the classes seen during `fit`
        self.classes_ = np.unique(y)

        # Store the training data to predict later
        self.X_ = X
        self.y_ = y

        self._train(X, y)

        # Return the classifier
        return self

    def _predict(self, X, proba=False):
        self.model.eval()
        self.model.to('cpu')
        with torch.no_grad():
            y_pred = self.model(torch.tensor(X).float().to('cpu'))
        if proba:
            return np.hstack((1-y_pred.numpy(), y_pred.numpy()))
        else:
            return y_pred.round().numpy().flatten()

    def predict(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        # X = self._validate_data(X, reset=False)
        
        y_pred = self._predict(X)
        return y_pred
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        #return self.y_[closest]

    def predict_proba(self, X):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        # X = self._validate_data(X, reset=False)
        
        y_pred = self._predict(X, proba=True)
        return y_pred
        #closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        #return self.y_[closest]
