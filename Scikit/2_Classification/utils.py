import numpy as np
from numpy.linalg import norm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def _gen_linear_data(n_samples, noise_level):
    fst_half = n_samples // 2
    snd_half = n_samples - fst_half
    Y = np.ones((n_samples,))
    Y[:fst_half] = -1
    X1 = np.random.normal([5, 5], scale=[1 * noise_level], size=(fst_half, 2))
    X2 = np.random.normal([8, 5], scale=[1 * noise_level], size=(snd_half, 2))
    return np.concatenate((X1, X2), 0), Y


def _gen_moons(n_samples, noise_level):
    X, Y = datasets.make_moons(n_samples=n_samples, shuffle=True, noise=noise_level)
    Y[Y == 0] = -1
    return X, Y


def _gen_circles(n_samples, noise_level):
    X, Y = datasets.make_circles(n_samples=n_samples, shuffle=True, noise=noise_level)
    Y[Y == 0] = -1
    return X, Y


def create_random_data(n_samples, noise_level, dataset="linear", seed=0):
    """Generates a random dataset. Can generate 'linear', 'moons' or 'circles'.
    Parameters
    ----------
    n_samples
        The total number of samples. These will be equally divided between positive
        and negative samples.
    noise_level
        The amount of noise: higher noise -> harder problem. The meaning of the noise
        is different for each dataset.
    dataset
        A string to specify the desired dataset. Can be 'linear', 'moons', 'circles'.
    seed
        Random seed for reproducibility.
    Returns
    -------
    X
        A 2D array of features
    Y
        A vector of targets (-1 or 1)
    """
    np.random.seed(seed)

    if dataset.lower() == "linear":
        return _gen_linear_data(n_samples, noise_level)
    elif dataset.lower() == "moons":
        return _gen_moons(n_samples, noise_level)
    elif dataset.lower() == "circles":
        return _gen_circles(n_samples, noise_level)
    else:
        raise ValueError(("Dataset '%s' is not valid. Valid datasets are:"
                          " 'linear', 'moons', 'circles'") % dataset)


def data_split(X, Y, n_train):
    assert n_train < X.shape[0]
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx[:n_train], :], X[idx[n_train:], :], Y[idx[:n_train]], Y[idx[n_train:]]


def plot_dataset(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], alpha=0.5)
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], alpha=0.5)


def plot_separation(X, Y, model):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z[Z < 0] = -1
    Z[Z >= 0] = 1

    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(X[Y == -1][:, 0], X[Y == -1][:, 1])
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1])


def create_sparse_random_data(n_samples, n_features, n_informative, noise_level, seed=0):
    """Generate a random dataset for regression, according to:
    y_i = w^\top x_i + epsilon_i
    where w has `s` non zero entries, x_i's are i.i.d. Gaussian
    with identity as covariance, and epsilon_i are i.i.d standard
    Gaussian, with variance controlled by noise_level.

    Parameters
    ----------
    n_samples: int
        The number of samples.
    n_features: int
        The number of features.
    n_informative: int
        Inverse of sparsity, i.e. the number of non-zero entries of w.
    noise_level: float
        Importance of noise. ||epsilon|| / ||X @ w|| = noise_level
    seed: int, optional (default=0)
        Seed for pseudo-random number generation.

    Returns
    -------
    X: np.array, shape (n_samples, n_features)
        Design matrix.
    y: np.array, shape (n_samples,)
        Observation vector.
    """
    if n_informative > n_features:
        raise ValueError("Sparsity s cannot be larger than "
                         "n_features, got %s > %s" % (n_informative, n_features))
    np.random.seed(seed)  # seed to always get same data
    X = np.random.randn(n_samples, n_features)
    w = np.zeros(n_features)
    support = np.random.choice(n_features, n_informative, replace=False)
    w[support] = np.random.randn(n_informative)
    epsilon = np.random.randn(n_samples)

    y = X @ w / norm(X @ w) + noise_level * epsilon / norm(epsilon)
    return X, y


def train_test_data(n_samples, n_features, n_informative_features,
                    noise_level):
    """Util function to generate and split random data.
    See the docstring of create_random_data for more details.
    """
    X, y = create_sparse_random_data(n_samples, n_features, n_informative_features, noise_level=noise_level)
    train_size = 0.8  # proportion of dataset used for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=train_size)
    return X_train, X_test, y_train, y_test
