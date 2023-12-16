import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

# fit a Gaussian Mixture Model with two components
n_components = 20
# n_components = list(range(2, 11))+[20, 30, 40, 50, 60, 70, 80, 90, 100]

# Read data
data1 = pd.read_csv("../dataset/toydata/gaussian_sample1.data")
data1 = data1.values
data2 = pd.read_csv("../dataset/toydata/gaussian_sample2.data")
data2 = data2.values

clf = GaussianMixture(n_components=n_components, covariance_type="full")
clf.fit(data1)
means = clf.means_
covariances = clf.covariances_
colors = ["navy"]
def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(
            gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

fig, ax = plt.subplots()
make_ellipses(clf, ax)

# display predicted scores by the model as a contour plot
# x = np.linspace(-50.0, 80.0)
# y = np.linspace(-50.0, 80.0)
# X, Y = np.meshgrid(x, y)
# XX = np.array([X.ravel(), Y.ravel()]).T
# Z = -clf.score_samples(XX)
# Z = Z.reshape(X.shape)

# CS = plt.contour(
#     X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
# )
# CB = plt.colorbar(CS, shrink=0.8, extend="both")
plt.scatter(data1[:, 0], data1[:, 1], 0.8)

plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")
plt.savefig("../vil_graph/gaussian_gmm.png")
