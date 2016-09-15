from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage.color import rgb2grey
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# from matmul import matmul
# from scamul import scalarmul


class HopfieldNetwork(object):
    def __init__(self, **kwargs):
        self.opt = {
                    "kind": "pearson",
                    "patterns": [np.zeros((10, 10))]*2,
                    "nr_units": 10
                    }
        self.setup_opts(kwargs)
        self.Weights = Weights(np.zeros((self.opt["nr_units"], self.opt["nr_units"])))
        self.Nodes = Nodes(np.empty((self.opt["nr_units"], self.opt["nr_units"])))

    def setup_opts(self, kwargs):
        for k, v in kwargs.items():
            if k in self.opt.keys():
                self.opt[k] = v

    def train(self, p, kind=None):
        if kind is None: kind = self.opt["kind"]
        self.Weights.compute_weights(p, kind)

    def recall(self, p, steps=10):
        return self.Nodes.Integrate(self.Weights, p, 
                    np.linspace(0, steps, steps))

class Nodes(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def Integrate(self, weights, p, tspace):
        from scipy.integrate import odeint
        def du(t, u, p):
            p1 = np.dot(weights, u)
            sum1 = p1 + p
            du_ = -u + 0.5*(1.00 + np.tanh(sum1))
            du_ = np.nan_to_num(du_)
            return du_.flatten()
        integ = odeint(du, self.flatten(), tspace, args=(p,))
        self = integ[-1].reshape(self.shape)
        return self

class Weights(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def compute_weights(self, p, kind="pearson"):
        if kind == "pearson":
            self += np.nan_to_num(np.corrcoef(p))
            self = (self - self.min()) / (self.max() - self.min())
        elif kind == "traditional":
            for i in range(len(p)):
                for j in range(len(p)):
                    self[i, j] += p[i, j]*p[j, i]
        self[np.diag_indices(len(self))] = 0
        return self

if __name__ == '__main__':
    plt.close('all')

    shp = 50
    hn = HopfieldNetwork(nr_units=shp) # Initialize Hopfield Network
    
    # Load patterns
    pattern = resize(rgb2grey(data.horse()), (shp, shp))
    pattern2 = resize(rgb2grey(data.astronaut()), (shp, shp))

    plt.figure()
    plt.imshow(pattern, interpolation='none', cmap='jet')
    plt.colorbar()

    # plt.figure()
    # plt.imshow(pattern2, interpolation='none', cmap='jet')
    # plt.colorbar()

    hn.train(pattern, kind="pearson") # Train
    recov = hn.recall(pattern, steps=1000) # Recall

    # hn.train(pattern2, kind='traditional')
    # recov2 = hn.recall(pattern2, steps=5)

    plt.figure()
    plt.imshow(recov, interpolation='none', cmap='jet')
    plt.colorbar()

    # plt.figure()
    # plt.imshow(recov2, interpolation='none', cmap='jet')
    # plt.colorbar()


    plt.show()
    plt.close('all')
    