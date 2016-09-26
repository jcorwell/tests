from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage.color import rgb2grey
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

plt.close('all')

class STDPNetwork(object):
    """
        See stdp() function near the bottom for an example
        This class creates a Spike-Timing Dependent Plasticity Network

        -Time offsets are given in the connections kwarg
        
        -A_p, A_n are lag coefficients, which determine the amount of inhibition received
        for positively/negatively weighted units respectively
    """
    def __init__(self, **kwargs):
        self.opt = {
            "kind": "traditional",
            "A_p": 0.001,
            "A_n": 0.1,
            "initialize": np.zeros,
            "gamma": 0.5,
            "connections": {((0, 0),(1, 0)): 1.0,
                           ((0, 1),(1, 1)): 1.0,
                           ((0, 0),(1, 1)): 3.0,
                           ((0, 1),(1, 0)): 3.0,
                           ((1, 0),(0, 0)): 1.0,
                           ((1, 0),(0, 1)): 3.0,
                           ((1, 1),(0, 0)): 3.0,
                           ((1, 1),(0, 1)): 1.0},
            "setup_dict": {0: 2, 1: 2},
            "nr_units": 10,
            "thresh": 0.5
        }
        self.setup_opts(kwargs)
        self.setup_layers()

    def _get(self, kw):
        return self.opt[kw]

    def setup_opts(self, kwargs):
        for k, v in kwargs.items():
            if k in self.opt.keys():
                self.opt[k] = v

    def setup_layers(self, setup_dict=None, connections=None):
        self.Layers = dict()
        if connections is None: connections = self.opt['connections']
        if setup_dict is None: setup_dict = self.opt["setup_dict"]
        for k, v in setup_dict.items():
            subpools = map(lambda vi: HopfieldNetwork(initialize=self._get("initialize"),
                                                      nr_units=self._get("nr_units"),
                                                      kind=self._get('kind'), 
                                                      id_=(k, vi),
                                                      gamma=self._get("gamma"),
                                                      thresh=self._get("thresh")), range(0, v))
            self.Layers.update({k: subpools})

    def train(self, patterns):
        for lix, prow in enumerate(patterns):
            layer = self.Layers[lix]
            for si, subpool in enumerate(layer):
                subpool.train(prow[si])

    def recall(self, p, time=10, nr_iters=10):
        A_n = self._get("A_n")
        A_p = self._get("A_p")
        for connk, offset in self._get("connections").items():
            s0, s1 = connk
            preHn = self.Layers[s0[0]][s0[1]]
            postHn = self.Layers[s1[0]][s1[1]]
            postHn.stdp_session(p, A_n, A_p, preHn, time=time, steps=nr_iters, offset=offset)
        return [[hn.Nodes.States for hn in subpool] for subpool in self.Layers.values()]

    def select_action(self, nodes, pshow):
        l = nodes[1][0][-1]
        r = nodes[1][1][-1]
        if np.abs(l - pshow).sum() < np.abs(r - pshow).sum():
            print "Left"
        elif np.abs(l - pshow).sum() > np.abs(r - pshow).sum():
            print "Right"
        else:
            print "No choice"

class HopfieldNetwork(object):
    """
        Class that builds the Hopfield Network.
        kwargs:
            kind: possible values are ["pearson", "traditional"] (determines weights type)
            nr_units: determines the shape of weights and neurons (must be same as shape of input patterns)
    """
    def __init__(self, **kwargs):
        self.opt = {
                    "kind": "traditional",
                    "nr_units": 10,
                    "id_": (0, 0),
                    "initialize": np.zeros,
                    "gamma": 0.5,
                    "thresh": 0.5
                    }
        self.setup_opts(kwargs)
        self.id_ = self.opt['id_']
        shape = tuple([self.opt["nr_units"]]*2)
        self.Weights = Weights(self.opt["initialize"](shape), 
                               gamma=self._get("gamma"))
        # self.Weights.gamma = self.opt['gamma']
        self.Nodes = Nodes(self.opt['initialize'](shape),
                           thresh=self._get("thresh"))

    def _get(self, kw):
        return self.opt[kw]

    def setup_opts(self, kwargs):
        for k, v in kwargs.items():
            if k in self.opt.keys():
                self.opt[k] = v

    def train(self, p, kind=None):
        if kind is None: kind = self.opt["kind"]
        self.Weights.compute_weights(p, kind)

    def recall(self, p, steps=10, S=0.8, D=1.25, t0=0):
        gamma = self.opt['gamma']
        self.Weights.update_weights(self.Nodes, p, gamma=gamma, S=S, D=D)
        return self.Nodes.Integrate(p, np.linspace(t0, steps, steps), self.Weights)

    def stdp_session(self, p, A_n, A_p, HN_pre, time=5, steps=5, kind="traditional", offset=0):
        Weights_pre = HN_pre.Weights
        self.Weights.stdp_update(A_n, A_p, self.Weights, Weights_pre, offset)
        self.Nodes.Integrate(p, np.linspace(offset, offset+time, steps), self.Weights)


class Nodes(np.ndarray):
    def __new__(cls, a, thresh=0.5):
        obj = np.asarray(a).view(cls)
        obj.States = list()
        obj.thresh = thresh
        return obj

    def Integrate(self, p, tspace, weights):
        from scipy.integrate import odeint
        def du(u, t, p):
            u = u.reshape(self.shape)
            if self.thresh is not None: u = u > self.thresh
            p1 = np.dot(weights, u)
            sum1 = p1 + p
            du_ = -u + 0.5*(1.00 + np.tanh(sum1))
            return du_.flatten()
        u = np.array(self.flatten())
        integ = odeint(du, u, tspace, args=(p,))
        integ = integ[-1].reshape(self.shape)
        self.States.append(integ)
        self = integ
        return self


class Weights(np.ndarray):
    def __new__(cls, a, gamma=0.5):
        obj = np.asarray(a).view(cls)
        obj.gamma = gamma
        return obj

    def compute_weights(self, p, kind="pearson"):
        if kind == "pearson":
            self += np.nan_to_num(np.corrcoef(p))
            self = (self - self.min()) / (self.max() - self.min())
        elif kind == "traditional":
            for i in range(len(p)):
                for j in range(len(p)):
                    self[i, j] += p[i, j]*p[i, j]
        return self

    def update_weights(self, u, p, S=0.8, D=1.25):
        """ p = input pattern
            gamma = time dependent decay factor
            S = Hebbian coefficient
            D = mismatch coefficient """
        gamma = self.gamma
        HLP = S*(u.T * u) - S*((1.0 - u).T * u) # Hebbian
        I_norm = (p - p.min()) / (p.max() - p.min())
        m = I_norm - u
        MID = D*(m.T * u).T # Mismatch
        delta_w = -gamma*self + MID + HLP
        self += delta_w
        return self

    def stdp_update(self, A_n, A_p, Weights0, Weightspre, offset):
        deltap = A_p*(1. - offset)
        deltan = -A_n*(1. - offset)**2
        delta = deltap + deltan
        print "Delta_w: ", delta
        self += (delta*Weightspre - self.gamma*self)
        return self
