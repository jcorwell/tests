from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage.color import rgb2grey
from skimage.transform import resize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.close('all')

class STDPNetwork(object):
    def __init__(self, **kwargs):
        self.opt = {
            "kind": "traditional",
            "A_p": 1.05,
            "A_n": 1.05,
            "initialize": np.zeros,
            "connections": {((0, 0),(1, 0)): 1.0,
                           ((0, 1),(1, 1)): 1.0,
                           ((0, 0),(1, 1)): 3.0,
                           ((0, 1),(1, 0)): 3.0,
                           ((1, 0),(0, 0)): 1.0,
                           ((1, 0),(0, 1)): 3.0,
                           ((1, 1),(0, 0)): 3.0,
                           ((1, 1),(0, 1)): 1.0},
            "setup_dict": {0: 2, 1: 2},
            "nr_units": 10
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
                                                      kind=self.opt['kind'], 
                                                      id_=(k, vi)), range(0, v))
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
            # print connk
            s0, s1 = connk
            preHn = self.Layers[s0[0]][s0[1]]
            postHn = self.Layers[s1[0]][s1[1]]
            postHn.stdp_session(p, A_n, A_p, preHn, time=time, steps=nr_iters, offset=offset)
        return [[hn.Nodes.States for hn in subpool] for subpool in self.Layers.values()]

    def select_action(self, nodes, pshow):
        l = nodes[1][0][-1]
        r = nodes[1][1][-1]
        print abs(l - r).sum()
        if np.abs(l - pshow).sum() < np.abs(r - pshow).sum():
            print "Left"
        else:
            print "Right"

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
                    "initialize": np.zeros
                    }
        self.setup_opts(kwargs)
        self.id_ = self.opt['id_']
        shape = tuple([self.opt["nr_units"]]*2)
        self.Weights = Weights(self.opt["initialize"](shape))
        self.Nodes = Nodes(self.opt['initialize'](shape))

    def setup_opts(self, kwargs):
        for k, v in kwargs.items():
            if k in self.opt.keys():
                self.opt[k] = v

    def train(self, p, kind=None):
        if kind is None: kind = self.opt["kind"]
        self.Weights.compute_weights(p, kind)

    def recall(self, p, steps=10, gamma=0.15, S=0.8, D=1.25, t0=0):
        self.Weights.update_weights(self.Nodes, p, gamma=gamma, S=S, D=D)
        return self.Nodes.Integrate(p, np.linspace(t0, steps, steps), self.Weights)

    def stdp_session(self, p, A_n, A_p, HN_pre, time=5, steps=5, kind="traditional", offset=0):
        Weights_pre = HN_pre.Weights
        self.Weights.stdp_update(A_n, A_p, self.Weights, Weights_pre, offset)
        self.Nodes.Integrate(p, np.linspace(offset, time, steps), self.Weights)


class Nodes(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        obj.States = list()
        return obj

    def Integrate(self, p, tspace, weights):
        from scipy.integrate import odeint
        def du(u, t, p):
            u = u.reshape(self.shape)
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
                    self[i, j] += p[i, j]*p[i, j]
        return self

    def update_weights(self, u, p, gamma=0.15, S=0.8, D=1.25):
        """ p = input pattern
            gamma = time dependent decay factor
            S = Hebbian coefficient
            D = mismatch coefficient """
        HLP = S*(u.T * u) - S*((1.0 - u).T * u) # Hebbian
        I_norm = (p - p.min()) / (p.max() - p.min())
        m = I_norm - u
        MID = D*(m.T * u).T # Mismatch
        delta_w = -gamma*self + MID + HLP
        self += delta_w
        return self

    def stdp_update(self, A_n, A_p, Weights0, Weightspre, offset):
        self[self > 0.5] += A_p*offset
        self[self < 0.5] += -A_n*offset
        return self
        

def HNTest():
    plt.close("all")
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, wspace=0.2)

    # Initialize Hopfield Network
    shp = 50  # (shp, shp) is shape of input patterns
    hn = HopfieldNetwork(nr_units=shp) # hopfield network instance
    
    # Load patterns
    pattern = resize(rgb2grey(data.horse()), (shp, shp))
    pattern2 = resize(rgb2grey(data.astronaut()), (shp, shp))

    # Plot first cue
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(pattern, interpolation='none', cmap='jet')

 ########################################################
    """ Training and recalling """
    hn.train(pattern, kind="traditional") # Train
    hn.train(pattern2, kind='traditional') # Train

    recov = hn.recall(pattern, steps=5) # Recall
    recov2 = hn.recall(pattern2, steps=5) # Recall
 ########################################################

    # Plot first recall
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(recov, interpolation='none', cmap='jet')

    # Plot second cue
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(pattern2, interpolation='none', cmap='jet')

    # Plot second recall
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(recov2, interpolation='none', cmap='jet')

    plt.show()
    plt.close('all')

def stdp():
    """
        Spike-timing Dependent Plasticity
        Experiment
    """
    plt.close('all')
    fig = plt.figure()
    gs = gridspec.GridSpec(5, 2, wspace=0.2)

    shp = 50 # Shape of the input cues
    
    p = resize(rgb2grey(data.horse()), (shp, shp)) # Cue A
    p2 = resize(rgb2grey(data.astronaut()), (shp, shp)) # Cue B
    p3 = np.zeros_like(p) # Hidden Cue L
    p4 = np.ones_like(p) # Hidden Cue R

    # Connections dictionary ((INPUT LAYER, ID), (OUTPUT LAYER, ID)): TIME OFFSET
    connections= {
                   ((0, 0),(1, 0)): 0,
                   ((0, 1),(1, 1)): 0,
                   ((0, 0),(1, 1)): 50,
                   ((0, 1),(1, 0)): 50,

                   ((1, 0),(0, 0)): 0,
                   ((1, 0),(0, 1)): 0,
                   ((1, 1),(0, 0)): 0,
                   ((1, 1),(0, 1)): 0}

    # Instantiate STDPNetwork
    sn = STDPNetwork(nr_units=shp, A_n=1.05, A_p=1.05, initialize=np.zeros, connections=connections)

    # Train Network on patterns (each pattern is shown only to its respective layer)
    sn.train([[p, p2], [p3, p4]])

    pshow=p # Cue for recall step
    steps = 50 # Number of time steps
    nodes = sn.recall(pshow, nr_iters=steps, time=20)
    sn.select_action(nodes, pshow)

    # Plot first cue
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.imshow(p, cmap='jet')

    # Plot second cue
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.imshow(p2, cmap='jet')

    # Left State
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.imshow(p3, cmap='jet')

    # Right State
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.imshow(p4, cmap='jet')

    # Plot first layer
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    ax5.imshow(nodes[0][0][-1], cmap='jet')

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    ax6.imshow(nodes[0][1][-1], cmap='jet')

    # Plot second layer
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.axis('off')
    ax7.imshow(nodes[1][0][-1], cmap='jet')

    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    ax8.imshow(nodes[1][1][-1],cmap='jet')

    # Plot mean of output layer
    # ax9 = fig.add_subplot(gs[4, :])
    # ax9.axis('off')
    # mn = (nodes[1][0][-1]+nodes[1][1][-2])/2.0
    # mn = (mn - mn.min()) / (mn.max() - mn.min())
    # mn[mn > 0.5] = 1.
    # mn[mn < 0.5] = 0.
    # ax9.imshow(mn, cmap='jet')


    plt.show()

    


if __name__ == '__main__':
    ''' This is where you'll run experiments.
    '''
    stdp()
    #HNTest()