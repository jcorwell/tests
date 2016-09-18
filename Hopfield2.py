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
            "nr_units": 10,
            "nr_layers": 2,
            "A_p": 1.05,
            "A_n": 1.05,
            "offsets": [np.arange(0, 10)]*2,
            "initialize": np.zeros
        }
        self.setup_opts(kwargs)
        self.Layers = map(lambda hni: HopfieldNetwork(kind=self.opt['kind'], nr_units=self.opt['nr_units'], 
                                        order=hni, offset=self.opt["offsets"][hni], initialize=self.opt['initialize']), 
                            range(0, self.opt['nr_layers']))

    def _get(self, kw):
        return self.opt[kw]

    def setup_opts(self, kwargs):
        for k, v in kwargs.items():
            if k in self.opt.keys():
                self.opt[k] = v

    def full_session(self, p, total_steps=10, gamma=0.15, S=0.8, D=1.25):
        t0 = 0
        A_n = self._get("A_n")
        A_p = self._get("A_p")
        HN0 = self.Layers[0]
        HN0.train(p, kind=self._get("kind"))
        HN0.recall(p, steps=2, t0=0, gamma=gamma, S=S, D=D)
        ii = 0
        while ii <= total_steps-2:
            print ii
            if t0 == 0: stix = 1
            else: stix = 0
            for HN in self.Layers[stix:]:
                HN.stdp_session(p, A_n, A_p, self.Layers[HN.order - 1], kind=self._get("kind"), 
                                    steps=2, t0=t0, gamma=gamma, S=S, D=D, nr_step=ii)
                t0 += 2
            ii += 1
        return [hn.Nodes.States for hn in self.Layers]


class HopfieldNetwork(object):
    """
        Class that builds the Hopfield Network.
        kwargs:
            kind: possible values are ["pearson", "traditional"] (determines weights type)
            nr_units: determines the shape of weights and neurons (must be same as shape of input patterns)
    """
    def __init__(self, **kwargs):
        self.opt = {
                    "kind": "pearson",
                    "nr_units": 10,
                    "order": 0,
                    "offset": [[2]],
                    "initialize": np.zeros
                    }
        self.setup_opts(kwargs)
        self.order = self.opt['order']
        shape = tuple([self.opt["nr_units"]]*2)
        self.Weights = Weights(self.opt["initialize"](shape))
        self.Nodes = Nodes(np.random.random(shape))

    def setup_opts(self, kwargs):
        for k, v in kwargs.items():
            if k in self.opt.keys():
                self.opt[k] = v

    def train(self, p, kind=None):
        if kind is None: kind = self.opt["kind"]
        self.Weights.compute_weights(p, kind)

    def recall(self, p, steps=10, gamma=0.15, S=0.8, D=1.25, t0=0, nr_step=0):
        t0 += self.opt["offset"][nr_step]
        self.Weights.update_weights(self.Nodes, p, gamma=gamma, S=S, D=D)
        return self.Nodes.Integrate(self.Weights, p, 
                    np.linspace(t0, steps, steps))

    def stdp_session(self, p, A_n, A_p, HN_pre, steps=5, gamma=0.15, S=0.8, D=1.25, kind="traditional", t0=0, nr_step=0):
        Nodes_pre = HN_pre.Nodes
        self.train(p, kind=kind)
        self.recall(p, steps=steps, gamma=gamma, S=S, D=D, t0=t0, nr_step=nr_step)
        self.Weights.stdp_update(A_n, A_p, self.Nodes.States, Nodes_pre)


class Nodes(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        obj.States = list()
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
        self.States.append([si.reshape(self.shape) for si in integ][-1])
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

    def stdp_update(self, A_n, A_p, States0, Nodes_pre):
        States_pre = Nodes_pre.States
        # print len(States_pre), len(States0)
        dtps = (States_pre[-1] - States0[-1])
        dtns = (States0[-1] - States_pre[-1])
        delta_p = A_p*self*dtps
        delta_n = A_n*self*dtns
        dw_dt = np.nan_to_num(delta_p - delta_n)
        dw_dt = (1.0 + np.tanh(dw_dt)) / 2.0
        # print dw_dt.sum()
        self += dw_dt
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
    import matplotlib.animation as animation
    plt.close('all')
    fig = plt.figure()
    # gs = gridspec.GridSpec(4, 2, wspace=0.2)

    shp = 50
    
    p = resize(rgb2grey(data.horse()), (shp, shp))
    p2 = resize(rgb2grey(data.astronaut()), (shp, shp))

    steps = 500
    # offsets = [[ii for ii in range(0, steps*3, 3)]]*2
    # offsets = offsets + [[ii for ii in range(steps, 0, -1)]]*2
    offx = np.linspace(0, steps, steps)
    offsets = list()
    offsets.append(np.exp(0.25*offx)+np.power(0.25*offx, 2)-np.power(0.25*offx, 3))
    offsets.append((1.0 + np.tanh(offx))*steps/2.0)
    offsets.append(offx*0.5)
    offsets.append(-np.power((offx-3.), 3.)+np.power(offx, 2.))
    sn = STDPNetwork(nr_layers=4, nr_units=shp, A_n=0.1, A_p=0.1, offsets=offsets, initialize=np.zeros)

    nodes = sn.full_session(p, total_steps=steps, gamma=0.05, D=1.0, S=0.75)

    # im1 = plt.imshow(nodes[0][0], interpolation='none', cmap='jet')

    # ax1 = fig.add_subplot(gs[0, 0])
    # ax1.imshow(p, interpolation='none', cmap='jet')

    # im1 = ax2 = fig.add_subplot(gs[0, 1])
    # ax2.imshow(nodes[0][0], interpolation='none', cmap='jet')

    # # Plot second cue
    # ax3 = fig.add_subplot(gs[1, 0])
    # ax3.imshow(p2, interpolation='none', cmap='jet')

    # # Plot second recall
    # ax4 = fig.add_subplot(gs[1, 1])
    # im2 = ax4.imshow(nodes[1][0], interpolation='none', cmap='jet')

    # ax5 = fig.add_subplot(gs[2, 1])
    # im3 = ax5.imshow(nodes[2][0], interpolation='none', cmap='jet')

    # ax6 = fig.add_subplot(gs[3, 1])
    # im4 = ax6.imshow(nodes[3][0], interpolation='none', cmap='jet')

    # def updatefig(*args):
    #     im1.set_array(nodes[0][i])
    #     # im2.set_array(nodes[1][i])
    #     # im3.set_array(nodes[2][i])
    #     # im4.set_array(nodes[3][i])
    #     return im1,

    ims = map(lambda i: [plt.imshow(nodes[0][i], interpolation='none', cmap='jet')], range(len(nodes[0])))

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    ani.save("STDPNetwork.mp4")
    plt.show()

    


if __name__ == '__main__':
    ''' This is where you'll run experiments.
    '''
    stdp()