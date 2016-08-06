# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage.color import rgb2grey
from skimage.transform import resize
import gc
import time
import multiprocessing as mp

try:
    __IPYTHON__
    get_ipython().magic(u'pylab')
    plt.ion()
except NameError:
    pass

Horse = rgb2grey(data.horse())
# cue1 = random_noise(Horse, mode="speckle", var=0.01)
# cue1 = data.astronaut()

class HopfieldNetwork:
    def __init__(self, **kwargs):
        self.__var = {
                      "x": None,
                      "cue": None,
                      "weight_type": "pearson",
                      "gamma": 0.05,
                      "tau": 1.00,
                      "S_hebb": 0.80,
                      "D_fact": 0.15,
                      "s0": 1.0,
                      "time_to_recall": 1.00,
                      "format": "img_bin",
                      "verbose": 0,
                      "cue_update": "mon_incr",
                      "shape": (20, 20),
                      "anim": True,
                      "atol": 1.49012e-8}
        self.__onstart(kwargs)
        self.WeightsContainer = self.HopfieldWeights(self)
        self.WeightsContainer()
        self.CueModifier = self.CueMod(self)

    def __call__(self, **kwargs):
        self.train()

    def __onstart(self, kwargs):
        self.__update_options(kwargs)
        self.__format_input()
        self.timespace = np.arange(0, self._get("time_to_recall"), step=self._get("tau"))
        self.memories = []

    def __update_options(self, kwargs):
        self.__var.update(kwargs)

    def __format_input(self):
        input_ = self._get("x").copy()
        if "img" in self._get("format"):
            self.update_param("x", rgb2grey(input_))
            self.update_cue(rgb2grey(self._get("cue")))
        for input_img in ["x", "cue"]:
            self.update_param(input_img, resize(self._get(input_img),
                                                self._get("shape")))
            if self._get("verbose") is True: print input_img, " size:", self._get(input_img).shape
        gc.collect()

    def load_im(self, fn, format=""):
        img = io.imread(fn, as_grey=True)
        self.update_param("x", img)
        gc.collect()

    def train(self, train_time=None, cue=None):
        if train_time is not None: 
            self.update_param("time_to_recall", train_time)
            self.timespace = np.arange(0, self._get("time_to_recall"), step=self._get("tau"))
        if cue is not None:
            cue_reshape = rgb2grey(resize(cue, self._get("shape")))
            self.update_params({"cue": cue_reshape}) # "x": cue_reshape, 
        self.__integrate()

    def expose(self, cue, expose_time=None):
        if expose_time is not None:
            self.update_param("time_to_recall", expose_time)
            self.timespace = np.arange(0, self._get("time_to_recall"), step=self._get("tau"))
        new_cue = rgb2grey(resize(cue, self._get("shape")))
        self.update_params({"cue": new_cue, "x": new_cue})
        self.__integrate()

    def __integrate(self):
        # starting place
        I0 = self.CueModifier(0.1)
        timespace = self.timespace
        memory_ = odeint(self.__iter, I0.flatten(), timespace, atol=self._get("atol"))
        self.memories.extend(list(map(lambda mem: mem.reshape(self._get("x").shape), memory_)))
        gc.collect()

    def __iter(self, u, t):
        u = u.reshape(self._get("x").shape)
        self.W = self.WeightsContainer.update_weights(u)
        I = self.CueModifier(t)
        du_ = -u + 0.50*(1.00 + np.tanh(np.sum(np.dot(self.W, u) + I, axis=1)))
        if self._get("verbose") >= 2:
            print(self.W.min(), self.W.max(), self.W.mean())
        return du_.flatten()

    def HLP(self, curr_u):
        S_hebb = self._get("S_hebb")
        u = curr_u.copy()
        HLP = S_hebb*((u.T*u))-S_hebb*((1.00 - u).T * u)
        gc.collect()
        return HLP

    def MID(self, curr_u):
        D_fact = self._get("D_fact")
        I = self._get("cue").copy()
        I_norm = self._get("cue").copy()
        I_norm -= I.min()
        I_norm /= (I.max() - I.min())
        u = curr_u.copy()
        m = I_norm - u
        MID = D_fact*(m.T * u)
        gc.collect()
        return MID

    def update_cue(self, new_cue):
        self.__var.update({"cue": new_cue})

    def update_wtype(self, new_wtype):
        self.__var.update({"weight_type": new_wtype})

    def update_param(self, param_name, new_param):
        self.__var.update({param_name: new_param})

    def update_params(self, new_params):
        self.__var.update(new_params)

    def _get(self, which):
        return self.__var[which]

    class HopfieldWeights:
        def __init__(self, H):
            self.H = H

        def __call__(self):
            return self.__compute_weights()

        def update_weights(self, curr_u):
            s0 = self.H._get("s0")
            T_new = self.H.MID(curr_u) + self.H.HLP(curr_u)
            self.W += -self.H._get("gamma")*self.W + T_new
            self.W[self.W > s0] = s0
            self.W[self.W < -s0] = -s0
            return self.W
        
        def __compute_weights(self):
            s0 = self.H._get("s0")
            w = getattr(self, self.wtype)()
            w[np.diag_indices_from(w)] = 0
            w[w > s0] = s0
            w[w < -s0] = -s0
            setattr(self.H, 'W', w)
            gc.collect()
            return self.W

        def pearson(self):
            return np.corrcoef(self.x)
        
        @property
        def wtype(self):
            return self.H._get("weight_type")

        @property
        def x(self):
            return self.H._get("x")

        @property
        def W(self):
            return self.H.W

        @property
        def I(self):
            return self.H._get("cue")

    class CueMod:
        def __init__(self, H):
            self.H = H

        def __call__(self, t):
            return self.__mod_cue(t)

        @property
        def I(self):
            return self.H._get("cue")

        def __mod_cue(self, t):
            return getattr(self, self.H._get("cue_update"))(t)

        def random(self, t):
            I = self.I
            I -= np.tanh(np.random.random(I.shape))
            self.H.update_param("cue", I)
            return I

        def sine(self, t):
            I = self.I
            I -= np.sin(I)*self.H._get("gamma")
            self.H.update_param("cue", I)
            return I

        def none(self, t):
            return self.I

        def mon_incr(self, t):
            f_t = 1.00 / (1.00 + np.exp(self.H._get("time_to_recall") / 2-t))
            I0 = self.H._get("x")
            I1 = self.H._get("cue")
            I = I0 + (I1 - I0)*f_t
            I_ = I.copy()
            I -= I_.min()
            I /= (I_.max() - I_.min())
            self.H.update_cue(I)
            gc.collect()
            return I


if __name__ == '__main__':
    hn = HopfieldNetwork(x=data.astronaut(), cue=data.horse())
    hn.train(train_time=1.0, cue=data.horse())
    hn.expose(expose_time=10.0, cue=data.astronaut())
    cnt = 0
    for mem in hn.memories:
        fig = plt.figure()
        im = plt.imshow(mem, cmap="jet")
        im.set_clim(vmin=0.0, vmax=1.0)
        plt.colorbar()
        fig.savefig("__im_%d.png" % cnt)
        cnt += 1
