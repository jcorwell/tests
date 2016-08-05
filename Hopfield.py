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

try:
    __IPYTHON__
    get_ipython().magic(u'pylab')
    plt.ion()
except NameError:
    pass

Horse = rgb2grey(data.horse())
# cue1 = random_noise(Horse, mode="speckle", var=0.01)
cue1 = data.astronaut()

class HopfieldNetwork:
    def __init__(self, **kwargs):
        self.__var = {
                      "x": Horse,
                      "cue": cue1,
                      "weight_type": "pearson",
                      "gamma": 0.01,
                      "tau": 1.00,
                      "time_to_recall": 10.00,
                      "format": "img_bin",
                      "verbose": True,
                      "cue_update": "sine",
                      "shape": (100, 100)}
        self.__onstart(kwargs)
        self.WeightsContainer = self.HopfieldWeights(self)
        self.CueModifier = self.CueMod(self)

    def __call__(self, **kwargs):
        self.train()
        return self.remember()

    def __onstart(self, kwargs):
        self.__update_options(kwargs)
        self.__format_input()
        self.W = np.zeros([self._get("x").shape[0]]*2)
        self.last_memory = np.zeros_like(self._get("x"))
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
            if self._get("verbose") is True: print self._get(input_img).shape
        gc.collect()

    def load_im(self, fn, format=""):
        img = io.imread(fn, as_grey=True)
        self.__var.update({"x": img > threshold_otsu(img)})
        gc.collect()

    def train(self):
        self.WeightsContainer() # updates weights
        return self.W

    def __recall(self, u, t):
        I = self._get("cue")
        I = self.CueModifier()
        u = u.reshape(self._get("x").shape)
        I = I.reshape(self._get("cue").shape)
        du_ = u.copy()
        du_ = -u + 0.50*(1.00 + np.tanh(np.sum(np.dot(self.W, u) + I, axis=1)))
        if self._get("verbose") is True: print("t = ", t,"Mean change: ", np.mean(du_))
        return du_.flatten()

    def __integrate(self):
        tau = self.__var["tau"]
        I = self.__var["cue"].flatten()
        timespace = np.arange(0, self._get("time_to_recall"), step=tau)
        memory_ = odeint(self.__recall, I, timespace)
        self.last_memory = memory_[-1].reshape(self._get("x").shape)
        self.memories.extend(list(map(lambda mem: mem.reshape(self._get("x").shape), memory_)))
        gc.collect()
        return self.last_memory

    def remember(self, I=None):
        if I is not None: self.update_param({"cue": I})
        return self.__integrate()

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

        def update_weights(self):
            self.W -= self.H._get("gamma")*self.W
            return self.W
        
        def __compute_weights(self):
            w = getattr(self, self.wtype)()
            w[np.diag_indices_from(w)] = -1
            w = np.tanh(np.nan_to_num(w))
            setattr(self.H, 'W', w)
            gc.collect()

        def pearson(self):
            return np.corrcoef(self.x)

        def hebbian(self):
            print("Coming soon...")
            pass

    class CueMod:
        def __init__(self, H):
            self.H = H

        def __call__(self):
            return self.__mod_cue()

        @property
        def I(self):
            return self.H._get("cue")

        def __mod_cue(self):
            return getattr(self, self.H._get("cue_update"))()

        def random(self):
            I = self.I
            I -= np.tanh(np.random.random(I.shape))
            self.H.update_param("cue", I)
            return I

        def sine(self):
            I = self.I
            I -= np.sin(I)
            self.H.update_param("cue", I)
            return I



if __name__ == '__main__':
    hn = HopfieldNetwork(time_to_recall=10.0)
    hn()
    plt.figure()
    plt.imshow(Horse)
    for im in hn.memories:
        plt.figure()
        plt.imshow(im)