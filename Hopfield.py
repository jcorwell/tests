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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rscripts.Plot.Arrow3D import Arrow3D

try:
    __IPYTHON__
    get_ipython().magic(u'pylab')
    plt.ion()
except NameError:
    pass

Horse = rgb2grey(data.horse())


class HopfieldNetwork:
    def __init__(self, **kwargs):
        self.__var = {
                      "x": None,
                      "cue": None,
                      "weight_type": "pearson",
                      "gamma": 0.30,
                      "tau": 1.00,
                      "S_hebb": 0.20,
                      "D_fact": 0.15,
                      "s0": 1.0,
                      "time_to_recall": 1.00,
                      "format": "img_bin",
                      "verbose": 0,
                      "cue_update": "mon_incr",
                      "shape": (10, 10),
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
        self.Is = []
        self.Us = []

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
        self.Is.append(self._get("cue"))
        self.__integrate()
        self.Us.append(self.memories[-1])

    def expose(self, cue, expose_time=None):
        if expose_time is not None:
            self.update_param("time_to_recall", expose_time)
            self.timespace = np.arange(0, self._get("time_to_recall"), step=self._get("tau"))
        new_cue = rgb2grey(resize(cue, self._get("shape")))
        self.update_params({"cue": new_cue, "x": new_cue})
        self.Is.append(self._get("cue"))
        self.__integrate()
        self.Us.append(self.memories[-1])

    def __integrate(self):
        # starting place
        I0 = self.CueModifier(0.0)
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

    def __scatter(self):
        shp = self._get("shape")[0]
        X = np.array(self.Is)
        X = X.reshape((len(self.Is), shp, shp))
        S_b = np.array(list(map(lambda I_i: np.transpose(I_i - X.mean(0), axes=[0, 2, 1]) * (I_i - X.mean(0)), X))).sum(0)
        S_b = np.nan_to_num(S_b)
        S_b = (S_b - S_b.mean(0)) / S_b.std(0)
        S_b = np.nan_to_num(S_b)
        return S_b

    def __energy(self):
        u = np.array(self.Us)
        E = -0.5*((self.W*u*np.transpose(u, axes=[0, 2, 1])).sum(2)) + 0.5*np.sum(u, axis=1)
        E = np.nan_to_num(E)
        E = (E - E.mean(0)) / E.std(0)
        E = np.nan_to_num(E)
        return E

    def energy_landscape(self, plot=False):
        S_b = self.__scatter()
        E = self.__energy()
        pca = PCA(n_components=2)
        S_bpr = pca.fit_transform(S_b)
        outs = [S_bpr, E]
        if plot is True:
            ax = plt.subplot(projection='3d')
            l = np.column_stack([S_bpr, E.mean(0)])
            colors = l[:, 2]
            ax.scatter(l[:, 0], l[:, 1], l[:, 2], c=colors)
            for ix in range(l.shape[0]-1):
                lx, ly, lz = l[ix, :]
                lx1, ly1, lz1 = l[ix+1, :]
                arr = Arrow3D([lx, lx1],
                              [ly, ly1],
                              [lz, lz1],
                              arrowstyle='-|>',
                              mutation_scale=20,
                              color='k')
                ax.add_artist(arr)
            outs.append(ax)
        return outs

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
            self.W = np.nan_to_num(self.W)
            self.W[self.W > s0] = s0
            self.W[self.W < -s0] = -s0
            return self.W
        
        def __compute_weights(self):
            s0 = self.H._get("s0")
            w = getattr(self, self.wtype)()
            w[np.diag_indices_from(w)] = 0
            w = np.nan_to_num(w)
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
    hn = HopfieldNetwork(x=data.horse(), cue=data.astronaut())
    hn.train(train_time=1.0, cue=data.astronaut())
    hn.expose(expose_time=5.0, cue=data.horse())
    hn.expose(expose_time=5.0, cue=data.astronaut())
    S_b, E, ax = hn.energy_landscape()
    # cnt = 0
    # for mem in hn.memories:
    #     fig = plt.figure()
    #     im = plt.imshow(mem, cmap="jet")
    #     im.set_clim(vmin=0.0, vmax=1.0)
    #     plt.colorbar()
    #     fig.savefig("__im_%d.png" % cnt)
    #     cnt += 1
