# coding: utf-8

import numpy as np
from scipy.integrate import odeint
from sklearn.decomposition import PCA

from skimage import data, io
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage.color import rgb2grey
from skimage.transform import resize

import gc
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from rscripts.Plot.Arrow3D import Arrow3D
from Arrow3D import Arrow3D

import scipy.io.wavfile as wav

try:
    __IPYTHON__
    get_ipython().magic(u'pylab')
    # plt.ion()
except NameError:
    pass

# Horse = rgb2grey(data.horse())
global AUDIO_
AUDIO_ = False

class HopfieldNetwork:
    def __init__(self, **kwargs):
        self.__var = {
                      "x": None,
                      "cue": None,
                      "weight_type": "pearson",
                      "gamma": 0.15,
                      "tau": 1.00,
                      "S_hebb": 0.80,
                      "D_fact": 1.25,
                      "s0": 1.0,
                      "time_to_recall": 1.00,
                      "format": "img_bin",
                      "verbose": 0,
                      "cue_update": "mon_incr",
                      "shape": (5, 5),
                      "tol": 1.49012e-8,
                      "full_output": 0,
                      "hmin": 1e-86}
        self.__onstart(kwargs)

    def __call__(self, **kwargs):
        self.__update_options(kwargs)
        self.__integrate()
        return {"Weights": self.WeightsContainer, "Cues": self.CueModifier, "Options": self.__var}

    def __onstart(self, kwargs):
        self.__update_options(kwargs)
        try:
            self.__format_input()
        except AttributeError:
            pass
        self.timespace = np.arange(0, self._get("time_to_recall"), step=self._get("tau"))
        self.total_time = 0.0
        self.memories = list()
        self.Is = list()
        self.Us = list()
        self.starting_time = 0.0
        self.WeightsContainer = self.HopfieldWeights(self)
        self.CueModifier = self.CueMod(self)

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
            if self._get("verbose") >= 1: print input_img, " size:", self._get(input_img).shape
        gc.collect()

    def load_im(self, fn, format=""):
        img = io.imread(fn, as_grey=True)
        self.update_param("x", img)
        gc.collect()

    def train(self, cue):
        new_cue = rgb2grey(resize(cue, self._get("shape")))
        self.update_param('x', new_cue)
        self.WeightsContainer() # compute weights
        # print self.W

    def reinforce(self, x0, x1, mixture=2.0, expose_time=None):
        """
            Reinforce:
            1) Update both x and cue
            2) Set mixture
            3) Append to Is
            4) Integrate for 1 time step
        """
        x0 = rgb2grey(resize(x0, self._get("shape")))
        x1 = rgb2grey(resize(x1, self._get("shape")))
        self.starting_time = mixture
        self.update_params({"x": x0, "cue": x1})
        self.Is.append(self._get("cue"))
        if expose_time is not None:
            self.update_param("time_to_recall", expose_time)
            self.timespace = np.arange(0, self._get("time_to_recall"), step=self._get("tau"))
        self.__integrate()
        self.Us.append(self.memories[-1])

    def expose(self, cue, expose_time=None):
        """
            Expose:
            1) Replace cue
            2) Add new cue to list of Is
            3) Integrate for expose time & update weights
        """
        if expose_time is not None:
            self.update_param("time_to_recall", expose_time)
            self.timespace = np.arange(0, self._get("time_to_recall"), step=self._get("tau"))
        new_cue = rgb2grey(resize(cue, self._get("shape")))
        self.update_cue(new_cue)
        self.Is.append(self._get("cue"))
        self.__integrate()
        self.Us.append(self.memories[-1])

    def __integrate(self):
        I0 = self.CueModifier(self.total_time) # starting place
        timespace = self.timespace
        odeout = odeint(self.__iter, I0.flatten(), timespace, 
                         atol=self._get("tol"), rtol=self._get("tol"),
                         full_output=self._get("full_output"),
                         hmin=self._get("hmin")
                        )
        memory_ = odeout
        self.memories.extend(list(map(lambda mem: mem.reshape(self._get("x").shape), memory_)))
        gc.collect()

    def __iter(self, u, t):
        u = u.reshape(self._get("x").shape)
        I = self.CueModifier(t + self.total_time)
        du_ = (-u + 0.50*(1.00 + np.tanh(np.sum(self.W*u.T*self.W.T, axis=1) + I.T)))
        self.W = self.WeightsContainer.update_weights(u)
        if self._get("verbose") >= 2:
            print("W: ", self.W.min(), self.W.max(), self.W.mean())
            print("du: ", self.W.min(), self.W.max(), self.W.mean())
        self.total_time += 1 # add to total run time
        return du_.flatten()

    def __scatter(self):
        Sbs = list()
        for I in self.Is:
            pca = PCA(n_components=2)
            S_b = pca.fit_transform(I)
            Sbs.append(S_b)
        Sbs = np.array(Sbs)
        Sbs = Sbs.mean(1)
        Sbs -= Sbs.min()
        Sbs /= (Sbs.max() - Sbs.min())
        return Sbs

    def __energy(self):
        Es = list()
        for u in self.Us:
            E = -0.5*(self.W*u*u.T).sum() + 0.5*(u.sum())
            Es.append(E)
        Es = np.array(Es)
        Es -= Es.min()
        Es /= (Es.max() - Es.min())
        return Es

    def energy_landscape(self, plot=False, outname="__energy_landscape.png"):
        S_b = self.__scatter()
        E = self.__energy()
        if plot is True:
            fig = plt.figure()
            ax = plt.subplot(projection='3d')
            ax.plot(S_b[:, 0], S_b[:, 1], zs=E, marker='o')
            for ii in range(0, len(S_b)-1):
                startx, starty = S_b[ii, :]
                endx, endy = S_b[ii+1, :]
                startz, endz = E[ii], E[ii+1]
                arrow=Arrow3D([startx, endx],
                        [starty, endy],
                        [startz, endz],
                        arrowstyle='-|>',
                        mutation_scale=20,
                        color='k')
                ax.add_artist(arrow)
            fig.savefig(outname)

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

        def HLP(self, curr_u):
            S_hebb = self.H._get("S_hebb")
            u = curr_u.copy()
            HLP = S_hebb*(u.T*u) - S_hebb*((1.00 - u).T * u)
            return HLP

        def MID(self, curr_u):
            D_fact = self.H._get("D_fact")
            I = self.I.copy()
            I_norm = self.I.copy()
            I_norm -= I.min()
            I_norm /= (I.max() - I.min())
            u = curr_u.copy()
            m = I_norm - u
            MID = D_fact*(m.T * u)
            return MID

        def update_weights(self, curr_u):
            s0 = self.H._get("s0")
            T_new = self.MID(curr_u) + self.HLP(curr_u)
            self.W = self.W + (-self.H._get("gamma")*self.W + T_new)
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

        def hebbian(self):
            return np.dot(self.x, self.x.T) / (np.sum(self.x.shape))
        
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
            return I


def image_cues(shape):
    x0 = data.horse()
    x1 = data.astronaut()
    x2 = data.chelsea()
    # rs = np.random.RandomState()
    # rs.seed(135221)
    # x0 = np.random.choice([0, 10], size=shape, replace=True)
    # x1 = np.random.choice([0, 10], size=shape, replace=True)
    # x2 = np.random.choice([0, 10], size=shape, replace=True)
    xs = [x0, x1, x2]
    return xs

def audio_cues(times=3):
    global AUDIO_
    AUDIO_ = True
    import pyaudio, thread
    def input_thread(l):
        raw_input()
        l.append(None)
    CHUNKSIZE = 2048
    global RATE
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNKSIZE)
    cues = list()
    streams = list()
    ts = list()
    tt = 0
    while tt < times:
        print "Stream: %d" % tt
        l = list()
        thread.start_new_thread(input_thread, (l,))
        t0 = time.time()
        while not l:
            # do this as long as you want fresh samples:
            data = stream.read(CHUNKSIZE)
            numpydata = np.fromstring(data, dtype=np.float32)
            streams.append(numpydata)
            numpydata = numpydata.reshape((1,CHUNKSIZE))
        ts.append(time.time()-t0)
        cues.append(numpydata)
        np.save("__stream_%d" % tt, numpydata)
        tt += 1
    stream.stop_stream()
    stream.close()
    p.terminate()
    wav.write("streams.wav", RATE, np.hstack(streams))
    return cues, ts

def do_audio_img_pair():
    xs, ts = audio_cues(times=3)
    mlx = np.mean([len(xx) for xx in xs])
    # shape = (round(mlx)**3, round(mlx)**3)
    shape = (100, 100)
    padsize = int((shape[0]-1.0)/2.0)
    xs = map(lambda npdat: np.pad(npdat, pad_width=((padsize, padsize), (0,0)), mode='constant', constant_values=0), xs)
    print "TIME:", np.mean(ts)
    xs = xs[0:-1]
    imcues = image_cues()
    xs.append(imcues[0])
    x0, x1, x2 = xs
    cnt1 = 0
    for xx in xs:
        fig1 = plt.figure()
        im = plt.imshow(xx, cmap="jet")
        im.set_clim(vmin=0, vmax=1)
        plt.colorbar()
        fig1.savefig("X%d.png" % cnt1)
        cnt1 += 1
    hn = HopfieldNetwork(x=x0, cue=x2, shape=shape, hmin=1e-86, full_output=1)
    hn.learn(cue=x0)
    hn.learn(cue=x1)
    hn.learn(cue=x2)
    hn.reinforce(x0, x2, mixture=1.0)
    hn.expose(cue=x1, expose_time=10.0)
    hn.expose(cue=x2, expose_time=10.0)
    S_b, E, el_ax, el_fig = hn.energy_landscape(plot=True)
    cnt = 0
    amems = list()
    for mem in hn.memories:
        if AUDIO_ is True:
            amems.append(mem[int(mem.shape[0]/2.)])
        fig = plt.figure()
        im = plt.imshow(mem, cmap="jet")
        im.set_clim(vmin=0.0, vmax=1.0)
        plt.colorbar()
        fig.savefig("__state_%d.png" % cnt)
        cnt += 1
    if AUDIO_ is True:
        amem = np.hstack(amems).astype(np.float32)
        wav.write("audio_learning.wav", RATE, amem)
