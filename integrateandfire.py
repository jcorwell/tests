import numpy as np
import matplotlib.pyplot as plt

def delta_g(g_init, t_pre, t_post, tau=20, gmax=0.01, Aneg=0.525, Apos=0.500):
    delta_t = t_pre - t_post
    if delta_t < 0: F_t = Apos*np.exp(delta_t/tau)
    elif delta_t >= 0: F_t = -Aneg*np.exp(-delta_t/tau)
    d_g = gmax*F_t
    return d_g

def calculate_I(g, V, E):
    return g*(E - V)

def integrate_fire(**kwargs):
    """ Single neuron set up for STDP
    """
    # Default values (dict)
    _kw = {
            "t0": 0,
            "total_time": 30,
            "dt": 0.125,
            "t_rest": 0,
            "Cm": 10,
            "Rm": 1.9,
            "tau_ref": 5,
            "Erev": 0,
            "I": 0.1,
            "t_pre": 20,
            "t_post": 25,
            "Vrest": -0.74,
            "Vth": -0.54,
            "Vreset": -0.60,
            "Vspike": 0.1,
            "color": "b",
            "show": False,
            "gmax": 0.7
            }
    # Update _kw dict
    for k, v in kwargs.items():
        if k in _kw.keys():
            _kw.update({k: v})
    tau_m = (_kw["Rm"] / np.float32(_kw["Cm"]))*_kw["dt"]
    time = np.arange(_kw["t0"], _kw["total_time"]+_kw["dt"], _kw["dt"])
    Vm = np.zeros(len(time))
    ginit = _kw["gmax"]/2.0
    g = ginit
    I = _kw["I"]
    for i, t in enumerate(time[:-1]):
        if t > _kw["t_rest"]*_kw["dt"]:
            g = g*np.exp(-t*_kw["dt"]/tau_m)
            g += delta_g(g, _kw["t_pre"]*_kw["dt"], _kw["t_post"]*_kw["dt"], tau=tau_m, gmax=_kw["gmax"]) # normalize to dt
            if g > _kw["gmax"] or g < 0: g = ginit
            I = calculate_I(g, Vm[i], _kw["Erev"])
            Vm[i] += _kw["Vrest"] - Vm[i-1] + I
            if Vm[i] >= _kw["Vth"]: 
                Vm[i] = _kw["Vspike"]
                Vm[i + 1] = _kw["Vreset"]
                t_rest = t*_kw["dt"] + _kw["tau_ref"]*_kw["dt"]
    plt.plot(time, Vm, color=_kw["color"])
    if _kw["show"] is True: plt.show()

colors = ["r", "b", "g", "m", "y", "k"]
times = [1, 10, 5]
for ix, ts in enumerate(times):
    color = colors[ix]
    dt = 1.*(1./abs(times[ix - 1] - ts))
    print ix, color, dt, times[ix - 1] - ts
    integrate_fire(t_rest=dt+ix, dt=dt, t_pre=times[ix - 1], t_post=ts, color=color)
plt.show()



# from numpy import *
# from pylab import *

# ## setup parameters and state variables
# T = 100 # total time to simulate (msec)
# dt = 0.125 # simulation time step (msec)
# time = arange(0, T+dt, dt) # time array
# t_rest = 0 # initial refractory time
# ## LIF properties
# Vm = zeros(len(time)) # potential (V) trace over time
# Rm = 1.9 # resistance (kOhm)
# Cm = 10 # capacitance (uF)
# tau_m = Rm*Cm # time constant (msec)
# tau_ref = 4 # refractory period (msec)
# Vth = .5 # spike threshold (V)
# V_spike = 0.5 # spike delta (V)
# ## Input stimulus
# #I_syn = g_syn*(E_syn - V)
# I = 5 # input current (A)
# ## iterate over each time step
# for i, t in enumerate(time):
#  if t > t_rest:
#  	Vm[i] = Vm[i-1] + (-Vm[i-1] + I*Rm) / tau_m * dt
#  	if Vm[i] >= Vth:
#  		Vm[i] += V_spike
#  		t_rest = t + tau_ref
# # ## plot membrane potential trace

#  		# Vm = 1 * (time - Vth)/tau_m * exp(-(time - Vth - tau_m)/tau_m)

# plot(time, Vm)
# title('Leaky Integrate-and-Fire Example')
# ylabel('Membrane Potential (V)')
# xlabel('Time (msec)')
# plt.axis([0, 10, 0, 1])
# show()

#ref http://files.meetup.com/469457/spiking-neurons.pdf




