# Edit this file.
## Tests.py

from Hopfield2 import *


### Each function below is a separate test.
### Scroll to the bottom to execute different tests.
### NOTE: Hashtags (#) are comments in Python ;-)

def HNTest():
    '''
        HOPFIELD NETWORK TEST
    '''
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
    ''' 
        Spike-timing Dependent Plasticity TEST
    '''
    plt.close('all')

    shp = 20 # Shape of the input cues
    
    p = resize(rgb2grey(data.horse()), (shp, shp)) # Cue A
    p2 = resize(rgb2grey(data.astronaut()), (shp, shp)) # Cue B
    # np.random.seed(14)
    # p = np.random.random((shp,shp))
    # p2 = np.random.random((shp,shp))

    p3 = np.zeros_like(p) # Hidden Cue L
    p4 = np.zeros_like(p) # Hidden Cue R

    # Connections dictionary 
    # ((INPUT LAYER #, ID#), (OUTPUT LAYER #, ID#)): TIME OFFSET
    connections = {
                   ((0, 0),(1, 0)): 0,
                   ((0, 1),(1, 1)): 0,
                   ((0, 0),(1, 1)): 10,
                   ((0, 1),(1, 0)): 10,

                   ((1, 0),(0, 0)): 0,
                   ((1, 1),(0, 1)): 0,
                   ((1, 0),(0, 1)): 10,
                   ((1, 1),(0, 0)): 10}

    # Instantiate STDPNetwork
    sn = STDPNetwork(nr_units=shp, A_n=0.1, A_p=0.1, gamma=0.1, initialize=np.zeros, connections=connections)

    # Train Network on patterns (each pattern is shown only to its respective layer)
    sn.train([[p, p2], [p3, p4]])
    
    # Cue for recall step
    for PSHOW in [p, p2]:
        fig = plt.figure(figsize=(10,10))
        gs = gridspec.GridSpec(5, 2, wspace=0.2, hspace=1.)
        if (PSHOW == p).all(): fig.suptitle("A")
        elif (PSHOW == p2).all(): fig.suptitle("B")
        steps = 30 # Number of time steps

        # Recall given cue
        nodes = sn.recall(PSHOW, nr_iters=steps, time=steps*2)
        # Select action
        sn.select_action(nodes, PSHOW)

        # Plot first cue
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Cue A")
        ax1.axis('off')
        ax1.imshow(p, cmap='jet', interpolation='none', vmin=0, vmax=1)

        # Plot second cue
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Cue B")
        ax2.axis('off')
        ax2.imshow(p2, cmap='jet', interpolation='none', vmin=0, vmax=1)

        # Left State
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title("Hidden L Cue")
        ax3.axis('off')
        ax3.imshow(p3, interpolation='none', vmin=0, vmax=1)

        # Right State
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title("Hidden R Cue")
        ax4.axis('off')
        ax4.imshow(p4, cmap='jet', interpolation='none', vmin=0, vmax=1)

        # Plot first layer
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_title("Sensory Layer: A")
        ax5.axis('off')
        ax5.imshow(nodes[0][0][-1], cmap='jet', interpolation='none', vmin=0, vmax=1)

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_title("Sensory Layer: B")
        ax6.axis('off')
        ax6.imshow(nodes[0][1][-1], cmap='jet', interpolation='none', vmin=0, vmax=1)

        # Plot second layer
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.set_title("Hidden Layer: L")
        ax7.axis('off')
        ax7.imshow(nodes[1][0][-1], cmap='jet', interpolation='none', vmin=0, vmax=1)

        ax8 = fig.add_subplot(gs[3, 1])
        ax8.set_title("Hidden Layer: R")
        ax8.axis('off')
        ax8.imshow(nodes[1][1][-1],cmap='jet', interpolation='none', vmin=0, vmax=1)

    plt.show()

    
if __name__ == '__main__':
    ''' This is where you'll run experiments.
    '''
    stdp()
    #HNTest()
