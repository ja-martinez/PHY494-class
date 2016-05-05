import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pendulums as pd

# This text files compares the functions for the analysis of the simulations of the pendulums


def getFrequency(phi, t, h):
    """ 
    This function calculates the frequency of a parameter using FFT
    phi corresponds to the parameter array
    t corresponds to the time array
    h corresponds to the time step size
    
    """
    sp = np.fft.fft(phi)
    spfreq = np.fft.fftfreq(t.size, h)
    powersp = np.abs(sp)**2
    frequency = np.abs(spfreq[np.argmax(powersp)])
    return frequency



def compare_simpleDouble():
    """
    This function compares the simple and double pendulums and
    prints out the frequency of each pendulum. 
    Does not return anything
    
    """
    phiSimple, tSimple = pd.simplePendulum(0.5, 0.5, 0.5, 0, .0001, 75)
    phiDouble, tDouble = pd.rk4DobPend(0.5, 0.0000001, 0.5, 0.0000001, 0.5, 0.0, 0.0, 0.0, .0001, 75, linearized=False, \
                                       purpose="comparison")
    
    freqSimple = getFrequency(phiSimple, tSimple, 0.0001)
    freqDouble = getFrequency(phiDouble, tDouble, 0.0001)
    
    print("main frequency for simple pendulum: {0:.3f}  period: {1:.3f}".format(freqSimple, 1/freqSimple))
    print("main frequency for double pendulum: {0:.3f}  period: {1:.3f}".format(freqDouble, 1/freqDouble))
    return


def compare_linearNonlinear():
    """
    This function compares the linear (small angle approximation) 
    and nonlinear (general) solutions for the double pendulum at 
    small angles and prints out the frequency of each pendulum. 
    Does not return anything
    
    """
    phiSmall, tSmall = pd.rk4DobPend(0.1, 0.1, 0.5, 0.5, 0.01, 0.0, 0.0, 0.0, .0001, 75, linearized=True, purpose="comparison")
    phiGen, tGen = pd.rk4DobPend(0.1, 0.1, 0.5, 0.5, 0.01, 0.0, 0.0, 0.0, .0001, 75, linearized=False, purpose="comparison")
    
    freqSmall = getFrequency(phiSmall, tSmall, 0.0001)
    freqGen = getFrequency(phiGen, tGen, 0.0001)
    
    print("main frequency for pendulum with small angle approximation: {0:.3f}  period: {1:.3f}".format(freqSmall, 1/freqSmall))
    print("main frequency for general pendulum: {0:.3f}  period: {1:.3f}".format(freqGen, 1/freqGen))
    return


def getCoordinates(L1, L2, phi1, phi2):
    """
    This function outputs the coordinates for the upper and lower parts of the pendulum.
    Used to graph trajectory and calculate energy.
    L1, L2, phi1, phi2 represent the length of the upper pendulum, length of the lower pendulum,
    angle of the upper pendulum, and angle of the lower angle, respectively.
    
    """
    
    y1 = -L1*np.cos(phi1)   
    x1 = L1*np.sin(phi1)   
    y2 = y1-L2*np.cos(phi2)  
    x2 = x1+L2*np.sin(phi2)
    
    return x1, y1, x2, y2


def graphTrajectory(L1, L2, phi1, phi2, filename='Trajectory.png'):
    """
    This function displays the trajectory of the upper and lower parts of the pendulum and
    saves it in a file determined by filename parameter.
    L1, L2, phi1, phi2 represent the length of the upper pendulum, length of the lower pendulum,
    angle of the upper pendulum, and angle of the lower angle, respectively.
    
    """
    x1, y1, x2, y2 = getCoordinates(L1, L2, phi1, phi2)
    
    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(211)
    ax1.plot(x1, y1, label=r"Upper")
    ax1.plot(x2, y2, label=r"Lower")
    ax1.legend(loc='lower right', labelspacing=0.1, borderpad=0.01)
    plt.xlabel('x')
    plt.ylabel('y')
    ax1.set_aspect(1)
    fig.savefig(filename)
    
    return


def graphEnergy(L1, L2, m1, m2, phi1, phi2, w1, w2, t, g=9.81, filename='Energy'):
    """
    This function displays the plot for the kinetic, potential, and hamiltonian energies,
    and saves it in a file determined by filename parameter.
    L1, L2, m1, m2, phi1, phi2, w1, w2, t represent the length of the upper pendulum, length of the lower pendulum,
    mass of upper pendulum, mass of lower pendulum, angle of the upper pendulum, and angle of the lower angle, 
    upper pendulum angular velocity, and lower pendulum angular velocity, and time, respectively.
    
    """
    
    x1, y1, x2, y2 = getCoordinates(L1, L2, phi1, phi2)
    
    V = -(m1+m2)*g*L1*np.cos(phi1) - m2*g*L2*np.cos(phi2)
    T = 0.5*m1*(L1*w1)**2 +0.5*m2*((L1*w1)**2 + (L2*w2)**2+2*L1*L2*w1*w2*np.cos(phi1-phi2))
    H = T + V
    
    fig = plt.figure(figsize=(7,7))
    ax2 = fig.add_subplot(311)
    ax2.plot(t, T, label=r"Kinetic")
    ax2.plot(t, V, label=r"Potential")
    ax2.plot(t, E, label=r"Hamiltonian")
    ax2.legend(loc='lower right', labelspacing=0.1, borderpad=0.01)
    plt.xlabel('Time (s)')
    plt.ylabel('Energy ( J )')
    fig.savefig(filename)

    return


def graphPhaseDiagrams(phi1, phi2, w1, w2, filename="Phase Diagram"):
    """
    This function displays the phase diagrams phi1 vs w1, phi2 vs w2 and
    saves them to files named filename_phi1, etc.
    The graphs will be dispayed in degrees and m/s
    phi1, phi2, w1, and w2 represent the angle of the upper pendulum, and angle of the lower angle, 
    upper pendulum angular velocity, and lower pendulum angular velocity, respectively.
    
    """
    
    filename1 = filename + "_phi1"
    filename2 = filename + "_phi2"
    
    fig1 = plt.figure(figsize=(10,3))
    ax1 = fig1.add_subplot(121)
    ax1.plot(np.rad2deg(phi1), w1)
    plt.xlabel('$\Phi_{1}(t)$', fontsize=12)
    plt.ylabel('$\omega_{1}(t)$', fontsize=12)
    plt.gcf().subplots_adjust(bottom=0.17)
    fig1.savefig(filename1)

    fig2 = plt.figure(figsize=(10,3))
    ax2 = fig2.add_subplot(122)
    ax2.plot(np.rad2deg(phi2), w2)
    plt.xlabel('$\Phi_{2}(t)$', fontsize=12)
    plt.ylabel('$\omega_{2}(t)$', fontsize=12)
    plt.gcf().subplots_adjust(bottom=0.17)
    fig2.savefig(filename2)
    
    return


def graphPoincareSlice(phi1,phi2, w1, w2, t, h, filename="Poincare Slice"):
    """
    This function displays the Poincare Slices phi1 vs w1, phi2 vs w2 and
    saves them to files named filename_phi1, etc.
    The graphs will be dispayed in degrees and m/s
    phi1, phi2, w1, w2, t, and h, represent the angle of the upper pendulum, and angle of the lower angle, 
    upper pendulum angular velocity, and lower pendulum angular velocity, time, and time step size, respectively.
    
    """
    filename1 = filename + "_phi1"
    filename2 = filename + "_phi2"
    
    freq1 = getFrequency(phi1, t, h)
    freq2 = getFrequency(phi2, t, h)
    
    period1 = 1/freq1
    period2 = 1/freq2
    
    #this is the period in terms of the aray points
    period1Adjusted = int(period1/h)
    period2Adjusted = int(period2/h)
    
    poincarePhi1 = phi1[::period1Adjusted]
    poincarePhi2 = phi2[::period2Adjusted]
    poincareW1 = w1[::period1Adjusted]
    poincareW2 = w2[::period2Adjusted]

    fig1 = plt.figure(figsize=(10,3))
    ax1 = fig1.add_subplot(121)
    ax1.plot(np.rad2deg(poincarePhi1), poincareW1, 'r.', ms=5, color='black')
    plt.xlabel('$\Phi_{1}(t)$', fontsize=12)
    plt.ylabel('$\omega_{1}(t)$', fontsize=12)
    plt.gcf().subplots_adjust(bottom=0.17)
    fig1.savefig(filename1)

    fig2 = plt.figure(figsize=(10,3))
    ax2 = fig2.add_subplot(122)
    ax2.plot(np.rad2deg(poincarePhi2), poincareW2, 'r.', ms=5, color='black')
    plt.xlabel('$\Phi_{2}(t)$', fontsize=12)
    plt.ylabel('$\omega_{2}(t)$', fontsize=12)
    plt.gcf().subplots_adjust(bottom=0.17)
    fig2.savefig(filename2)
    
    return

    
Phimax = 3.14
Phisteps = 0.01
PhiPoints = int(Phimax/Phisteps) + 1
tmax = 50
tsteps = 0.01
tpoints = int(tmax/tsteps)
massmax = 10
massmin = 1
masssteps = 0.01
masspoints = int((massmax - massmin)/masssteps)

massPointsArray = np.arange(massmin, massmax, masssteps)
Mass = np.zeros([masspoints, tpoints])
PhiPointsArray = np.arange(0, Phimax, Phisteps)
Phi = np.zeros([PhiPoints, tpoints])

for i, phi0 in enumerate(PhiPointsArray):
    Phi[i] = pd.rk4DobPendBifurcation(0.5, 0.5, 0.5, 0.5, phi0, 0, 0, 0, tsteps, tmax)
    
def bifurcationMass(tmax=50, tsteps=0.01, massMax=10, massMin=1, massSteps=0.1, filename="bifurcationMass"):
    
    tpoints = int(tmax/tsteps)
    masspoints = int((massMax-massMin)/massSteps)
    
    massPointsArray = np.arange(massMin, massMax, massSteps)
    Mass = np.zeros([masspoints, tpoints])
    
    for i, mass in enumerate(massPointsArray):
        Mass[i] = pd.rk4DobPend(mass, 0.5, 0.5, 0.5, 0.5, 0, 0, 0, tsteps, tmax, purpose="bifurcation mass")
        
    plt.plot(massPointsArray[::2], np.abs(Mass[::2, 4500::]),'r.', ms=1, alpha=.1)
    plt.set_xlabel("mass of top pendulum")
    plt.set_xlabel("angular velocity of lower pendulum")
    plt.savefig(filename)
    plt.show()
    
    return

def bifurcationPhi(tmax=50, tsteps=0.01, phiMax=10, phiMin=1, massSteps=0.1, filename="bifurcationAngle"):
    
    tpoints = int(tmax/tsteps)
    phipoints = int((phiMax-phiMin)/phiSteps)
    
    phiPointsArray = np.arange(phiMin, phiMax, phiSteps)
    Phi = np.zeros([phipoints, tpoints])
    
    for i, phi0 in enumerate(massPointsArray):
        Phi[i] = pd.rk4DobPend(0.5, 0.5, 0.5, 0.5, phi0, 0, 0, 0, tsteps, tmax, purpose="bifurcation angle")
        
    plt.plot(phiPointsArray[::2], np.abs(Mass[::2, 4500::]),'r.', ms=1, alpha=.1)
    plt.set_xlabel("Initial angle of upper pendulum")
    plt.set_xlabel("angular velocity of lower pendulum")
    plt.savefig(filename)
    plt.show()
    
    return


    
    
    