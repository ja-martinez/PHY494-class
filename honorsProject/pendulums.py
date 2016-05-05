import numpy as np

# This text files contains the functions for the actual simulation of the pendulums

### function for Simple Pendulum ###
def simplePendulum(m, L, theta0, w0, h, t_max, g=9.81):
    
    def f1(t, theta, w):
            return w

    def f2(t, theta, w):
            return -g/L*np.sin(theta)

    Nsteps = t_max/h
    t_range = h*np.arange(Nsteps, dtype=np.float64)
    theta = np.zeros_like(t_range)
    w = np.zeros_like(t_range)
    y = np.array([theta0, w0], dtype=np.float64)
    
    A0 = np.zeros(2)
    A1 = np.zeros(2)
    A2 = np.zeros(2)
    A3 = np.zeros(2)
    
    for i, t in enumerate(t_range):
        theta[i] = y[0]
        w[i] = y[1]

        k0 = h*f1(t, y[0], y[1])
        d0 = h*f2(t, y[0], y[1])
        A0[:] = k0, d0

        k1 = h*f1(t + 0.5*h, y[0] + 0.5*k0, y[1] + 0.5*d0)
        d1 = h*f2(t + 0.5*h, y[0] + 0.5*k0, y[1] + 0.5*d0)
        A1[:] = k1, d1

        k2 = h*f1(t + 0.5*h, y[0] + 0.5*k1, y[1] + 0.5*d1)
        d2 = h*f2(t + 0.5*h, y[0] + 0.5*k1, y[1] + 0.5*d1)
        A2[:] = k2, d2

        k3 = h*f1(t + h, y[0] + k2, y[1] + d2)
        d3 = h*f2(t + h, y[0] + k2, y[1] + d2)
        A3[:] = k3, d3

        y[:] = y + (A0 + 2*A1 + 2*A2 + A3)/6
    return t_range, theta
    
    
    
### Function for double pendulum ###   
def rk4DobPend(m1, m2, L1, L2, phi10, phi20, w10, w20, h, t_max, g=9.81, linearized=False, purpose="all"):
    """
    m1, L1, phi10, w10 correspond to the mass length, initial angle, and initial angular velocity of the top pendulum
    m2, L2, phi20, w20 correspond to the same, but for the bottom pendulum
    h corresponds to the time step
    t_max corresponds to the greatest amout of time needed to calculate
    linearized parameter determines use of linearized or nonlinearized solution. type=bool
    purpose parameter  determines what variables are outputed. 
        Options for purpose include "all", "comparison", "bifurcation mass", and "bifurcation angle"
    """
    
    phi10 = np.deg2rad(phi10)
    phi20 = np.deg2rad(phi10)
    
    def f1(t, phi1, phi2, w1, w2):
        return w1
    
    def g1(t, phi1, phi2, w1, w2, linearized):
        """Equations for phi1''=w1'"""
        if linearized:
            return (m2*g)*phi2/(m1*L1) - (m1+m2)*g*phi1/(m1*L1)
        
        return (-g*(2*m1+m2)*np.sin(phi1)-m2*g*np.sin(phi1-2*phi2)-2*np.sin(phi1-phi2)*m2*((w2**2)*L2+(w1**2)*L1*np.cos(phi1-phi2))) \
                /(L1*(2*m1+m2-m2*np.cos(2*phi1-2*phi2)))
    
    def f2(t, phi1, phi2, w1, w2):
        return w2
    
    def g2(t, phi1, phi2, w1, w2, linearized):
        """Equations for phi1''=w1'  """
        if linearized:
            return -(m2/m1+1)*g*phi2/L2 + (m1+m2)*g*phi1/(m1*L2)
        
        return (2*np.sin(phi1-phi2)*((w1**2)*L1*(m1+m2)+g*(m1+m2)*np.cos(phi1)+(w2**2)*L2*m2*np.cos(phi1-phi2))) \
            /(L2*(2*m1+m2-m2*np.cos(2*phi1-2*phi2)))

    
    Nsteps = t_max/h
    t_range = h*np.arange(Nsteps, dtype=np.float64)
    phi1 = np.zeros_like(t_range)
    phi2 = np.zeros_like(t_range)
    w1 = np.zeros_like(t_range)
    w2 = np.zeros_like(t_range)
    y = np.array([phi10, phi20, w10, w20], dtype=np.float64)
    
    A0 = np.zeros(4)
    A1 = np.zeros(4)
    A2 = np.zeros(4)
    A3 = np.zeros(4)
    
    for i, t in enumerate(t_range):
        phi1[i] = y[0]
        phi2[i] = y[1]
        w1[i] = y[2]
        w2[i] = y[3]
        
        k0 = h*f1(t, y[0], y[1], y[2], y[3])
        d0 = h*f2(t, y[0], y[1], y[2], y[3])
        l0 = h*g1(t, y[0], y[1], y[2], y[3], linearized=linearized)
        q0 = h*g2(t, y[0], y[1], y[2], y[3], linearized=linearized)
        A0[:] = k0, d0, l0, q0
        
        k1 = h*f1(t + 0.5*h, y[0] + 0.5*k0, y[1] + 0.5*d0, y[2] + 0.5*l0, y[3] + 0.5*q0)
        d1 = h*f2(t + 0.5*h, y[0] + 0.5*k0, y[1] + 0.5*d0, y[2] + 0.5*l0, y[3] + 0.5*q0)
        l1 = h*g1(t + 0.5*h, y[0] + 0.5*k0, y[1] + 0.5*d0, y[2] + 0.5*l0, y[3] + 0.5*q0, linearized=linearized)
        q1 = h*g2(t + 0.5*h, y[0] + 0.5*k0, y[1] + 0.5*d0, y[2] + 0.5*l0, y[3] + 0.5*q0, linearized=linearized)
        A1[:] = k1, d1, l1, q1
        
        k2 = h*f1(t + 0.5*h, y[0] + 0.5*k1, y[1] + 0.5*d1, y[2] + 0.5*l1, y[3] + 0.5*q1)
        d2 = h*f2(t + 0.5*h, y[0] + 0.5*k1, y[1] + 0.5*d1, y[2] + 0.5*l1, y[3] + 0.5*q1)
        l2 = h*g1(t + 0.5*h, y[0] + 0.5*k1, y[1] + 0.5*d1, y[2] + 0.5*l1, y[3] + 0.5*q1, linearized=linearized)
        q2 = h*g2(t + 0.5*h, y[0] + 0.5*k1, y[1] + 0.5*d1, y[2] + 0.5*l1, y[3] + 0.5*q1, linearized=linearized)
        A2[:] = k2, d2, l2, q2
        
        k3 = h*f1(t + h, y[0] + k2, y[1] + d2, y[2] + l2, y[3] + q2)
        d3 = h*f2(t + h, y[0] + k2, y[1] + d2, y[2] + l2, y[3] + q2)
        l3 = h*g1(t + h, y[0] + k2, y[1] + d2, y[2] + l2, y[3] + q2, linearized=linearized)
        q3 = h*g2(t + h, y[0] + k2, y[1] + d2, y[2] + l2, y[3] + q2, linearized=linearized)
        A3[:] = k3, d3, l3, q3
        
        y[:] = y + (A0 + 2*A1 + 2*A2 + A3)/6
    
    if (purpose == "all"):
        return t_range, phi1, phi2, w1, w2, L1, L2, m1, m2, g 
    if (purpose == "comparison"):
        return t_range, phi1
    if (purpose == "bifurcation mass"):
        phi1 = np.arctan2(np.sin(phi1), np.cos(phi1))
        phi2 = np.arctan2(np.sin(phi2), np.cos(phi2))
        return np.abs(w2)
    if (puspose == "bifurcation angle"):
        phi1 = np.arctan2(np.sin(phi1), np.cos(phi1))
        phi2 = np.arctan2(np.sin(phi2), np.cos(phi2))
        return np.abs(w2)
    
    
    
    
    
    
    
    
    
    
    
    
    