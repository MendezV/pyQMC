import numpy as np
import matplotlib.pyplot as plt
def f(E,T):
    return 1/(np.exp(E/T)+1)

def ek(k, mu):
    t=-1
    return 2*t*np.cos(k)-mu

N=6
k=2*np.arange(-N//2,N//2)*np.pi/N
print(np.size(k))
E=ek(k,0)
beta=4
T=1/beta
gs=2
print("total energy", gs*np.sum(E*f(E, T)))
print("electrons per unit cell", gs*np.sum(f(E, T)) /N)
plt.plot(k, E)
plt.show()