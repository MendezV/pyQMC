import numpy as np
from scipy.linalg import circulant
from scipy.linalg import eigh
import sys
class AuxField:
    def __init__(self,Ntau,Nsites):
        self.Nsites=Nsites
        self.Ntau=Ntau
        self.valfield=[-1,1]
        self.fields=self.init_fields()
        
    def init_fields(self):
        return np.random.choice(self.valfield,[self.Ntau,self.Nsites])

    def flipfield(self, tau, site):
        # the structure will be of Ntau x Nsites, where Ntau are rows
        self.fields[tau,site]=-self.fields[tau,site]
        return None



def main()->int:
    A=AuxField(3,4)
    print(A.fields)
    A.flipfield(2,0)
    print(A.fields)
    print(A.fields[0])
    return 0

if __name__=='__main__':
    sys.exit(main())