#6
import numpy as np
import Obs_eq

class Simulation:
    
    def __init__(self, Geq_pl,Geq_min,Nsweeps ):
        self.Geq_pl=Geq_pl
        self.Geq_min=Geq_min
        self.Nsweeps=Nsweeps
        self.Ntau=self.Geq_pl.stringops.gamma.Ntau
        self.Nsites=self.Geq_pl.stringops.gamma.Nsites

    def MC_step(self,Obs_store_sweep):
        for _ in range(self.Nsweeps):
            Obs_store=[]
            for tau in range(self.Ntau):
                if tau%self.Geq_pl.stringops.Nwrap==0:
                    self.Geq_pl.Glist[tau]=self.Geq_pl.construct_G_tau(tau)
                    self.Geq_min.Glist[tau]=self.Geq_min.construct_G_tau(tau)
                for site in range(self.Nsites):
                    [R_pl, Delta_pl]=self.Geq_pl.Ratio( tau, site )
                    [R_min, Delta_min]=self.Geq_min.Ratio( tau, site )
                    R=R_pl*R_min
                    xi=np.random.random()
                    if R>xi:
                        self.Geq_pl.update_G_tau(tau,site, R_pl,Delta_pl) #updating Green's functions
                        self.Geq_min.update_G_tau(tau,site, R_min,Delta_min) #updating Green's functions
                        self.Geq_pl.stringops.flip_field_update_B( tau, site) #updating B's
                        self.Geq_min.stringops.flip_field_update_B( tau, site) #updating B's
                        
                indtau=int((tau+1)%self.Ntau)
                self.Geq_pl.Glist[indtau]=self.Geq_pl.AdvanceG(tau, 1, self.Geq_pl.Glist[tau])
                self.Geq_min.Glist[indtau]=self.Geq_min.AdvanceG(tau, 1, self.Geq_min.Glist[tau])
                Obs_store.append(Obs_eq.Measure_Etot(self.Geq_pl, self.Geq_min,indtau))
                
            Obs_store_sweep.append(np.sum(Obs_store, axis=0)/self.Ntau)
            
        return None
       