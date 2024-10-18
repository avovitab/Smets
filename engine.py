import numpy as np
import misc as mi

epsilon = 1e-6

def phiSuperbee(r):
        return max(0, min(2*r,1), min(r,2))

class MUSCL():

    def __init__(self, testCase, form='KT'):
        self.form = form
        self.dx = testCase.dx
        self.dt = testCase.dt
        self.tFinal = testCase.tFinal
        self.nu = testCase.nu
        self.u0 = testCase.u0
        self.flux = testCase.flux
        self.u_star = testCase.u_star
        self.a = testCase.a
        self.jac = self.a
        self.x = testCase.x
        self.uFinal = testCase.uFinal

    def fillFlux0_KT(self, w, f, a):
        N = w.shape[0]
        flux = np.empty(N)

        for j in range(2, N-2):
            if (w[j] - w[j-1]) == 0:
                riL = np.inf
            else:
                riL = (w[j-1] - w[j-2])/(w[j] - w[j-1])
            if (w[j+1] - w[j]) == 0:
                riR = np.inf
            else:
                riR = (w[j] - w[j-1])/(w[j+1] - w[j])


            uL = w[j] + .5 * phiSuperbee(riL) * (w[j] - w[j-1])
            uR = w[j] - .5 * phiSuperbee(riR) * (w[j+1] - w[j])

            rho = max(self.jac(uR),self.jac(uL))

            flux[j] = .5 * (f(uL) + f(uR) - rho * (uR - uL))

        flux = mi.fillGhosts(flux)
        return flux
    
    def fillFlux1_KT(self, w, f, a):
        N = w.shape[0]
        flux = np.empty(N)

        for j in range(2, N-2):
            if (w[j+1] - w[j]) == 0:
                riL = np.inf
            else:
                riL = (w[j] - w[j-1])/(w[j+1] - w[j])
            if (w[j+2] - w[j+1]) == 0:
                riR = np.inf
            else:
                riR  = (w[j+1] - w[j])/(w[j+2] - w[j+1])

            uL = w[j] + .5 * phiSuperbee(riL) * (w[j+1] - w[j])
            uR = w[j] - .5 * phiSuperbee(riR) * (w[j+2] - w[j+1])

            rho = max(self.jac(uR),self.jac(uL))

            flux[j] = .5 * (f(uL) + f(uR) - rho * (uR - uL))

        flux = mi.fillGhosts(flux)
        return flux

    def fillFlux0(self, w, f, a):
        if self.form == 'KT':
            return self.fillFlux0_KT(w, f, a)


    def fillFlux1(self, w, f, a):
        if self.form == 'KT':
            return self.fillFlux1_KT(w, f, a)


    def compute(self, tFinal):
        Nt = int(tFinal/self.dt)
        dx = self.dx

        u0w = mi.addGhosts(self.u0(self.x),num_of_ghosts=2)
        u0w = mi.fillGhosts(u0w,num_of_ghosts=2)

        xw = mi.addGhosts(self.x,num_of_ghosts=2)
        xw[1] = xw[2]-dx
        xw[0] = xw[1]-dx
        xw[-2] = xw[-3]+dx
        xw[-1] = xw[-2]+dx
        
        print(xw.shape)
        print(u0w.shape)
        u1w = np.empty((u0w.shape[0]))
        F0w = np.empty((u0w.shape[0]))
        F1w = np.empty((u0w.shape[0]))

        for _ in range(Nt):
            F0w = self.fillFlux0(u0w, self.flux, self.a)
            F1w = self.fillFlux1(u0w, self.flux, self.a)

            F1w = mi.fillGhosts(F1w,num_of_ghosts=2)

            u1w[2:-2] = u0w[2:-2] - self.nu * (F1w[2:-2] - F0w[2:-2])

            u1w = mi.fillGhosts(u1w,num_of_ghosts=2)

            u0w = u1w

        self.uF = u1w[2:-2]
