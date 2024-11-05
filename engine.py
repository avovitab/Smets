import numpy as np
import misc as mi

def phiSuperbee(r):
        # return max(0.,min(1,r))
        return max(0., min(2.*r,1.), min(r,2.))

def phiVanLeer(r):
    R = abs(r)
    return (r + R)/(1+R)

def phiMinMod(r):
    return max(0,min(1.,r))

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

    def fillFlux0_KT(self, w, f, phi):
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

            uL = w[j-1] + .5 * phi(riL) * (w[j] - w[j-1])
            uR = w[j] - .5 * phi(riR) * (w[j+1] - w[j])

            rho = max(abs(self.jac(uR)),abs(self.jac(uL)))

            flux[j] = .5 * (f(uL) + f(uR) - rho * (uR - uL))

        flux = mi.fillGhosts(flux)
        return flux
    
    def fillFlux1_KT(self, w, f, phi):
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

            uL = w[j] + .5 * phi(riL) * (w[j+1] - w[j])
            uR = w[j+1] - .5 * phi(riR) * (w[j+2] - w[j+1])

            rho = max(abs(self.jac(uR)),abs(self.jac(uL)))

            flux[j] = .5 * (f(uL) + f(uR) - rho * (uR - uL))

        flux = mi.fillGhosts(flux)
        return flux
    
    def fillFlux0_KT_parabolic(self, w, f, phi):
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

            k = 1/3

            uL = w[j-1] + .25 * phi(riL) * ((1-k)*(w[j-1]-w[j-2]) + (1+k)*(w[j] - w[j-1]))
            uR = w[j] - .25 * phi(riR) * ((1-k)*(w[j+1]-w[j]) + (1+k)*(w[j] - w[j-1]))

            rho = max(abs(self.jac(uR)),abs(self.jac(uL)))

            flux[j] = .5 * (f(uL) + f(uR) - rho * (uR - uL))

        flux = mi.fillGhosts(flux)
        return flux
    
    def fillFlux1_KT_parabolic(self, w, f, phi):
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

            k = 1/3

            uL = w[j] + .25 * phi(riL) * ((1-k)*(w[j]-w[j-1]) + (1+k)*(w[j+1] - w[j]))
            uR = w[j+1] - .25 * phi(riR) * ((1-k)*(w[j+2]-w[j+1]) + (1+k)*(w[j+1] - w[j]))

            rho = max(abs(self.jac(uR)),abs(self.jac(uL)))

            flux[j] = .5 * (f(uL) + f(uR) - rho * (uR - uL))

        flux = mi.fillGhosts(flux)
        return flux
    
    def fillDamp(self,w):
        N = w.shape[0]
        damp = np.empty(N)
        for j in range(2,N-2):
            damp[j] = (w[j-1]-2*w[j]+w[j+1])

        return damp

    def fillFlux0(self, w, f, phi):
        if self.form == 'KT':
            return self.fillFlux0_KT(w, f, phi)
        if self.form == 'KTparabolic':
            return self.fillFlux0_KT_parabolic(w, f, phi)

    def fillFlux1(self, w, f, phi):
        if self.form == 'KT':
            return self.fillFlux1_KT(w, f, phi)
        if self.form == 'KTparabolic':
            return self.fillFlux1_KT_parabolic(w, f, phi)


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
        
        u1w = np.empty((u0w.shape[0]))
        F0w = np.empty((u0w.shape[0]))
        F1w = np.empty((u0w.shape[0]))
        Dw = np.empty((u0w.shape[0]))

        for _ in range(Nt):
            F0w = self.fillFlux0(u0w, self.flux, phiMinMod)
            F1w = self.fillFlux1(u0w, self.flux, phiMinMod)
            Dw = self.fillDamp(u0w)

            u1w[2:-2] = u0w[2:-2] - self.nu * (F1w[2:-2] - F0w[2:-2]) # + 0.1*self.dt/self.dx/self.dx*Dw[2:-2]

            u1w = mi.fillGhosts(u1w,num_of_ghosts=2)

            u0w = u1w

        self.uF = u1w[2:-2]
