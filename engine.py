# -*- coding: utf-8 -*-


#created by Matthieu Couturier and Esther Annézo--Sébire
# Different Muscl scheme
# ============================================

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

def phivanAlbada(r):
    return 2*r/(1+r**2)

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

            uL = w[j] + .25 * phi(riL) * ((1-k)*(w[j]-w[j-1]) + (1+k)*(w[j+1] - w[j])) #use flux limiter between two methods?
            uR = w[j+1] - .25 * phi(riR) * ((1-k)*(w[j+2]-w[j+1]) + (1+k)*(w[j+1] - w[j]))

            rho = max(abs(self.jac(uR)),abs(self.jac(uL)))

            flux[j] = .5 * (f(uL) + f(uR) - rho * (uR - uL))

        flux = mi.fillGhosts(flux)
        return flux
    
    def fillFlux1_MUSCL_Laney(self, w, f, phi):
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

            uL = w[j] + 0.5 * phi(riL) * (w[j+1] - w[j])
            uR = w[j+1] - 0.5 * phi(riR) * (w[j+2] - w[j+1])

            fL = f(w[j]) + 0.5 * self.a(w[j]) * (uL - w[j])
            fR = f(w[j+1]) - 0.5 * self.a(w[j+1]) * (w[j+1] - uR)

            aL = self.jac(uL)
            aR = self.jac(uR)

            a = max(abs(aL), abs(aR))

            flux[j] = 0.5 * (fL + fR) - 0.5 * a * (uR - uL)
            flux[j] += 0.5 * self.nu * (uR - uL)

        flux = mi.fillGhosts(flux)
        return flux
    
    def fillDamp(self,w):
        N = w.shape[0]
        damp = np.empty(N)
        for j in range(2,N-2):
            damp[j] = (w[j-1]-2*w[j]+w[j+1]) # w[j]-w[j-1] no? if taking 2.48 expression

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
            F0w = self.fillFlux0(u0w, self.flux, phiSuperbee)
            F1w = self.fillFlux1(u0w, self.flux, phiSuperbee)
            Dw = self.fillDamp(u0w)

            k1 = -self.nu * (F1w[2:-2] - F0w[2:-2]) #RK4
            u1 = u0w[2:-2] + 0.5 * k1
            u1_w = mi.addGhosts(u1, num_of_ghosts=2)
            u1_w = mi.fillGhosts(u1_w, num_of_ghosts=2)
            
            F0w_1 = self.fillFlux0(u1_w, self.flux, phiMinMod)
            F1w_1 = self.fillFlux1(u1_w, self.flux, phiMinMod)
            k2 = -self.nu * (F1w_1[2:-2] - F0w_1[2:-2])
            u2 = u0w[2:-2] + 0.5 * k2
            u2_w = mi.addGhosts(u2, num_of_ghosts=2)
            u2_w = mi.fillGhosts(u2_w, num_of_ghosts=2)
            
            F0w_2 = self.fillFlux0(u2_w, self.flux, phiMinMod)
            F1w_2 = self.fillFlux1(u2_w, self.flux, phiMinMod)
            k3 = -self.nu * (F1w_2[2:-2] - F0w_2[2:-2])
            u3 = u0w[2:-2] + k3
            u3_w = mi.addGhosts(u3, num_of_ghosts=2)
            u3_w = mi.fillGhosts(u3_w, num_of_ghosts=2)
            
            F0w_3 = self.fillFlux0(u3_w, self.flux, phiMinMod)
            F1w_3 = self.fillFlux1(u3_w, self.flux, phiMinMod)
            k4 = -self.nu * (F1w_3[2:-2] - F0w_3[2:-2])
            
            u1w[2:-2] = u0w[2:-2] + (k1 + 2*k2 + 2*k3 + k4) / 6

            #u1w[2:-2] = u0w[2:-2] - self.nu * (F1w[2:-2] - F0w[2:-2]) # + 0.1*self.dt/self.dx/self.dx*Dw[2:-2]

            # u_half = u0w[2:-2] - 0.5 * self.nu * (F1w[2:-2] - F0w[2:-2]) #RK2
            # u_half_w = mi.addGhosts(u_half, num_of_ghosts=2)
            # u_half_w = mi.fillGhosts(u_half_w, num_of_ghosts=2)
            
            # F0w_half = self.fillFlux0(u_half_w, self.flux, phiMinMod)
            # F1w_half = self.fillFlux1(u_half_w, self.flux, phiMinMod)
            
            # u1w[2:-2] = u0w[2:-2] - self.nu * (F1w_half[2:-2] - F0w_half[2:-2])

            u1w = mi.fillGhosts(u1w,num_of_ghosts=2)

            u0w = u1w

        self.uF = u1w[2:-2]
