import casadi as ca
from NumOpt import Opti
import numpy as np


class Section:
    def __init__(self, af, theta, r, chord):
        self.af = af
        self.theta = theta
        self.r = r
        self.chord = chord

    def set_rotor_parameters(self, Nb, Rhub, Rtip):
        self.Nb = Nb
        self.Rhub = Rhub
        self.Rtip = Rtip
        self.solidity = self.Nb * self.chord / (2 * np.pi * self.r)

    def prandtl(self, phi):
        phi_sin = ca.sin(phi)

        ftip = self.Nb / 2.0 * (self.Rtip - self.r) / (self.r * phi_sin)
        Ftip = 2.0 / np.pi * ca.arccos(ca.exp(-((ftip**2) ** 0.5)))

        fhub = self.Nb / 2.0 * (self.r - self.Rhub) / (self.Rhub * phi_sin)
        Fhub = 2.0 / np.pi * np.arccos(np.exp(-((fhub**2) ** 0.5)))
        F = Ftip * Fhub
        return F

    def induction_factors(self, phi, Cn, Ct, F=1.0):
        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)
        k = Cn * self.solidity / (4 * F * phi_sin**2)
        kp = Ct * self.solidity / (4 * F * phi_sin * phi_cos)

        a = k / (1 - k)
        ap = kp / (1 + kp)

        return a, ap

    def residual(self, phi, Vx, Vy, rho, mu, sos):
        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)
        alpha = self.theta - phi
        alpha_deg = alpha / np.pi * 180.0

        W0 = np.sqrt(Vx**2 + Vy**2)
        Re = rho * W0 * self.chord / mu
        Mach = W0 / sos

        CL, CD = self.af(alpha_deg, Re, Mach)

        Cn = CL * phi_cos - CD * phi_sin
        Ct = CL * phi_sin + CD * phi_cos

        F = self.prandtl(phi)
        # k = Cn * self.solidity / (4 * F * phi_sin ** 2)
        # kp = Ct * self.solidity / (4 * F * phi_sin * phi_cos)

        # a=k / (1 - k)
        # ap=kp / (1 + kp)

        # R = phi_sin / (1 + a) - Vx / Vy * phi_cos / (1 - ap)
        R = Vy * (4 * F * phi_sin**2 - Cn * self.solidity) - Vx * (4 * F * phi_sin * phi_cos + Ct * self.solidity)

        return R
    



class Blade:
    def __init__(self, Rtip, Rhub, Nb, chord_distribution,twist_distribution,airfoilmodel,pitch,radius_resolution):
        self.Rtip = Rtip
        self.Rhub = Rhub
        self.Nb = Nb
        self.chord_distribution=chord_distribution 
        self.twist_distribution=twist_distribution 
        self.pitch=pitch 
        self.radius_resolution=radius_resolution
        if isinstance(self.radius_resolution,int):
            self.r_list=np.linspace(self.Rhub,self.Rtip,self.radius_resolution)
        elif isinstance(self.radius_resolution,np.ndarray):
            self.r_list=np.clip(self.radius_resolution,self.Rhub,self.Rtip)
        else:
            raise ValueError("the `radius_resolution` must be `np.ndarray` or `int`. ")

        self.chord_list=self.chord_distribution(self.r_list)
        self.twist_list=self.twist_distribution(self.r_list)

        self.airfoilmodel=airfoilmodel 

    def __gen_redidual_func():
        phi=ca.MX.sym("phi")
        theta=ca.MX.sym("theta")
        Vx=ca.MX.sym("Vx")
        omega=ca.MX.sym("omega")
        rho=ca.MX.sym("rho")
        chord=ca.MX.sym("chord")
        mu=ca.MX.sym("mu")
        sos=ca.MX.sym("sos")
        r=ca.MX.sym("r")
        Nb=ca.MX.sym("Nb")
        Rtip=ca.MX.sym("Rtip")
        Rhub=ca.MX.sym("Rhub")

        
        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)
        alpha = theta - phi
        alpha_deg = alpha / np.pi * 180.0

        Vy=omega*r
        W0 = np.sqrt(Vx**2 + Vy**2)
        Re = rho * W0 * chord / mu
        Mach = W0 / sos

        ftip = Nb / 2.0 * (Rtip - r) / (r * phi_sin)
        Ftip = 2.0 / np.pi * ca.arccos(ca.exp(-((ftip**2) ** 0.5)))
        fhub = Nb / 2.0 * (r - Rhub) / (Rhub * phi_sin)
        Fhub = 2.0 / np.pi * np.arccos(np.exp(-((fhub**2) ** 0.5)))
        F = Ftip * Fhub




    def solve(self, Vx, omega, rho=1.225, mu=1.81e-5, sos=340.0):
        ...

