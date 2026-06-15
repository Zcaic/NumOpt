import numpy as np
import casadi as ca
from dataclasses import dataclass

# import aerosandbox.numpy as anp
# import aerosandbox as asb
from NumOpt import Opti

# from scipy.optimize import root_scalar

# from scipy.interpolate import interp1d


class FileAirfoil:
    def __init__(self, csvfile):
        self.air_data = np.loadtxt(csvfile, ndmin=2)
        self.CL = ca.interpolant("CL", "linear", [self.air_data[:, 0]], self.air_data[:, 1])
        # self.CL = interp1d(
        #     self.air_data[:, 0],
        #     self.air_data[:, 1],
        #     kind="slinear",
        #     fill_value=(self.air_data[0, 1], self.air_data[-1, 1]),
        #     bounds_error=False,
        # )
        self.CD = ca.interpolant("CD", "linear", [self.air_data[:, 0]], self.air_data[:, 2])
        # self.CD = interp1d(
        #     self.air_data[:, 0],
        #     self.air_data[:, 2],
        #     kind="slinear",
        #     fill_value=(self.air_data[0, 2], self.air_data[-1, 2]),
        #     bounds_error=False,
        # )

    def __call__(self, alpha):
        """
        alpha: deg

        """
        cl = self.CL(alpha)
        cd = self.CD(alpha)
        return cl, cd


@dataclass
class SectionAero:
    Tn: float = 0.0
    Tt: float = 0.0
    Q: float = 0.0
    phi: float = 0.0
    alpha: float = 0.0
    W: float = 0.0
    CL: float = 0.0
    CD: float = 0.0
    Cn: float = 0.0
    Ct: float = 0.0


class Section:
    def __init__(self, af, theta, r, b):
        self.af = af
        self.theta = theta
        self.r = r
        # self.Nb = None
        self.b = b
        # self.Rhub = None
        # self.Rtip = None
        self.eps = 1e-6
        # self.solidity = None

        self.quadrants = np.array(
            [
                [self.eps, np.pi-self.eps],
                [-np.pi+self.eps, -self.eps]
            ]
        )

    def set_rotor_parameters(self, Nb, Rhub, Rtip):
        self.Nb = Nb
        self.Rhub = Rhub
        self.Rtip = Rtip
        # if np.abs(self.r - self.Rhub) <= self.eps:
        #     self.r = self.Rhub + self.eps
        # elif np.abs(self.r - self.Rtip) <= self.eps:
        #     self.r = self.Rtip - self.eps
        self.solidity = self.Nb * self.b / (2 * np.pi * self.r)

    def _residual(self, phi, Vx, Vy):
        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)
        alpha = self.theta - phi
        alpha_deg = alpha / np.pi * 180.0
        CL, CD = self.af(alpha_deg)

        Cn = CL * phi_cos - CD * phi_sin
        Ct = CL * phi_sin + CD * phi_cos

        # F = 1.0
        F=self.prandtl(phi)
        k = Cn * self.solidity / (4 * F * ca.sin(phi) ** 2)
        kp = Ct * self.solidity / (4 * F * ca.sin(phi) * ca.cos(phi))

        a = k / (1 - k)
        ap = kp / (1 + kp)

        R = ca.sin(phi) / (1 + a) - Vx / Vy * ca.cos(phi) / (1 - ap)
        return R, a, ap, Cn, Ct, alpha, CL, CD

    def prandtl(self, phi):
        phi_sin = ca.fabs(ca.sin(phi))

        ftip = self.Nb / 2.0 * (self.Rtip - self.r) / (self.r * phi_sin)
        Ftip = 2.0 / np.pi * ca.arccos(ca.exp(-ftip))

        fhub = self.Nb / 2.0 * (self.r - self.Rhub) / (self.Rhub * phi_sin)
        Fhub = 2.0 / np.pi * ca.arccos(ca.exp(-fhub))
        F = Ftip * Fhub
        return F

    def find_root(self, V0, omega, rho):
        if np.fabs(self.r-self.Rhub)<=self.eps or np.fabs(self.r-self.Rtip)<=self.eps:
            return SectionAero()

        Vx = V0
        Vy = omega * self.r

        Vx_equal_zero = np.abs(Vx) <= self.eps
        Vy_equal_zero = np.abs(Vy) <= self.eps

        if Vx_equal_zero and Vy_equal_zero:
            return SectionAero()
        # elif Vx_equal_zero:
        #     ...
        # elif Vy_equal_zero:
        #     ...
        else:
            order = self.quadrants[[0, 1]]
            for i in order:
                phi_min, phi_max = i
                opti = Opti()
                phi = opti.variable(init_guess=phi_min, lower_bound=phi_min, upper_bound=phi_max)
                residual, a, ap, Cn, Ct, alpha, CL, CD = self._residual(phi, Vx, Vy)
                # opti.minimize(residual**2)
                opti.subject_to(residual==0.0)
                opti.solver()
                try:
                    sol = opti.solve()
                    phi = sol(phi)
                    a = sol(a)
                    ap = sol(ap)
                    Cn = sol(Cn)
                    Ct = sol(Ct)
                    alpha = sol(alpha)
                    CL = sol(CL)
                    CD = sol(CD)
                    u = a * Vx
                    v = ap * Vy

                    W2 = (Vx + u) ** 2 + (Vy - v) ** 2
                    dT = Cn * 0.5 * rho * W2 * self.b
                    dF = Ct * 0.5 * rho * W2 * self.b
                    dQ = dF * self.r
                    return SectionAero(Tn=dT, Tt=dF, Q=dQ, phi=phi, alpha=alpha, W=np.sqrt(W2), CL=CL, CD=CD, Cn=Cn, Ct=Ct)

                except:
                    continue


class Blade:
    def __init__(self, Rtip, Rhub, Nb, sections: list[Section]):
        self.Rtip = Rtip
        self.Rhub = Rhub
        self.Nb = Nb
        self.sections = sections

        for i in sections:
            i.set_rotor_parameters(self.Nb,self.Rhub,self.Rtip)

    def solve(self, V0, omega, rho):
        rs = []
        dTs = []
        dFs = []
        dQs = []
        for i in self.sections:
            sec_aero = i.find_root(V0, omega, rho)
            dTs.append(np.array(sec_aero.Tn))
            dFs.append(np.array(sec_aero.Tt))
            dQs.append(np.array(sec_aero.Q))
            rs.append(i.r)
        T = self.Nb * np.trapezoid(y=dTs, x=rs)
        F = self.Nb * np.trapezoid(y=dFs, x=rs)
        M = self.Nb * np.trapezoid(y=dQs, x=rs)

        D = self.Rtip * 2
        n = omega / (2 * np.pi)
        CT = T / (rho * n**2 * D**4)
        CQ = M / (rho * n**2 * D**5)
        CP = 2 * np.pi * CQ

        J = V0 / (n * D)
        if J == 0:
            eta = CT / CP
        else:
            eta = (CT / CP) * J
        return {"T": T, "M": M, "CT": CT, "CQ": CQ, "CP": CP, "eta": eta}


def test01():
    import matplotlib.pyplot as plt

    af = FileAirfoil("./CLARKY.csv")
    alpha = np.linspace(-180.0, 180.0, 100)
    CL, CD = af(alpha)
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121)
    ax.plot(af.air_data[:, 0], af.air_data[:, 1], label="CL")
    ax.plot(alpha, CL, "o", markersize=1, label="CL_interp")
    ax = fig.add_subplot(122)
    ax.plot(af.air_data[:, 0], af.air_data[:, 2], label="CD")
    ax.plot(alpha, CD, "o", markersize=1, label="CD_interp")
    ax.set_ylim(bottom=-0.50,top=1.30)

    plt.legend()
    plt.show()


def test02():
    import matplotlib.pyplot as plt

    Nb = 3
    Rtip = 3.054 / 2.0
    Rhub = 0.375
    rs = np.array([Rhub, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip])
    chords = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12])
    pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))
    # rs = np.array([ 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425])
    # chords = np.array([0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12])
    # pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))

    secs = [Section(af=FileAirfoil("./CLARKY.csv"), theta=theta, r=r, b=b) for theta, r, b in zip(pitchs, rs, chords)]
    blade = Blade(Rhub=Rhub, Rtip=Rtip, Nb=Nb, sections=secs)

    ret_list = []
    vinf_list = np.linspace(1.0, 44.0, 20)
    for i in vinf_list:
        ret = blade.solve(V0=i, omega=1100 * 2 * np.pi / 60.0, rho=1.225)
        ret_list.append(ret)

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(121)
    ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), [i["CT"] for i in ret_list], label="My Bemt")
    print([i["CT"] for i in ret_list])

    exp_data = np.loadtxt("./propeller_dat.csv", skiprows=1, ndmin=2)
    ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
    ax.set_xlabel("J")
    ax.set_ylabel("CT")
    ax.legend()

    ax = fig.add_subplot(122)
    ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), [i["CP"] for i in ret_list], label="My Bemt")
    print([i["CT"] for i in ret_list])

    exp_data = np.loadtxt("./propeller_dat.csv", skiprows=1, ndmin=2)
    ax.plot(exp_data[:, 0], exp_data[:, 2], label="exp")
    ax.set_xlabel("J")
    ax.set_ylabel("CP")
    ax.set_ylim(bottom=0.0,top=0.14)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    # test01()
    test02()
