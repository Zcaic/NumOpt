import numpy as np

# from scipy.optimize.elementwise import find_root
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from dataclasses import dataclass

# import aerosandbox.numpy as anp
# import aerosandbox as asb


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


@dataclass
class Oper:
    Vx: float = 0.0
    Vy: float = 0.0
    rho: float = 1.225
    mu: float = 1.81e-5
    sos: float = 340.0


def find_bracket(f, args, xmin, xmax, n, backsearch=False):
    x_list = np.linspace(xmin, xmax, n)
    if backsearch:
        x_list = x_list[::-1]
    # y_list = []
    # for i in x_list:
    #     y = f(i, *args)
    #     y_list.append(y)
    # y_list=np.array(y_list).flatten()
    y_list=f(x_list,*args)
    flag = y_list[:-1] * y_list[1:]
    idx = np.where(flag < 0.0)
    if len(idx) == 0:
        return False, None, None
    else:
        return True, (x_list[idx[0]], x_list[idx[0] + 1])


class FileAirfoil:
    def __init__(self, csvfile):
        self.air_data = np.loadtxt(csvfile, ndmin=2)
        self.CL = interp1d(
            self.air_data[:, 0],
            self.air_data[:, 1],
            kind="quadratic",
        )

        self.CD = interp1d(
            self.air_data[:, 0],
            self.air_data[:, 2],
            kind="quadratic",
        )

    def __call__(self, Alpha, Reynold, Mach):
        """
        alpha: deg

        """
        cl = self.CL(Alpha)
        cd = self.CD(Alpha)
        return cl, cd


class Section:
    def __init__(self, af, theta, r, b):
        self.af = af
        self.theta = theta
        self.r = r
        self.b = b
        self.eps = 1e-6

        self.quadrants = np.array([[self.eps, np.pi / 2.0], [-np.pi / 2.0, -self.eps], [np.pi / 2.0, np.pi - self.eps], [-np.pi + self.eps, -np.pi / 2.0]])

    def set_rotor_parameters(self, Nb, Rhub, Rtip):
        self.Nb = Nb
        self.Rhub = Rhub
        self.Rtip = Rtip

        self.solidity = self.Nb * self.b / (2 * np.pi * self.r)

    def residual(self, phi, oper):
        R, a, ap, Cn, Ct, alpha, CL, CD = self._residual(phi, oper)
        return R

    def _residual(self, phi, oper: Oper):
        phi = np.atleast_1d(phi)
        phi_sin = np.sin(phi)
        phi_cos = np.cos(phi)
        alpha = self.theta - phi
        alpha_deg = alpha / np.pi * 180.0

        W0 = np.sqrt(oper.Vx**2 + oper.Vy**2)
        Re = oper.rho * W0 * self.b / oper.mu
        Mach = W0 / oper.sos

        CL, CD = self.af(alpha_deg, Re, Mach)

        Cn = CL * phi_cos - CD * phi_sin
        Ct = CL * phi_sin + CD * phi_cos

        # F = 1.0
        F = self.prandtl(phi)
        k = Cn * self.solidity / (4 * F * np.sin(phi) ** 2)
        kp = Ct * self.solidity / (4 * F * np.sin(phi) * np.cos(phi))

        if np.isclose(oper.Vx, 0.0, atol=self.eps):
            # u=np.sign(phi)*kp*Cn/Ct*oper.Vy
            # v=np.zeros_like(phi)
            a = np.zeros_like(phi)
            ap = np.zeros_like(phi)
            R = np.sign(phi) - k
        elif np.isclose(oper.Vy, 0.0, atol=self.eps):
            # u=np.zeros_like(phi)
            # v=k*Ct/Cn*np.fabs(oper.Vx)
            a = np.zeros_like(phi)
            ap = np.zeros_like(phi)
            R = np.sign(oper.Vx) + kp
        else:
            R = np.empty_like(phi)
            a = np.empty_like(phi)
            ap = np.empty_like(phi)
            # if phi < 0.0:
            #     k *= -1
            idx=phi<0.0
            k[idx]*=-1

            # if np.isclose(k, 1.0, self.eps):
            #     return 1.0, 0, 0, Cn, Ct, alpha, CL, CD
            idx1=np.fabs(k-1.0)<=self.eps
            R[idx1]=1.0
            a[idx1]=0.0
            ap[idx1]=0.0

            # if k >= -2.0 / 3:
            #     a = k / (1 - k)
            idx2=k>=-2.0/3
            idx2[idx1]=False
            a[idx2]=k[idx2]/(1-k[idx2])
            # else:
            #     g1 = 2 * k + 1.0 / 9
            #     g2 = -2 * k - 1.0 / 3
            #     g3 = -2 * k - 7.0 / 9
            #     a = (g1 + np.sqrt(g2)) / g3
            idx2[idx1]=True
            idx3=~idx2
            g1 = 2 * k[idx3] + 1.0 / 9
            g2 = -2 * k[idx3] - 1.0 / 3
            g3 = -2 * k[idx3] - 7.0 / 9
            a[idx3] = (g1 + np.sqrt(g2)) / g3

            # u=a*oper.Vx
            if oper.Vx < 0.0:
                kp *= -1

            # if np.isclose(kp, -1.0, atol=self.eps):
            #     return 1.0, 0, 0, Cn, Ct, alpha, CL, CD
            idx4=np.fabs(kp+1.0)<=self.eps
            R[idx4]=1.0
            a[idx4]=0.0
            ap[idx4]=0.0

            ap = kp / (1 + kp)

            R = np.sin(phi) / (1 + a) - oper.Vx / oper.Vy * np.cos(phi) / (1 - ap)
        return R, a, ap, Cn, Ct, alpha, CL, CD

    def prandtl(self, phi):
        phi_sin = np.fabs(np.sin(phi))

        ftip = self.Nb / 2.0 * (self.Rtip - self.r) / (self.r * phi_sin)
        Ftip = 2.0 / np.pi * np.arccos(np.exp(-ftip))

        fhub = self.Nb / 2.0 * (self.r - self.Rhub) / (self.Rhub * phi_sin)
        Fhub = 2.0 / np.pi * np.arccos(np.exp(-fhub))
        F = Ftip * Fhub
        return F

    def find_root(self, V0, omega, rho=1.225, mu=1.81e-5, sos=340.0):
        if np.isclose(self.r, self.Rhub, self.eps) or np.isclose(self.r, self.Rtip, self.eps):
            return SectionAero()

        Vx = V0
        Vy = omega * self.r

        Vx_equal_zero = np.abs(Vx) <= self.eps
        Vy_equal_zero = np.abs(Vy) <= self.eps

        oper = Oper(Vx=V0, Vy=Vy, rho=rho, mu=mu, sos=sos)

        if Vx_equal_zero and Vy_equal_zero:
            return SectionAero()
        elif Vx_equal_zero:
            startfrom90 = False
            if Vy > 0 and self.theta > 0:
                order = self.quadrants[[0, 1]]
            elif Vy > 0 and self.theta < 0:
                order = self.quadrants[[1, 0]]
            elif Vy < 0 and self.theta > 0:
                order = self.quadrants[[2, 3]]
            else:
                order = self.quadrants[[3, 2]]
        elif Vy_equal_zero:
            startfrom90 = True
            if Vx > 0 and np.fabs(self.theta) < np.pi / 2:
                order = self.quadrants[[0, 2]]
            elif Vx < 0 and np.fabs(self.theta) < np.pi / 2:
                order = self.quadrants[[1, 3]]
            elif Vx > 0 and np.fabs(self.theta) > np.pi / 2:
                order = self.quadrants[[2, 0]]
            else:
                order = self.quadrants[[3, 1]]
        else:
            startfrom90 = False
            if Vx > 0 and Vy > 0:
                order = self.quadrants[[0, 1, 2, 3]]
            elif Vx < 0 and Vy > 0:
                order = self.quadrants[[1, 0, 3, 2]]
            elif Vx > 0 and Vy < 0:
                order = self.quadrants[[2, 3, 0, 1]]
            else:
                order = self.quadrants[[3, 2, 1, 0]]

        for i in order:
            phi_min, phi_max = i
            backsearch = False
            if not startfrom90:
                if phi_min == -np.pi / 2.0 or phi_max == -np.pi / 2.0:
                    backsearch = True
            else:
                if phi_max == np.pi / 2.0:
                    backsearch = True
            sucess, bracket = find_bracket(self.residual, args=(oper,), xmin=phi_min, xmax=phi_max, n=50, backsearch=backsearch)
            if sucess:
                sol = root_scalar(self.residual, method="bisect", bracket=bracket, args=(oper,))
                phi_star = sol.root
                R, a, ap, Cn, Ct, alpha, CL, CD = self._residual(phi_star, oper)
                u = a * Vx
                v = ap * Vy
                W2 = (Vx + u) ** 2 + (Vy - v) ** 2
                dT = Cn * 0.5 * rho * W2 * self.b
                dF = Ct * 0.5 * rho * W2 * self.b
                dQ = dF * self.r
                return SectionAero(Tn=dT[0], Tt=dF[0], Q=dQ[0], phi=phi_star, alpha=alpha[0], W=np.sqrt(W2[0]), CL=CL[0], CD=CD[0], Cn=Cn[0], Ct=Ct[0])
        return SectionAero()


class Blade:
    def __init__(self, Rtip, Rhub, Nb, sections: list[Section]):
        self.Rtip = Rtip
        self.Rhub = Rhub
        self.Nb = Nb
        self.sections = sections

        for i in sections:
            i.set_rotor_parameters(self.Nb, self.Rhub, self.Rtip)

    def solve(self, V0, omega, rho=1.225, mu=1.81e-5, sos=340.0):
        rs = []
        dTs = []
        dFs = []
        dQs = []
        for i in self.sections:
            sec_aero = i.find_root(V0, omega, rho, mu, sos)
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
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    ax.plot(af.air_data[:, 0], af.air_data[:, 1], label="CL")
    ax.plot(alpha, CL, "o", markersize=1, label="CL_interp")
    ax = fig.add_subplot(122)
    ax.plot(af.air_data[:, 0], af.air_data[:, 2], label="CD")
    ax.plot(alpha, CD, "o", markersize=1, label="CD_interp")
    ax.set_ylim(bottom=-0.50, top=1.30)

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

    fig = plt.figure(figsize=(16, 8))
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
    ax.set_ylim(bottom=0.0, top=0.14)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    # test01()
    test02()
