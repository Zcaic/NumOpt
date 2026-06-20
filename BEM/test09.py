import numpy as np
from abc import ABC, abstractmethod

# from scipy.optimize.elementwise import find_root
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d
from dataclasses import dataclass
import re
import io

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
    u: float = 0.0
    v: float = 0.0


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
    y_list = []
    for i in x_list:
        y = f(i, *args)
        y_list.append(y)
    y_list = np.array(y_list)
    flag = y_list[:-1] * y_list[1:]
    idx = np.where(flag < 0.0)
    if len(idx) == 0:
        return False, None, None
    else:
        return True, (x_list[idx[0]], x_list[idx[0] + 1])


class AirfoilEvaluator(ABC):

    @abstractmethod
    def __call__(self, Alpha, Reynold, Mach): ...

    @staticmethod
    def viterna(alpha, cl, cd, cr75, num_alpha=50):
        """
        alpha : unit: deg

        """
        AR = 1.0 / cr75
        cdmaxAR = 1.11 + 0.018 * AR
        cdmax = np.maximum(np.max(cd), cdmaxAR)

        i_ps = np.argmax(cl)
        cl_ps = cl[i_ps]
        cd_ps = cd[i_ps]
        a_ps = alpha[i_ps]

        i_bs = alpha < a_ps
        i_ns = np.argmin(cl[i_bs])
        cl_ns = cl[i_bs][i_ns]
        cd_ns = cd[i_bs][i_ns]
        a_ns = alpha[i_bs][i_ns]

        B1pos = cdmax
        A1pos = B1pos / 2.0 * np.ones(num_alpha)
        sa = np.sin(a_ps)
        ca = np.cos(a_ps)
        A2pos = (cl_ps - cdmax * sa * ca) * sa / ca**2
        B2pos = (cd_ps - cdmax * sa**2) / ca * np.ones(num_alpha)

        B1neg = cdmax
        A1neg = B1neg / 2.0
        sa = np.sin(a_ns)
        ca = np.cos(a_ns)
        A2neg = (cl_ns - cdmax * sa * ca) * sa / ca**2 * np.ones(num_alpha)
        B2neg = (cd_ns - cdmax * sa**2) / ca * np.ones(num_alpha)

        # angles of attack to extrapolate to
        apos = np.linspace(alpha[-1], np.pi, num=num_alpha + 1)
        apos = apos[1:]  # don't duplicate point
        aneg = np.linspace(-np.pi, alpha[0], num=num_alpha + 1)
        aneg = aneg[:-1]  # don't duplicate point

        # high aoa adjustments
        adjpos = np.ones(num_alpha)
        idx = apos >= np.pi / 2
        adjpos[idx] = -0.7
        A1pos[idx] *= -1
        B2pos[idx] *= -1

        # idx = findall(aneg .<= -alpha[end])

        adjneg = np.ones(num_alpha)
        idx = aneg <= -np.pi / 2
        adjneg[idx] = 0.7
        A2neg[idx] *= -1
        B2neg[idx] *= -1

        # extrapolate
        clpos = adjpos * (A1pos * np.sin(2 * apos) + A2pos * np.cos(apos) ** 2 / np.sin(apos))
        cdpos = B1pos * np.sin(apos) ** 2 + B2pos * np.cos(apos)
        clneg = adjneg * (A1neg * np.sin(2 * aneg) + A2neg * np.cos(aneg) ** 2 / np.sin(aneg))
        cdneg = B1neg * np.sin(aneg) ** 2 + B2neg * np.cos(aneg)

        # override with linear variation at ends
        idx = apos >= np.pi - a_ps
        clpos[idx] = (apos[idx] - np.pi) / a_ps * cl_ps * 0.7
        idx = aneg <= -np.pi - a_ns
        clneg[idx] = (aneg[idx] + np.pi) / a_ns * cl_ns * 0.7

        # concatenate
        alphafull = np.hstack([aneg, alpha, apos])
        clfull = np.hstack([clneg, cl, clpos])
        cdfull = np.hstack([cdneg, cd, cdpos])

        # don't allow negative drag
        cdfull = np.maximum(cdfull, 0.0001)
        return alphafull, clfull, cdfull


class FileAirfoil:
    def __init__(self, csvfile):
        # self.air_data = np.loadtxt(csvfile, ndmin=2)
        self.air_data = self.read_af_cfd(csvfile)
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

    def read_af_cfd(self, csvfile):
        pattern = r"^\s*(-?\d+.?\d+\s*)(-?\d+.?\d+\s*)*(-?\d+.?\d+\s*)$"
        pattern = re.compile(pattern)
        with open(csvfile, "r") as fin:
            content = fin.readlines()
        for idx_line, line in enumerate(content):
            match_res = re.match(pattern, line)
            if match_res is not None:
                idx_data = idx_line
                break
        data = np.loadtxt(io.StringIO("".join(content[idx_data:])), ndmin=2)
        return data


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
        R, a, ap, Cn, Ct, alpha, CL, CD, u, v = self._residual(phi, oper)
        return R

    def _residual(self, phi, oper: Oper):
        # phi = np.atleast_1d(phi)
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
            u = np.sign(phi) * kp * Cn / Ct * oper.Vy
            v = np.zeros_like(phi)
            a = np.zeros_like(phi)
            ap = np.zeros_like(phi)
            R = np.sign(phi) - k
        elif np.isclose(oper.Vy, 0.0, atol=self.eps):
            u = np.zeros_like(phi)
            v = k * Ct / Cn * np.fabs(oper.Vx)
            a = np.zeros_like(phi)
            ap = np.zeros_like(phi)
            R = np.sign(oper.Vx) + kp
        else:
            R = np.empty_like(phi)
            a = np.empty_like(phi)
            ap = np.empty_like(phi)
            if phi < 0.0:
                k *= -1
            # idx=phi<0.0
            # k[idx]*=-1

            if np.isclose(k, 1.0, self.eps):
                return 1.0, 0, 0, Cn, Ct, alpha, CL, CD, 0.0, 0.0
            # idx1=np.fabs(k-1.0)<=self.eps
            # R[idx1]=1.0
            # a[idx1]=0.0
            # ap[idx1]=0.0

            if k >= -2.0 / 3:
                a = k / (1 - k)
            # idx2=k>=-2.0/3
            # idx2[idx1]=False
            # a[idx2]=k[idx2]/(1-k[idx2])
            else:
                g1 = 2 * k + 1.0 / 9
                g2 = -2 * k - 1.0 / 3
                g3 = -2 * k - 7.0 / 9
                a = (g1 + np.sqrt(g2)) / g3
            # idx2[idx1]=True
            # idx3=~idx2
            # g1 = 2 * k[idx3] + 1.0 / 9
            # g2 = -2 * k[idx3] - 1.0 / 3
            # g3 = -2 * k[idx3] - 7.0 / 9
            # a[idx3] = (g1 + np.sqrt(g2)) / g3

            u = a * oper.Vx
            if oper.Vx < 0.0:
                kp *= -1

            if np.isclose(kp, -1.0, atol=self.eps):
                return 1.0, 0, 0, Cn, Ct, alpha, CL, CD, 0.0, 0.0
            # idx4=np.fabs(kp+1.0)<=self.eps
            # R[idx4]=1.0
            # a[idx4]=0.0
            # ap[idx4]=0.0

            ap = kp / (1 + kp)
            v = ap * oper.Vy

            R = np.sin(phi) / (1 + a) - oper.Vx / oper.Vy * np.cos(phi) / (1 - ap)
        return R, a, ap, Cn, Ct, alpha, CL, CD, u, v

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
                R, a, ap, Cn, Ct, alpha, CL, CD, u, v = self._residual(phi_star, oper)
                W2 = (Vx + u) ** 2 + (Vy - v) ** 2
                dT = Cn * 0.5 * rho * W2 * self.b
                dF = Ct * 0.5 * rho * W2 * self.b
                dQ = dF * self.r
                return SectionAero(Tn=dT, Tt=dF, Q=dQ, phi=phi_star, alpha=alpha, W=np.sqrt(W2), CL=CL, CD=CD, Cn=Cn, Ct=Ct, u=u, v=v)
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
        P = M * omega

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
        return {"T": T, "M": M, "P": P, "CT": CT, "CQ": CQ, "CP": CP, "eta": eta}


def test01():
    import matplotlib.pyplot as plt
    import scienceplots

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
    import scienceplots

    Nb = 3
    Rtip = 3.054 / 2.0
    Rhub = 0.375
    rs = np.array([Rhub, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip])
    chords = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12])
    pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))
    # rs = np.array([ 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425])
    # chords = np.array([0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12])
    # pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))

    secs = [Section(af=FileAirfoil("./pyBEMT/pybemt/airfoils/CLARKY.dat"), theta=theta, r=r, b=b) for theta, r, b in zip(pitchs, rs, chords)]
    blade = Blade(Rhub=Rhub, Rtip=Rtip, Nb=Nb, sections=secs)

    ret_list = []
    vinf_list = np.linspace(1.0, 44.0, 20)
    for i in vinf_list:
        ret = blade.solve(V0=i, omega=1100 * 2 * np.pi / 60.0, rho=1.225)
        ret_list.append(ret)

    with plt.style.context(["science", "nature", "high-vis", "no-latex"]):
        with plt.rc_context(
            {
                "axes.linewidth": 1,
                "lines.linewidth": 2,
                "axes.labelsize": 15,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.grid": True,
                "axes.grid.which": "both",
                "grid.linestyle": "--",
                # "figure.subplot.wspace":0.5
            }
        ):
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(121)
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), [i["CT"] for i in ret_list], label="My Bemt")
            print([i["CT"] for i in ret_list])

            exp_data = np.loadtxt("./propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CT")
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")

            ax = fig.add_subplot(122)
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), [i["CP"] for i in ret_list], label="My Bemt")
            print([i["CT"] for i in ret_list])

            exp_data = np.loadtxt("./propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 2], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CP")
            ax.set_ylim(bottom=0.0, top=0.14)
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.savefig("./runtime/pic01.png", dpi=300, transparent=True)
            plt.show()


def test03():
    import matplotlib.pyplot as plt
    import scienceplots

    Nb = 2
    Rtip = 0.7112 / 2.0
    Rhub = 0.03
    rs = np.array([0.07112, 0.10668, 0.14224, 0.1778, 0.21336, 0.24892, 0.28448, 0.32004])
    chords = np.array([0.056, 0.07, 0.07, 0.065, 0.058, 0.05, 0.043, 0.034])
    pitchs = np.deg2rad(np.array([19.6, 17.9, 14.4, 11.6, 9.7, 8.4, 7.2, 6.7]))
    # rs = np.array([ 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425])
    # chords = np.array([0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12])
    # pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))
    afs = [
        FileAirfoil("./pyBEMT/pybemt/airfoils/NACA_4412.dat"),
        FileAirfoil("./pyBEMT/pybemt/airfoils/GOE_450.dat"),
        FileAirfoil("./pyBEMT/pybemt/airfoils/GOE_450.dat"),
        FileAirfoil("./pyBEMT/pybemt/airfoils/GOE_450.dat"),
        FileAirfoil("./pyBEMT/pybemt/airfoils/GOE_450.dat"),
        FileAirfoil("./pyBEMT/pybemt/airfoils/GOE_450.dat"),
        FileAirfoil("./pyBEMT/pybemt/airfoils/GOE_408.dat"),
        FileAirfoil("./pyBEMT/pybemt/airfoils/GOE_408.dat"),
    ]
    secs = [Section(af=af, theta=theta, r=r, b=b) for af, theta, r, b in zip(afs, pitchs, rs, chords)]
    blade = Blade(Rhub=Rhub, Rtip=Rtip, Nb=Nb, sections=secs)

    ret_list = []
    rpm_list = np.linspace(1000.0, 3200.0, 20)
    for i in rpm_list:
        ret = blade.solve(V0=0.0, omega=i * np.pi * 2 / 60.0, rho=1.225)
        ret_list.append(ret)

    with plt.style.context(["science", "nature", "high-vis", "no-latex"]):
        with plt.rc_context(
            {
                "axes.linewidth": 1,
                "lines.linewidth": 2,
                "axes.labelsize": 15,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.grid": True,
                "axes.grid.which": "both",
                "grid.linestyle": "--",
                # "figure.subplot.wspace":0.5
            }
        ):
            with open("./pyBEMT/examples/tmotor28_data.csv", "r") as fin:
                contents = fin.read()
            contents = contents.replace(";", " ")
            exp_data = np.loadtxt(io.StringIO(contents), skiprows=1, ndmin=2)

            fig = plt.figure(figsize=(12, 5))

            ax = fig.add_subplot(121)
            ax.plot(rpm_list, [i["T"] for i in ret_list], label="My Bemt")
            print([i["T"] for i in ret_list])

            # exp_data = np.loadtxt("./pyBEMT/examples/tmotor28_data.csv", skiprows=1, ndmin=2, delimiter=";")

            ax.plot(exp_data[:, 0], exp_data[:, 3], "o", label="exp")
            ax.set_xlabel("RPM")
            ax.set_ylabel("T(N)")
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")

            ax = fig.add_subplot(122)
            ax.plot(rpm_list, [i["P"] for i in ret_list], label="My Bemt")
            print([i["P"] for i in ret_list])

            ax.plot(exp_data[:, 0], exp_data[:, 5], "o", label="exp")
            ax.set_xlabel("RPM")
            ax.set_ylabel("P")
            # ax.set_ylim(bottom=0.0, top=0.14)
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.savefig("./runtime/pic01.png", dpi=300, transparent=True)
            plt.show()


def test04():
    import matplotlib.pyplot as plt
    import scienceplots

    xfoil_data = """
 -14.000  -1.0990   0.02637   0.02282  -0.0871   0.9992   0.0166
 -13.750  -1.0711   0.02533   0.02165  -0.0885   0.9979   0.0170
 -13.500  -1.0462   0.02365   0.01985  -0.0903   0.9963   0.0177
 -13.250  -1.0163   0.02288   0.01905  -0.0918   0.9951   0.0183
 -13.000  -0.9847   0.02237   0.01850  -0.0933   0.9943   0.0189
 -12.750  -0.9549   0.02183   0.01790  -0.0943   0.9930   0.0195
 -12.500  -0.9260   0.02126   0.01724  -0.0952   0.9911   0.0201
 -12.250  -0.8954   0.02078   0.01666  -0.0963   0.9894   0.0206
 -12.000  -0.8682   0.01946   0.01525  -0.0976   0.9877   0.0214
 -11.750  -0.8365   0.01894   0.01471  -0.0990   0.9866   0.0220
 -11.500  -0.8038   0.01852   0.01424  -0.1004   0.9857   0.0227
 -11.250  -0.7707   0.01808   0.01375  -0.1019   0.9849   0.0235
 -11.000  -0.7369   0.01769   0.01328  -0.1035   0.9843   0.0242
 -10.750  -0.7070   0.01745   0.01297  -0.1041   0.9819   0.0246
 -10.500  -0.6803   0.01619   0.01161  -0.1049   0.9793   0.0257
 -10.250  -0.6491   0.01569   0.01109  -0.1060   0.9775   0.0264
 -10.000  -0.6172   0.01529   0.01065  -0.1071   0.9759   0.0271
  -9.750  -0.5850   0.01491   0.01022  -0.1082   0.9742   0.0279
  -9.500  -0.5547   0.01456   0.00981  -0.1089   0.9718   0.0287
  -9.250  -0.5287   0.01426   0.00944  -0.1085   0.9665   0.0292
  -9.000  -0.5023   0.01345   0.00855  -0.1085   0.9622   0.0299
  -8.750  -0.4769   0.01285   0.00791  -0.1082   0.9574   0.0309
  -8.500  -0.4513   0.01249   0.00752  -0.1078   0.9519   0.0317
  -8.250  -0.4243   0.01214   0.00713  -0.1076   0.9474   0.0324
  -8.000  -0.3979   0.01184   0.00678  -0.1073   0.9422   0.0333
  -7.750  -0.3715   0.01155   0.00644  -0.1070   0.9363   0.0340
  -7.500  -0.3442   0.01127   0.00609  -0.1068   0.9313   0.0345
  -7.250  -0.3183   0.01080   0.00556  -0.1064   0.9249   0.0354
  -7.000  -0.2921   0.01033   0.00505  -0.1061   0.9186   0.0365
  -6.750  -0.2649   0.01003   0.00471  -0.1059   0.9125   0.0375
  -6.500  -0.2377   0.00977   0.00441  -0.1057   0.9053   0.0384
  -6.000  -0.1825   0.00935   0.00389  -0.1054   0.8910   0.0404
  -5.750  -0.1549   0.00912   0.00360  -0.1052   0.8835   0.0414
  -5.500  -0.1275   0.00880   0.00325  -0.1051   0.8751   0.0435
  -5.000  -0.0718   0.00845   0.00283  -0.1049   0.8578   0.0476
  -4.750  -0.0441   0.00824   0.00259  -0.1047   0.8488   0.0519
  -4.500  -0.0162   0.00810   0.00243  -0.1046   0.8388   0.0569
  -4.250   0.0117   0.00793   0.00228  -0.1045   0.8288   0.0655
  -4.000   0.0394   0.00780   0.00213  -0.1044   0.8184   0.0745
  -3.750   0.0674   0.00769   0.00201  -0.1044   0.8073   0.0820
  -3.500   0.0954   0.00761   0.00191  -0.1043   0.7964   0.0890
  -3.250   0.1232   0.00752   0.00180  -0.1042   0.7851   0.0977
  -3.000   0.1512   0.00745   0.00171  -0.1041   0.7733   0.1066
  -2.750   0.1791   0.00737   0.00163  -0.1040   0.7616   0.1182
  -2.500   0.2069   0.00729   0.00156  -0.1040   0.7497   0.1332
  -2.250   0.2346   0.00723   0.00150  -0.1039   0.7378   0.1502
  -2.000   0.2625   0.00715   0.00145  -0.1038   0.7254   0.1697
  -1.750   0.2903   0.00709   0.00142  -0.1038   0.7132   0.1927
  -1.500   0.3180   0.00703   0.00141  -0.1037   0.7012   0.2214
  -1.250   0.3456   0.00701   0.00139  -0.1036   0.6886   0.2466
  -1.000   0.3734   0.00697   0.00138  -0.1035   0.6754   0.2686
  -0.750   0.4012   0.00694   0.00137  -0.1035   0.6626   0.2903
  -0.500   0.4288   0.00691   0.00138  -0.1034   0.6497   0.3203
  -0.250   0.4562   0.00686   0.00139  -0.1033   0.6365   0.3629
   0.000   0.4833   0.00678   0.00141  -0.1032   0.6232   0.4192
   0.250   0.5102   0.00658   0.00146  -0.1031   0.6101   0.5177
   0.500   0.5366   0.00635   0.00153  -0.1029   0.5975   0.6393
   0.750   0.5622   0.00617   0.00160  -0.1024   0.5856   0.7449
   1.000   0.5842   0.00594   0.00170  -0.1009   0.5740   0.8717
   1.250   0.6163   0.00588   0.00177  -0.1014   0.5622   0.9842
   1.500   0.6525   0.00598   0.00181  -0.1033   0.5505   1.0000
   1.750   0.6788   0.00611   0.00186  -0.1029   0.5398   1.0000
   2.000   0.7055   0.00622   0.00192  -0.1026   0.5294   1.0000
   2.250   0.7325   0.00633   0.00199  -0.1024   0.5204   1.0000
   2.500   0.7592   0.00646   0.00206  -0.1022   0.5112   1.0000
   2.750   0.7865   0.00656   0.00213  -0.1020   0.5029   1.0000
   3.250   0.8405   0.00681   0.00231  -0.1016   0.4847   1.0000
   3.500   0.8672   0.00696   0.00240  -0.1014   0.4746   1.0000
   3.750   0.8941   0.00709   0.00250  -0.1012   0.4646   1.0000
   4.000   0.9210   0.00722   0.00260  -0.1010   0.4540   1.0000
   4.250   0.9473   0.00739   0.00272  -0.1007   0.4426   1.0000
   4.500   0.9734   0.00758   0.00284  -0.1004   0.4273   1.0000
   4.750   0.9993   0.00778   0.00297  -0.1001   0.4110   1.0000
   5.000   1.0254   0.00797   0.00311  -0.0998   0.3979   1.0000
   5.250   1.0518   0.00813   0.00326  -0.0995   0.3861   1.0000
   5.500   1.0777   0.00834   0.00342  -0.0992   0.3731   1.0000
   5.750   1.1031   0.00857   0.00359  -0.0988   0.3575   1.0000
   6.000   1.1280   0.00884   0.00379  -0.0983   0.3398   1.0000
   6.250   1.1523   0.00914   0.00401  -0.0978   0.3207   1.0000
   6.500   1.1761   0.00948   0.00426  -0.0971   0.2993   1.0000
   6.750   1.1988   0.00989   0.00455  -0.0963   0.2737   1.0000
   7.000   1.2208   0.01036   0.00488  -0.0954   0.2461   1.0000
   7.250   1.2417   0.01089   0.00526  -0.0943   0.2173   1.0000
   7.500   1.2614   0.01149   0.00569  -0.0931   0.1865   1.0000
   7.750   1.2793   0.01220   0.00621  -0.0915   0.1526   1.0000
   8.000   1.2973   0.01288   0.00672  -0.0900   0.1252   1.0000
   8.250   1.3164   0.01345   0.00719  -0.0887   0.1065   1.0000
   8.500   1.3346   0.01404   0.00769  -0.0872   0.0893   1.0000
   8.750   1.3514   0.01469   0.00823  -0.0854   0.0729   1.0000
   9.000   1.3676   0.01527   0.00875  -0.0836   0.0622   1.0000
   9.250   1.3835   0.01581   0.00926  -0.0817   0.0563   1.0000
   9.500   1.4004   0.01631   0.00976  -0.0799   0.0521   1.0000
   9.750   1.4171   0.01682   0.01028  -0.0782   0.0491   1.0000
  10.000   1.4317   0.01746   0.01091  -0.0762   0.0459   1.0000
  10.250   1.4484   0.01797   0.01147  -0.0746   0.0442   1.0000
  10.500   1.4653   0.01849   0.01203  -0.0731   0.0427   1.0000
  10.750   1.4805   0.01911   0.01267  -0.0714   0.0411   1.0000
  11.000   1.4938   0.01986   0.01343  -0.0695   0.0392   1.0000
  11.250   1.5061   0.02069   0.01430  -0.0676   0.0376   1.0000
  11.500   1.5221   0.02129   0.01495  -0.0662   0.0368   1.0000
  11.750   1.5369   0.02199   0.01570  -0.0647   0.0356   1.0000
  12.000   1.5500   0.02282   0.01656  -0.0631   0.0343   1.0000
  12.250   1.5608   0.02382   0.01758  -0.0614   0.0330   1.0000
  12.500   1.5688   0.02506   0.01888  -0.0594   0.0316   1.0000
  12.750   1.5831   0.02588   0.01975  -0.0582   0.0308   1.0000
  13.000   1.5959   0.02683   0.02075  -0.0569   0.0298   1.0000
  13.250   1.6066   0.02796   0.02192  -0.0555   0.0286   1.0000
  13.500   1.6141   0.02939   0.02338  -0.0540   0.0273   1.0000
  13.750   1.6213   0.03089   0.02494  -0.0526   0.0262   1.0000
  14.000   1.6325   0.03209   0.02620  -0.0516   0.0252   1.0000
  14.250   1.6414   0.03354   0.02770  -0.0505   0.0241   1.0000
  14.500   1.6474   0.03528   0.02947  -0.0493   0.0229   1.0000
  14.750   1.6508   0.03731   0.03156  -0.0482   0.0218   1.0000
  15.000   1.6585   0.03899   0.03332  -0.0474   0.0209   1.0000
  15.250   1.6638   0.04096   0.03533  -0.0466   0.0198   1.0000
  15.500   1.6661   0.04330   0.03772  -0.0458   0.0187   1.0000
  15.750   1.6666   0.04589   0.04037  -0.0451   0.0179   1.0000
  16.000   1.6698   0.04827   0.04284  -0.0447   0.0171   1.0000
  16.250   1.6706   0.05099   0.04562  -0.0443   0.0164   1.0000
  16.500   1.6692   0.05402   0.04871  -0.0440   0.0157   1.0000
  16.750   1.6638   0.05759   0.05235  -0.0439   0.0151   1.0000
  17.000   1.6605   0.06101   0.05587  -0.0439   0.0146   1.0000
  17.250   1.6584   0.06435   0.05931  -0.0441   0.0142   1.0000
  17.500   1.6548   0.06793   0.06298  -0.0444   0.0138   1.0000
  17.750   1.6497   0.07175   0.06689  -0.0448   0.0134   1.0000
  18.000   1.6430   0.07583   0.07106  -0.0453   0.0131   1.0000
  18.250   1.6346   0.08024   0.07555  -0.0461   0.0128   1.0000
  18.500   1.6237   0.08507   0.08047  -0.0470   0.0124   1.0000
  18.750   1.6097   0.09040   0.08590  -0.0482   0.0121   1.0000
"""
    xfoil_data = np.loadtxt(io.StringIO(xfoil_data), ndmin=2)
    xfoil_data[:, 0] = xfoil_data[:, 0] / 180.0 * np.pi
    alpha, cl, cd = AirfoilEvaluator.viterna(xfoil_data[:, 0], xfoil_data[:, 1], xfoil_data[:, 2], cr75=0.128)

    with plt.style.context(["science", "nature", "high-vis", "no-latex"]):
        with plt.rc_context(
            {
                "axes.linewidth": 1,
                "lines.linewidth": 2,
                "axes.labelsize": 15,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.grid": True,
                "axes.grid.which": "both",
                "grid.linestyle": "--",
                # "figure.subplot.wspace":0.5
            }
        ):
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(121)
            ax.plot(alpha, cl, label="CL")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel("CL")
            ax.legend(fontsize=15)

            ax = fig.add_subplot(122)
            ax.plot(alpha, cd, label="CD")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel("CD")
            ax.legend(fontsize=15)

            plt.show()


if __name__ == "__main__":
    # test01()
    # test02()
    # test03()
    test04()
