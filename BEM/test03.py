import numpy as np
import casadi as ca
import aerosandbox.numpy as anp

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
        cl = self.CL(alpha)
        cd = self.CD(alpha)
        return cl, cd


class Section:
    def __init__(self, af, theta, r, b):
        self.af = af
        self.theta = theta
        self.V0 = None
        self.r = r
        self.omega = None  # rad/s
        self.Nb = None
        self.b = b
        self.Rhub = None
        self.Rtip = None
        self.rho = None

        self.solver_options = {
            "ipopt.sb": "yes",  # Hide the IPOPT banner.
            "ipopt.max_iter": 1000,
            "ipopt.max_cpu_time": 1e20,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.fast_step_computation": "yes",
            "detect_simple_bounds": False,
            "expand": True,
            "ipopt.print_level": 0,
            "print_time":False
        }
        

    def _residual(self, beta):
        phi = self.phi0 + beta
        alpha = self.theta - phi

        alpha_deg = alpha/anp.pi*180.0

        CL, CD = self.af(alpha_deg)
        tan_gamma = CD / CL

        # F=self.prandtl(phi)
        F = 1.0
        residual = CL * self.solidity / F - (4 * anp.sin(self.phi0 + beta) * anp.tan(beta)) / (1 - tan_gamma * anp.tan(beta))
        return residual

    def _find_root(self,beta_init):
        opti = ca.Opti()
        beta = opti.variable()
        residial = self._residual(beta)
        opti.subject_to(
            [
                residial == 0.0,
                opti.bounded(-0.5*anp.pi, beta, 0.5*anp.pi),
            ]
        )
        opti.solver("ipopt", self.solver_options)
        opti.set_initial(beta,beta_init)
        sol=opti.solve()
        sol_beta=sol.value(beta)
        return sol_beta

    def prandtl(self, phi):
        sinphi = np.abs(np.sin(phi))

        dr = np.maximum(0.0, self.Rtip - self.r)
        ftip = self.Nb / 2.0 * (dr) / (self.r * sinphi)
        Ftip = 2 / np.pi * np.arccos(np.exp(-ftip))

        dr = np.maximum(0.0, self.r - self.Rhub)
        fhub = self.Nb / 2.0 * (dr) / (self.Rhub * sinphi)
        Fhub = 2 / np.pi * np.arccos(np.exp(-fhub))

        F = Ftip * Fhub
        return F

    def find_root(self):
        self.phi0 = np.arctan(self.V0 / (self.omega * self.r))
        self.solidity = self.Nb * self.b / (2 * np.pi * self.r)

        beta_init=10./180*anp.pi
        beta = self._find_root(beta_init=beta_init)

        phi = self.phi0 + beta
        alpha = self.theta - phi
        sinphi = anp.sin(phi)
        cosphi = anp.cos(phi)

        alpha_deg = alpha/anp.pi*180.0
        CL, CD = self.af(alpha_deg)
        Cn = CL * cosphi - CD * sinphi
        Ct = CL * sinphi + CD * cosphi

        tan_gamma = CD / CL
        gamma = anp.arctan(tan_gamma)

        # a=(np.tan(phi)*(1+np.tan(self.phi0)*np.tan(phi+gamma)))/(np.tan(self.phi0)*(1+np.tan(phi)*np.tan(phi+gamma)))-1.0
        # a_=a*np.tan(self.phi0)*np.tan(phi+gamma)

        Vn = self.omega * self.r * anp.tan(phi) * (1 + anp.tan(self.phi0) * anp.tan(phi + gamma)) / (1 + anp.tan(phi) * anp.tan(phi + gamma))
        W = Vn / anp.sin(phi)
        W2 = W**2

        dT = 0.5 * self.rho * W2 * Cn * self.b * self.Nb
        dQ = 0.5 * self.rho * W2 * Ct * self.b * self.Nb * self.r

        return dT, dQ


class Blade:
    def __init__(self, Rtip, Rhub, Nb, sections: list[Section]):
        self.Rtip = Rtip
        self.Rhub = Rhub
        self.Nb = Nb
        self.sections = sections
        for i in sections:
            i.Rtip = Rtip
            i.Rhub = Rhub
            i.Nb = Nb

    def solve(self, V0, omega, rho):
        rs = []
        dTs = []
        dQs = []
        for i in self.sections:
            i.V0 = V0
            i.omega = omega
            i.rho = rho
            dT, dQ = i.find_root()
            dTs.append(np.array(dT)[0,0])
            dQs.append(np.array(dQ)[0,0])
            rs.append(i.r)
        T = np.trapezoid(y=dTs, x=rs)
        M = np.trapezoid(y=dQs, x=rs)

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
    alpha = np.linspace(-30, 30.0, 100)
    CL, CD = af(alpha)
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(af.air_data[:, 0], af.air_data[:, 1], label="CL")
    ax.plot(alpha, CL, "o", markersize=1, label="CL_interp")
    ax = fig.add_subplot(122)
    ax.plot(af.air_data[:, 0], af.air_data[:, 2], label="CD")
    ax.plot(alpha, CD, "o", markersize=1, label="CD_interp")

    plt.legend()
    plt.show()


def test02():
    import matplotlib.pyplot as plt

    Nb = 3
    Rtip = 3.054 / 2.0
    Rhub = 0.375
    rs = np.array([Rhub + 0.001, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip - 0.001])
    chords = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12])
    pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))

    secs = [Section(af=FileAirfoil("./CLARKY.csv"), theta=theta, r=r, b=b) for theta, r, b in zip(pitchs, rs, chords)]
    blade = Blade(Rhub=Rhub, Rtip=Rtip, Nb=Nb, sections=secs)

    ret_list = []
    vinf_list = np.linspace(1.0, 44.0, 20)
    for i in vinf_list:
        ret = blade.solve(V0=i, omega=1100 * 2 * np.pi / 60.0, rho=1.225)
        ret_list.append(ret)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), [i["CT"] for i in ret_list], label="My Bemt")
    print([i["CT"] for i in ret_list])

    exp_data = np.loadtxt("./propeller_dat.csv", skiprows=1, ndmin=2)
    ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
    ax.legend()
    ax.plot()
    plt.show()


if __name__ == "__main__":
    # test01()
    test02()
