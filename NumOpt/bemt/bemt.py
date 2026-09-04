import casadi as ca
from NumOpt import Opti
import numpy as np
from dataclasses import dataclass
import aerosandbox.numpy as anp


def integrate(x, f):
    dx = ca.diff(x, 1)
    f_center = (f[0, :-1] + f[0, 1:]) / 2.0
    T_tot = ca.dot(dx, f_center)
    return T_tot


@dataclass
class Atmosphere:
    rho: float = 1.225
    mu: float = 1.81e-5
    sos: float = 340.0


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
        phi_abssin = ca.fabs(ca.sin(phi))

        ftip = self.Nb / 2.0 * (self.Rtip - self.r) / (self.r * phi_abssin + 1e-10)
        Ftip = 2.0 / np.pi * ca.arccos(ca.exp(-ftip))
        Ftip = ca.fmax(Ftip, 1e-6)

        fhub = self.Nb / 2.0 * (self.r - self.Rhub) / (self.Rhub * phi_abssin + 1e-10)
        Fhub = 2.0 / np.pi * np.arccos(np.exp(-fhub))
        Fhub = ca.fmax(Fhub, 1e-6)

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
        k = Cn * self.solidity / (4 * F * phi_sin**2)
        kp = Ct * self.solidity / (4 * F * phi_sin * phi_cos)

        a = k / (1 - k)
        ap = kp / (1 + kp)

        # R = phi_sin / (1 + a) - Vx / Vy * phi_cos / (1 - ap)
        R = Vy * (4 * F * phi_sin**2 - Cn * self.solidity) - Vx * (4 * F * phi_sin * phi_cos + Ct * self.solidity)

        return R, a, ap


class Blade:
    def __init__(self, Rtip, Rhub, Nb, chord_distribution, twist_distribution, airfoilmodel, pitch, radius_resolution=10):
        self.Rtip = Rtip
        self.Rhub = Rhub
        self.Nb = Nb
        self.chord_distribution = chord_distribution
        self.twist_distribution = twist_distribution
        self.pitch = pitch
        self.radius_resolution = radius_resolution
        if isinstance(self.radius_resolution, int):
            self.r_list = np.linspace(self.Rhub, self.Rtip, self.radius_resolution)
        elif isinstance(self.radius_resolution, np.ndarray):
            self.r_list = np.clip(self.radius_resolution, self.Rhub, self.Rtip)
        else:
            raise ValueError("the `radius_resolution` must be `np.ndarray` or `int`. ")

        self.chord_list = self.chord_distribution(self.r_list).T
        self.twist_list = self.twist_distribution(self.r_list).T

        self.airfoilmodel = airfoilmodel

        # self.residual, self.residual_map = self.__gen_residual_func()
        self.residual = self.__gen_residual_func_batch()

        self.solve_func: ca.Function = self.__solve_batch()

    def __gen_residual_func_batch(self):
        n_station = self.r_list.shape[0]
        r = self.r_list.reshape((1, -1))
        Nb = self.Nb

        phi = ca.MX.sym("phi", 1, n_station)
        theta = ca.MX.sym("theta", 1, n_station)
        Vx = ca.MX.sym("Vx", 1, 1)
        omega = ca.MX.sym("omega", 1, 1)
        rho = ca.MX.sym("rho", 1, 1)
        chord = ca.MX.sym("chord", 1, n_station)
        mu = ca.MX.sym("mu", 1, 1)
        sos = ca.MX.sym("sos", 1, 1)

        Rtip = ca.MX.sym("Rtip", 1, 1)
        Rhub = ca.MX.sym("Rhub", 1, 1)

        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)
        alpha = theta - phi
        alpha_deg = alpha / np.pi * 180.0
        solidity = Nb * chord / (2 * np.pi * r)

        Vy = omega * r
        W0 = np.sqrt(Vx**2 + Vy**2)
        Re = rho * W0 * chord / mu
        Mach = W0 / sos

        CL, CD = self.airfoilmodel(alpha_deg, Re, Mach, r)
        Cn = CL * phi_cos - CD * phi_sin
        Ct = CL * phi_sin + CD * phi_cos

        ftip = Nb / 2.0 * (Rtip - r) / (r * ca.fabs(phi_sin))
        Ftip = 2.0 / np.pi * ca.arccos(ca.exp(-ftip))
        Ftip = ca.fmax(Ftip, 1e-6)
        fhub = Nb / 2.0 * (r - Rhub) / (Rhub * ca.fabs(phi_sin))
        Fhub = 2.0 / np.pi * np.arccos(np.exp(-fhub))
        Fhub = ca.fmax(Fhub, 1e-6)

        F = Ftip * Fhub
        # F=1.0

        k = Cn * solidity / (4 * F * phi_sin**2)
        kp = Ct * solidity / (4 * F * phi_sin * phi_cos)

        a = k / (1 - k)
        ap = kp / (1 + kp)

        # R = Vy * (4 * F * phi_sin**2 - Cn * solidity) - Vx * (4 * F * phi_sin * phi_cos + Ct * solidity)
        R = phi_sin / (1 + a) - Vx / Vy * phi_cos / (1 - ap)

        func_R = ca.Function(
            "func_R",
            [phi, theta, Vx, omega, rho, chord, mu, sos, Rtip, Rhub],
            [R, a, ap, Cn, Ct],
            ["phi", "theta", "Vx", "omega", "rho", "chord", "mu", "sos", "Rtip", "Rhub"],
            ["residual", "a", "ap", "Cn", "Ct"],
        )

        return func_R

    def __gen_residual_func(self):
        phi = ca.MX.sym("phi")
        theta = ca.MX.sym("theta")
        Vx = ca.MX.sym("Vx")
        omega = ca.MX.sym("omega")
        rho = ca.MX.sym("rho")
        chord = ca.MX.sym("chord")
        mu = ca.MX.sym("mu")
        sos = ca.MX.sym("sos")
        r = ca.MX.sym("r")
        Nb = ca.MX.sym("Nb")
        Rtip = ca.MX.sym("Rtip")
        Rhub = ca.MX.sym("Rhub")

        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)
        alpha = theta - phi
        alpha_deg = alpha / np.pi * 180.0
        solidity = Nb * chord / (2 * np.pi * r)

        Vy = omega * r
        W0 = np.sqrt(Vx**2 + Vy**2)
        Re = rho * W0 * chord / mu
        Mach = W0 / sos

        CL, CD = self.airfoilmodel(alpha_deg, Re, Mach, r)
        Cn = CL * phi_cos - CD * phi_sin
        Ct = CL * phi_sin + CD * phi_cos

        ftip = Nb / 2.0 * (Rtip - r) / (r * ca.fabs(phi_sin))
        Ftip = 2.0 / np.pi * ca.arccos(ca.exp(-ftip))
        Ftip = ca.fmax(Ftip, 1e-6)
        fhub = Nb / 2.0 * (r - Rhub) / (Rhub * ca.fabs(phi_sin))
        Fhub = 2.0 / np.pi * np.arccos(np.exp(-fhub))
        Fhub = ca.fmax(Fhub, 1e-6)

        # lambda_r=omega*r/Vx
        # ftip = Nb / 2.0 *ca.fabs((Rtip-r)/(r*phi_sin))*(ca.exp(-0.15*(Nb*lambda_r-21))+0.1)
        # Ftip=2.0/np.pi*ca.arccos(ca.exp(-ftip))
        # Ftip=ca.fmax(Ftip,1e-6)
        # fhub = Nb / 2.0 *ca.fabs((Rhub-r)/(r*phi_sin))*(ca.exp(-0.15*(Nb*lambda_r-21))+0.1)
        # Fhub=2.0/np.pi*ca.arccos(ca.exp(-fhub))
        # Fhub=ca.fmax(Fhub,1e-6)

        F = Ftip * Fhub
        # F=1.0

        k = Cn * solidity / (4 * F * phi_sin**2)
        kp = Ct * solidity / (4 * F * phi_sin * phi_cos)

        a = k / (1 - k)
        ap = kp / (1 + kp)

        # R = Vy * (4 * F * phi_sin**2 - Cn * solidity) - Vx * (4 * F * phi_sin * phi_cos + Ct * solidity)
        R = phi_sin / (1 + a) - Vx / Vy * phi_cos / (1 - ap)

        func_R = ca.Function(
            "func_R",
            [phi, theta, Vx, omega, rho, chord, mu, sos, r, Nb, Rtip, Rhub],
            [R, a, ap, Cn, Ct],
            ["phi", "theta", "Vx", "omega", "rho", "chord", "mu", "sos", "r", "Nb", "Rtip", "Rhub"],
            ["residual", "a", "ap", "Cn", "Ct"],
        )
        func_R_map = func_R.map("func_R_map", "thread", self.radius_resolution, ["Vx", "omega", "rho", "mu", "sos", "Nb", "Rtip", "Rhub"], [], {})
        return func_R, func_R_map

    def solve(self, Vx, omega, atmos: Atmosphere):
        opti = Opti()

        # phi = opti.variable(init_guess=np.full((1, self.radius_resolution), np.deg2rad(1.0)), lower_bound=1e-6, upper_bound=np.deg2rad(90 - 1e-6))

        theta = self.twist_list + self.pitch
        chord = self.chord_list
        r = np.asarray(self.r_list).reshape((1, -1))
        Nb = self.Nb
        Rhub = self.Rhub
        Rtip = self.Rtip

        # lambda_r=omega*r/Vx
        # lambda_r=np.asarray(lambda_r).reshape((1,-1))
        # phi_init=ca.atan(lambda_r)

        # phi = opti.variable(init_guess=phi_init, lower_bound=ca.fmax(1e-3,0.0), upper_bound=ca.fmax(np.deg2rad(89.0),0.0))
        x = opti.variable(init_guess=np.full((1, self.radius_resolution), ca.tan(np.deg2rad(10.0))), lower_bound=ca.tan(1e-6), upper_bound=ca.tan(np.deg2rad(89.0)))
        # x = opti.variable(init_guess=np.full((1, self.radius_resolution), ca.tan(np.deg2rad(1.0))))
        # x=opti.variable(init_guess=lambda_r*0.1,lower_bound=ca.tan(1e-6),upper_bound=ca.tan(np.deg2rad(89.0)))
        phi = ca.atan(x)

        # result = self.residual_map(phi=phi, theta=theta, Vx=Vx, omega=omega, rho=atmos.rho, chord=chord, mu=atmos.mu, sos=atmos.sos, r=r, Nb=Nb, Rtip=Rtip, Rhub=Rhub)
        result = self.residual(phi=phi, theta=theta, Vx=Vx, omega=omega, rho=atmos.rho, chord=chord, mu=atmos.mu, sos=atmos.sos,Rtip=Rtip, Rhub=Rhub)

        residual = result["residual"]
        a = result["a"]
        ap = result["ap"]
        Cn = result["Cn"]
        Ct = result["Ct"]

        Un = (1 + a) * Vx
        Ut = (1 - ap) * omega * r
        U2 = Un**2 + Ut**2

        # solidity = Nb * chord / (2 * np.pi * r)
        dT = Cn * 0.5 * atmos.rho * U2 * chord * Nb
        dF = Ct * 0.5 * atmos.rho * U2 * chord * Nb
        dQ = dF * r

        T = integrate(r, dT)
        Q = integrate(r, dQ)

        # opti.minimize(ca.sum(residual**2))
        # opti.subject_to(ca.sum(residual**2)/10000.0==0.0)
        opti.subject_to(residual == 0.0)

        opti.ipopt_solver(tol=1e-6, mu_strategy="monotone", start_with_resto="yes")
        sol = opti.solve()

        phi_star = sol.value(phi)
        T_star = sol.value(T)
        Q_star = sol.value(Q)

        # a=sol.value(a)
        # ap=sol.value(ap)
        # Cn=sol.value(Cn)
        # Ct=sol.value(Ct)
        # dT=sol.value(dT)
        # dQ=sol.value(dQ)
        # residual=sol.value(residual)

        D = Rtip * 2
        n = omega / (2 * np.pi)
        CT = T_star / (atmos.rho * n**2 * D**4)
        CQ = Q_star / (atmos.rho * n**2 * D**5)
        CP = 2 * np.pi * CQ

        return {"phi": phi_star, "T": T_star, "Q": Q_star, "CT": CT, "CQ": CQ, "CP": CP}

    def __solve_batch(self):
        opti = Opti()

        Vx = opti.parameter(0.0)
        omega = opti.parameter(0.0)
        rho = opti.parameter(1.225)
        mu = opti.parameter(1.789e-5)
        sos = opti.parameter(340.0)

        theta = self.twist_list + self.pitch
        chord = self.chord_list
        r = np.asarray(self.r_list).reshape((1, -1))
        Nb = self.Nb
        Rhub = self.Rhub
        Rtip = self.Rtip

        x = opti.variable(init_guess=np.full((1, self.radius_resolution), ca.tan(np.deg2rad(10.0))), lower_bound=ca.tan(1e-6), upper_bound=ca.tan(np.deg2rad(89.0)))

        phi = ca.atan(x)

        # result = self.residual_map(phi=phi, theta=theta, Vx=Vx, omega=omega, rho=rho, chord=chord, mu=mu, sos=sos, r=r, Nb=Nb, Rtip=Rtip, Rhub=Rhub)
        result = self.residual(phi=phi, theta=theta, Vx=Vx, omega=omega, rho=rho, chord=chord, mu=mu, sos=sos, Rtip=Rtip, Rhub=Rhub)
        residual = result["residual"]
        a = result["a"]
        ap = result["ap"]
        Cn = result["Cn"]
        Ct = result["Ct"]

        Un = (1 + a) * Vx
        Ut = (1 - ap) * omega * r
        U2 = Un**2 + Ut**2

        dT = Cn * 0.5 * rho * U2 * chord * Nb
        dF = Ct * 0.5 * rho * U2 * chord * Nb
        dQ = dF * r

        T = integrate(r, dT)
        Q = integrate(r, dQ)

        D = Rtip * 2
        n = omega / (2 * np.pi)
        CT = T / (rho * n**2 * D**4)
        CQ = Q / (rho * n**2 * D**5)
        CP = 2 * np.pi * CQ

        opti.subject_to(residual == 0.0)

        opti.ipopt_solver(tol=1e-6, mu_strategy="monotone", start_with_resto="yes")
        # sol = opti.solve()
        solve_func = opti.to_function(
            "solve_func", [Vx, omega, rho, mu, sos], [phi, T, Q, CT, CQ, CP], ["Vx", "omega", "rho", "mu", "sos"], ["phi", "T", "Q", "CT", "CQ", "CP"]
        )
        return solve_func

    def solve_batch(self, Vx, omega, atmos: Atmosphere):
        Vx = np.atleast_1d(Vx)
        omega = np.atleast_1d(omega)

        if Vx.shape[0] == omega.shape[0]:
            ...
        elif Vx.shape[0] == 1 and omega.shape[0] != 1:
            Vx = np.tile(Vx, omega.shape[0])
        elif Vx.shape[0] != 1 and omega.shape[0] == 1:
            omega = np.tile(omega, Vx.shape[0])
        else:
            raise ValueError("Vx shape and omega shape are inconsistent...")
        Vx = Vx.reshape((1, -1))
        omega = omega.reshape((1, -1))
        N = Vx.shape[1]
        solve_mpi = self.solve_func.map("solve_mpi", "thread", N, ["rho", "mu", "sos"], [], {})
        ret = solve_mpi(Vx=Vx, omega=omega, rho=atmos.rho, mu=atmos.mu, sos=atmos.sos)

        return {
            "phi": ret["phi"].toarray().ravel(),
            "T": ret["T"].toarray().ravel(),
            "Q": ret["Q"].toarray().ravel(),
            "CT": ret["CT"].toarray().ravel(),
            "CQ": ret["CQ"].toarray().ravel(),
            "CP": ret["CP"].toarray().ravel(),
        }


def test01():
    import pandas as pd
    import re as regex
    import io
    import numpy as np
    import aerosandbox.numpy as anp

    class FileAirfoil:
        def __init__(self, csvfile):
            # self.air_data = np.loadtxt(csvfile, ndmin=2)
            self.air_data = self.read_af_cfd(csvfile)
            self.rlist = np.array([0.375, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, 3.054 / 2.0])
            dataset = []
            for r in self.rlist:
                df = pd.DataFrame({"r": r, "alpha": self.air_data[:, 0], "CL": self.air_data[:, 1], "CD": self.air_data[:, 2]})
                dataset.append(df)
            dataset = pd.concat(dataset, axis=0, ignore_index=True)
            dataset = dataset.sort_values(by=["r", "alpha"])
            dataset = dataset.set_index(["r", "alpha"])
            ds = dataset.to_xarray()
            # print(ds.sel(r=0.525,alpha=-180.0))
            self.CL_model = ca.interpolant("CL_model", "linear", [ds["r"].values, ds["alpha"].values], ds["CL"].values.ravel(order="F"))
            self.CD_model = ca.interpolant("CD_model", "linear", [ds["r"].values, ds["alpha"].values], ds["CD"].values.ravel(order="F"))

            # print(CL_model([0.525,-9.250]))
            # print(CD_model([0.525,-9.250]))
            # self.CL = interp1d(
            #     self.air_data[:, 0],
            #     self.air_data[:, 1],
            #     kind="quadratic",
            # )

            # self.CD = interp1d(
            #     self.air_data[:, 0],
            #     self.air_data[:, 2],
            #     kind="quadratic",
            # )

        def __call__(self, Alpha=None, Reynold=None, Mach=None, r=None):
            """
            alpha: deg

            """
            # r=anp.atleast_2d(r).res
            # Alpha=anp.atleast_2d(Alpha)
            coord = anp.concatenate([r, Alpha], axis=0)
            cl = self.CL_model(coord)
            cd = self.CD_model(coord)
            return cl, cd

        def read_af_cfd(self, csvfile):
            pattern = r"^\s*(-?\d+.?\d+\s*)(-?\d+.?\d+\s*)*(-?\d+.?\d+\s*)$"
            pattern = regex.compile(pattern)
            with open(csvfile, "r") as fin:
                content = fin.readlines()
            for idx_line, line in enumerate(content):
                match_res = regex.match(pattern, line)
                if match_res is not None:
                    idx_data = idx_line
                    break
            data = np.loadtxt(io.StringIO("".join(content[idx_data:])), ndmin=2)
            return data

    afmodel = FileAirfoil("./BEM/pyBEMT/pybemt/airfoils/CLARKY.dat")
    print(afmodel(Alpha=np.array([-180.0, -8.5]).reshape((1, -1)), r=np.array([0.525, 0.975]).reshape((1, -1))))


def test02():
    import pandas as pd
    import re as regex
    import io
    import matplotlib.pyplot as plt
    import scienceplots

    class FileAirfoil:
        def __init__(self, csvfile):
            # self.air_data = np.loadtxt(csvfile, ndmin=2)
            self.air_data = self.read_af_cfd(csvfile)
            self.rlist = np.array([0.375, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, 3.054 / 2.0])
            dataset = []
            for r in self.rlist:
                df = pd.DataFrame({"r": r, "alpha": self.air_data[:, 0], "CL": self.air_data[:, 1], "CD": self.air_data[:, 2]})
                dataset.append(df)
            dataset = pd.concat(dataset, axis=0, ignore_index=True)
            dataset = dataset.sort_values(by=["r", "alpha"])
            dataset = dataset.set_index(["r", "alpha"])
            ds = dataset.to_xarray()
            # print(ds.sel(r=0.525,alpha=-180.0))
            self.CL_model = ca.interpolant("CL_model", "linear", [ds["r"].values, ds["alpha"].values], ds["CL"].values.ravel(order="F"))
            self.CD_model = ca.interpolant("CD_model", "linear", [ds["r"].values, ds["alpha"].values], ds["CD"].values.ravel(order="F"))
            # self.CL_model=ca.interpolant("CL_model","linear",[self.air_data[:,0]],self.air_data[:,1])
            # self.CD_model=ca.interpolant("CD_model","linear",[self.air_data[:,0]],self.air_data[:,2])

        def __call__(self, Alpha=None, Reynold=None, Mach=None, r=None):
            """
            alpha: deg

            """
            coord = anp.concatenate([r, Alpha], axis=0)
            cl = self.CL_model(coord)
            cd = self.CD_model(coord)
            return cl, cd

        def read_af_cfd(self, csvfile):
            pattern = r"^\s*(-?\d+.?\d+\s*)(-?\d+.?\d+\s*)*(-?\d+.?\d+\s*)$"
            pattern = regex.compile(pattern)
            with open(csvfile, "r") as fin:
                content = fin.readlines()
            for idx_line, line in enumerate(content):
                match_res = regex.match(pattern, line)
                if match_res is not None:
                    idx_data = idx_line
                    break
            data = np.loadtxt(io.StringIO("".join(content[idx_data:])), ndmin=2)
            return data

    afmodel = FileAirfoil("./BEM/pyBEMT/pybemt/airfoils/CLARKY.dat")
    Nb = 3
    Rtip = 3.054 / 2.0
    Rhub = 0.375
    rs = np.array([Rhub, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip])
    chords = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12])
    pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))

    chords_dist = ca.interpolant("chord", "linear", [rs], chords)
    twists_dist = ca.interpolant("twist", "linear", [rs], pitchs)

    rotor = Blade(
        Rhub=Rhub,
        Rtip=Rtip,
        Nb=Nb,
        chord_distribution=chords_dist,
        twist_distribution=twists_dist,
        airfoilmodel=afmodel,
        pitch=0.0,
        radius_resolution=10,
    )

    ret_list = []
    vinf_list = np.linspace(1.0, 44.0, 20)
    atmos = Atmosphere(rho=1.225)
    for i in vinf_list:
        ret = rotor.solve(Vx=i, omega=1100 * 2 * np.pi / 60.0, atmos=atmos)
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
            # print([i["CT"] for i in ret_list])
            # print([np.rad2deg(i["phi"]) for i in ret_list])

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CT")
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")

            ax = fig.add_subplot(122)
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), [i["CP"] for i in ret_list], label="My Bemt")

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 2], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CP")
            # ax.set_ylim(bottom=0.0, top=0.14)
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.savefig("./runtime/pic01.png", dpi=300, transparent=True)
            plt.show()


def test03():
    import aerosandbox as asb
    import matplotlib.pyplot as plt
    import scienceplots

    class Airfoil:
        def __init__(self):
            self.air_data = asb.Airfoil("clarky")

        def __call__(self, Alpha=None, Reynold=None, Mach=None, r=None):
            aero = self.air_data.get_aero_from_neuralfoil(alpha=Alpha.T, mach=Mach.T, Re=Reynold.T, model_size="large")
            cl = aero["CL"].T
            cd = aero["CD"].T
            return cl, cd

    afmodel = Airfoil()
    Nb = 3
    Rtip = 3.054 / 2.0
    Rhub = 0.375
    rs = np.array([Rhub, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip])
    chords = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12])
    pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))

    chords_dist = ca.interpolant("chord", "linear", [rs], chords)
    twists_dist = ca.interpolant("twist", "linear", [rs], pitchs)

    rotor = Blade(
        Rhub=Rhub,
        Rtip=Rtip,
        Nb=Nb,
        chord_distribution=chords_dist,
        twist_distribution=twists_dist,
        airfoilmodel=afmodel,
        pitch=0.0,
        radius_resolution=10,
    )

    ret_list = []
    vinf_list = np.linspace(1.0, 44.0, 20)
    atmos = Atmosphere(rho=1.225)
    for i in vinf_list:
        ret = rotor.solve(Vx=i, omega=1100 * 2 * np.pi / 60.0, atmos=atmos)
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
            # print([i["CT"] for i in ret_list])
            # print([np.rad2deg(i["phi"]) for i in ret_list])

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CT")
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")

            ax = fig.add_subplot(122)
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), [i["CP"] for i in ret_list], label="My Bemt")

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 2], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CP")
            # ax.set_ylim(bottom=0.0, top=0.14)
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.savefig("./runtime/pic01.png", dpi=300, transparent=True)
            plt.show()


def test04():
    ...

def test05():
    import aerosandbox as asb
    import matplotlib.pyplot as plt
    import scienceplots

    class Airfoil:
        def __init__(self):
            # self.air_data = np.loadtxt(csvfile, ndmin=2)
            # self.station = station
            self.air_data = asb.Airfoil("clarky")

        def __call__(self, Alpha=None, Reynold=None, Mach=None, r=None):
            aero = self.air_data.get_aero_from_neuralfoil(alpha=Alpha.T, mach=Mach.T, Re=Reynold.T, model_size="large")
            cl = aero["CL"].T
            cd = aero["CD"].T
            return cl, cd

    afmodel = Airfoil()
    Nb = 3
    Rtip = 3.054 / 2.0
    Rhub = 0.375
    rs = np.array([Rhub, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip])
    chords = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12])
    pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))

    chords_dist = ca.interpolant("chord", "linear", [rs], chords)
    twists_dist = ca.interpolant("twist", "linear", [rs], pitchs)

    rotor = Blade(
        Rhub=Rhub,
        Rtip=Rtip,
        Nb=Nb,
        chord_distribution=chords_dist,
        twist_distribution=twists_dist,
        airfoilmodel=afmodel,
        pitch=0.0,
        radius_resolution=10,
    )

    vinf_list = np.linspace(1.0, 44.0, 20)
    atmos = Atmosphere(rho=1.225)

    ret = rotor.solve_batch(Vx=vinf_list, omega=1100 * 2 * np.pi / 60.0, atmos=atmos)

    # for i in vinf_list:
    #     ret = rotor.solve(Vx=i, omega=1100 * 2 * np.pi / 60.0, atmos=atmos)
    #     ret_list.append(ret)

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
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), ret["CT"], label="My Bemt")
            # print([i["CT"] for i in ret_list])
            # print([np.rad2deg(i["phi"]) for i in ret_list])

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CT")
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")

            ax = fig.add_subplot(122)
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), ret["CP"], label="My Bemt")

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 2], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CP")
            # ax.set_ylim(bottom=0.0, top=0.14)
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.savefig("./runtime/pic01.png", dpi=300, transparent=True)
            plt.show()


def test06():
    import aerosandbox as asb
    import matplotlib.pyplot as plt
    import scienceplots

    class Airfoil:
        def __init__(self, Rtip):
            # self.air_data = np.loadtxt(csvfile, ndmin=2)
            self.Rtip = Rtip
            self.station = np.array([0.0, 0.2, 0.5, 0.75, 1.0])
            self.station = ca.DM(self.station)
            self.air_data = asb.Airfoil("clarky")

            v = ca.MX.sym("v", self.station.shape[0])
            x = ca.MX.sym("x")
            self.findidx = ca.low(v, x)

        def __call__(self, Alpha=None, Reynold=None, Mach=None, r=None):
            r_nor = r / self.Rtip
            # idx = np.searchsorted(self.station, r_nor, side="right")
            idx = self.findidx(self.station, r_nor)
            left = self.station.get(False, idx)
            right = self.station.get(False, idx + 1)

            dr = right - left
            dr_left = r_nor - left
            dr_right = right - r_nor

            w_left = dr_right / dr
            w_right = dr_left / dr

            aero_left = self.air_data.get_aero_from_neuralfoil(alpha=Alpha, mach=Mach, Re=Reynold, model_size="large")
            aero_right = self.air_data.get_aero_from_neuralfoil(alpha=Alpha, mach=Mach, Re=Reynold, model_size="large")
            cl = aero_left["CL"] * w_left + aero_right["CL"] * w_right
            cd = aero_left["CD"] * w_left + aero_right["CD"] * w_right
            return cl, cd

    Nb = 3
    Rtip = 3.054 / 2.0
    Rhub = 0.375
    rs = np.array([Rhub, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip])
    chords = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12])
    pitchs = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]))

    afmodel = Airfoil(Rtip=Rtip)

    chords_dist = ca.interpolant("chord", "linear", [rs], chords)
    twists_dist = ca.interpolant("twist", "linear", [rs], pitchs)

    rotor = Blade(
        Rhub=Rhub,
        Rtip=Rtip,
        Nb=Nb,
        chord_distribution=chords_dist,
        twist_distribution=twists_dist,
        airfoilmodel=afmodel,
        pitch=0.0,
        radius_resolution=10,
    )

    vinf_list = np.linspace(1.0, 44.0, 20)
    atmos = Atmosphere(rho=1.225)

    ret = rotor.solve_batch(Vx=vinf_list, omega=1100 * 2 * np.pi / 60.0, atmos=atmos)

    # for i in vinf_list:
    #     ret = rotor.solve(Vx=i, omega=1100 * 2 * np.pi / 60.0, atmos=atmos)
    #     ret_list.append(ret)

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
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), ret["CT"], label="My Bemt")
            # print([i["CT"] for i in ret_list])
            # print([np.rad2deg(i["phi"]) for i in ret_list])

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CT")
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")

            ax = fig.add_subplot(122)
            ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), ret["CP"], label="My Bemt")

            exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
            ax.plot(exp_data[:, 0], exp_data[:, 2], label="exp")
            ax.set_xlabel("J")
            ax.set_ylabel("CP")
            # ax.set_ylim(bottom=0.0, top=0.14)
            ax.legend(fontsize=15)
            # ax.grid(True,"both",linestyle="-")
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.savefig("./runtime/pic01.png", dpi=300, transparent=True)
            plt.show()


if __name__ == "__main__":
    # test01()
    # test02()
    test03()
    # test04()
    # test05()
    # test06()
