from NumOpt import Opti, ca
import numpy as np
import aerosandbox as asb
import gc
import subprocess
from pathlib import Path
import re as regex
import io
import pandas as pd


class Blade:
    def __init__(self, Rtip, radius_resolution, Nb, afmodel):
        self.rr = radius_resolution.reshape((1, -1))
        self.Rtip = Rtip
        self.Rhub = self.rr[0, 0] * self.Rtip
        self.Nb = Nb
        self.n_stator = self.rr.shape[1] - 2
        self.afmodel = afmodel

        self.residual, self.find_phi = self.gen_residual()

        # self.dllfile=self.compile()
        # self.load_compile(self.dllfile.as_posix())

    # def compile(self):
    #     Cg=ca.CodeGenerator("bemt.c")
    #     Cg.add(self.gen_residual())
    #     cfile=Path(Cg.generate())
    #     dllfile=cfile.with_suffix(".dll")
    #     INC = Path(ca.GlobalOptions.getCasadiIncludePath()).resolve().as_posix()
    #     LIB = Path(ca.GlobalOptions.getCasadiPath()).resolve().as_posix()
    #     cmd = ["gcc", "-O3", "-fPIC", "-shared", "-o", dllfile.as_posix(), cfile.resolve().as_posix(), "-I", INC, "-L", LIB, "-lipopt", "-lm"]
    #     pid = subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE, timeout=20.0)
    #     if pid.returncode !=0:
    #         raise RuntimeError(pid.stderr.decode())
    #     return dllfile

    # def load_compile(self,dllfile):
    #     self.residual=ca.external("residual_func",dllfile)
    def mingw_jit(self):
        mingw_jit_options = {
            "jit": False,
            "jit_temp_suffix": True,
            # "jit_name":"bemt.c",
            "compiler": "shell",
            "jit_options": {
                "compiler": "gcc",  # Force GCC instead of cl.exe
                "linker": "gcc",  # Force GCC for linking
                "compiler_setup": "-fPIC -c",  # GNU compilation flags
                "linker_setup": "-shared",  # GNU linking flags
                "compiler_output_flag": "-o ",
                "linker_output_flag": "-o ",
                "flags": [
                    "-O3",
                    "-lm" "-march=native",
                    "-ffast-math",
                    # "-flto",
                    # "-fopenmp",
                ],  # Optimize output code
                "verbose": True,
            },
        }
        return mingw_jit_options

    def gen_residual(self):
        mingw_jit = self.mingw_jit()

        r = self.rr[[0], 1:-1] * self.Rtip
        n_stator = self.n_stator
        Rhub = self.Rhub
        Rtip = self.Rtip
        Nb = self.Nb

        phi = ca.MX.sym("phi", 1, n_stator)
        Vx = ca.MX.sym("Vx")
        omega = ca.MX.sym("omega")
        chord = ca.MX.sym("chord", 1, n_stator)
        twist = ca.MX.sym("twist", 1, n_stator)
        pitch = ca.MX.sym("pitch")
        rho = ca.MX.sym("rho")
        mu = ca.MX.sym("mu")
        sos = ca.MX.sym("sos")

        Vy = omega * r

        W0 = np.sqrt(Vx**2 + Vy**2)
        Re = rho * W0 * chord / mu
        Mach = W0 / sos

        alpha = twist + pitch - phi
        alpha_deg = alpha / np.pi * 180.0
        CL, CD = self.afmodel(Reynold=Re, Mach=Mach, Alpha=alpha_deg, r=r)

        R, a, ap, Cn, Ct, F = self._residual(phi=phi, Vx=Vx, Vy=Vy, chord=chord, CL=CL, CD=CD, Rhub=Rhub, Rtip=Rtip, Nb=Nb, r=r)

        func = ca.Function(
            "residual_func",
            [phi, Vx, omega, chord, twist, pitch, rho, mu, sos],
            [R, a, ap, Cn, Ct, F],
            ["phi", "Vx", "omega", "chord", "twist", "pitch", "rho", "mu", "sos"],
            ["R", "a", "ap", "Cn", "Ct", "F"],
            mingw_jit,
        )

        p_all = ca.vertcat(
            Vx,
            omega,
            chord.T,
            twist.T,
            pitch,
            rho,
            mu,
            sos,
        )

        mingw_jit["max_step"] = 1.0 / 180.0 * np.pi
        func_find_phi = ca.rootfinder(
            "find_phi",
            "newton",
            {"x": phi.T, "p": p_all, "g": R.T},
            {"max_step":5.0 / 180.0 * np.pi}
            # mingw_jit,
        )

        return func, func_find_phi

    def solve(self, Vx, omega, chord, twist, pitch, rho, mu, sos):
        p_all = ca.vertcat(
            Vx,
            omega,
            chord[[0], 1:-1].T,
            twist[[0], 1:-1].T,
            pitch,
            rho,
            mu,
            sos,
        )
        phi_star = self.find_phi(x0=np.deg2rad(np.full((self.n_stator, 1), 10.0)), p=p_all)["x"]
        R = self.residual(phi=phi_star.T, Vx=Vx, omega=omega, chord=chord[[0],1:-1], twist=twist[[0],1:-1], pitch=pitch, rho=rho, mu=mu, sos=sos)
        a=R["a"]
        ap=R["ap"]
        Cn=R["Cn"]
        Ct=R["Ct"]

        phi_star = ca.horzcat(0.0, phi_star.T, 0.0)
        a = ca.horzcat(0.0, a, 0.0)
        ap = ca.horzcat(0.0, ap, 0.0)
        Cn = ca.horzcat(0.0, Cn, 0.0)
        Ct = ca.horzcat(0.0, Ct, 0.0)

        r = self.rr * self.Rtip
        Un = (1 + a) * Vx
        Ut = (1 - ap) * omega * r
        U2 = Un**2 + Ut**2

        dT = Cn * 0.5 * rho * U2 * chord * self.Nb
        dF = Ct * 0.5 * rho * U2 * chord * self.Nb
        dQ = dF * r

        T = self.integrate(r, dT)
        Q = self.integrate(r, dQ)

        D = self.Rtip * 2
        n = omega / (2 * np.pi)
        CT = T / (rho * n**2 * D**4)
        CQ = Q / (rho * n**2 * D**5)
        CP = 2 * np.pi * CQ

        print(T)
        # print(phi_star)

        return {"phi": phi_star, "T": T, "Q": Q, "CT": CT, "CQ": CQ, "CP": CP}

    @staticmethod
    def integrate(x, f):
        dx = ca.diff(x, 1)
        f_center = (f[0, :-1] + f[0, 1:]) / 2.0
        T_tot = ca.dot(dx, f_center)
        return T_tot

    @staticmethod
    def _residual(phi, Vx, Vy, chord, CL, CD, Rhub, Rtip, Nb, r):

        def prantl_loss(phi, r, Rhub, Rtip, Nb):
            sphi = ca.fabs(ca.sin(phi)) + 1e-9

            ftip = Nb / 2.0 * (Rtip - r) / (r * sphi)
            Ftip = (2.0 / np.pi) * ca.arccos(ca.exp(-ftip))

            fhub = Nb / 2.0 * (r - Rhub) / (Rhub * sphi)
            Fhub = 2.0 / np.pi * np.arccos(np.exp(-fhub))

            F = Ftip * Fhub
            # F=1.0

            return F

        def axial_induction(k, phi):
            k = ca.if_else(phi < 0.0, -k, k)
            a_normal = k / (1 - k)
            g1 = 2 * k + 1.0 / 9
            g2 = -2 * k - 1.0 / 3
            g3 = -2 * k - 7.0 / 9
            a_buhl = (g1 + ca.sqrt(g2)) / g3
            a = ca.if_else(k >= -2 / 3, a_normal, a_buhl)
            invalid = ca.fabs(k - 1.0) < 1e-9
            return a, invalid

        def tangential_induction(kp, Vx):
            kp = ca.if_else(Vx < 0.0, -kp, kp)
            ap = kp / (1 + kp)
            invaild = ca.fabs(kp + 1.0) < 1e-9
            return ap, invaild

        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)

        solidity = Nb * chord / (2 * np.pi * r)

        Cn = CL * phi_cos - CD * phi_sin
        Ct = CL * phi_sin + CD * phi_cos

        F = prantl_loss(phi=phi, r=r, Rhub=Rhub, Rtip=Rtip, Nb=Nb)

        k = Cn * solidity / (4 * F * phi_sin**2 + 1e-9)
        kp = Ct * solidity / (4 * F * phi_sin * phi_cos + 1e-9)

        a, inv_a = axial_induction(k=k, phi=phi)
        ap, inv_ap = tangential_induction(kp=kp, Vx=Vx)

        invaild = ca.logic_or(inv_a, inv_ap)

        R_reg = np.sin(phi) / (1 + a) - Vx / Vy * np.cos(phi) / (1 - ap)

        R = ca.if_else(invaild, 1.0, R_reg)

        return R, a, ap, Cn, Ct, F


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

    def __call__(self, Alpha=None, Reynold=None, Mach=None, r=None, opti=None):
        """
        alpha: deg

        """
        coord = ca.vcat([r, Alpha])
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


af_model = FileAirfoil(csvfile="../BEM/pyBEMT/pybemt/airfoils/CLARKY.dat")
chord = np.array([0.18, 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12, 0.12]).reshape((1, -1))
twist = np.deg2rad(np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0])).reshape((1, -1))

Rtip = 3.054 / 2.0
Rhub = 0.375
r = np.array([Rhub, 0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425, Rtip]).reshape((1, -1))
sos = 340.0
mu = 0.0000181
rho = 1.225

rotor = Blade(Rtip=Rtip, radius_resolution=r / Rtip, Nb=3, afmodel=FileAirfoil(csvfile="../BEM/pyBEMT/pybemt/airfoils/CLARKY.dat"))

aero_list=[]
vinf_list = np.linspace(1.0, 44.0, 40)
for vx in vinf_list:
    aero=rotor.solve(Vx=vx, omega=1100 * 2 * np.pi / 60.0, chord=chord, twist=twist, pitch=0.0, rho=rho, mu=mu, sos=sos)
    aero_list.append(aero)

import matplotlib.pyplot as plt 
import scienceplots 

CT=ca.vcat([i["CT"] for i in aero_list])

CP=ca.vcat([i["CP"] for i in aero_list])

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
        ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), CT, label="My Bemt")
        # print([i["CT"] for i in ret_list])

        exp_data = np.loadtxt("../BEM/propeller_dat.csv", skiprows=1, ndmin=2)
        ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
        ax.set_xlabel("J")
        ax.set_ylabel("CT")
        ax.legend(fontsize=15)
        # ax.grid(True,"both",linestyle="-")

        ax = fig.add_subplot(122)
        ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), CP, label="My Bemt")
        # print([i["CT"] for i in ret_list])

        exp_data = np.loadtxt("../BEM/propeller_dat.csv", skiprows=1, ndmin=2)
        ax.plot(exp_data[:, 0], exp_data[:, 2], label="exp")
        ax.set_xlabel("J")
        ax.set_ylabel("CP")
        # ax.set_ylim(bottom=0.0, top=0.14)
        ax.legend(fontsize=15)
        # ax.grid(True,"both",linestyle="-")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        # plt.savefig("./runtime/pic01.png", dpi=300, transparent=True)
        plt.show()