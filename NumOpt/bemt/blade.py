from wisdem.ccblade.ccblade import CCBlade, CCAirfoil
import numpy as np 
import re as regex 
import io
import matplotlib.pyplot as plt
import scienceplots


def read_af_cfd(csvfile):
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

airdata=read_af_cfd("./BEM/pyBEMT/pybemt/airfoils/CLARKY.dat")
airmodel=CCAirfoil(alpha=airdata[:,0],Re=[],cl=airdata[:,1],cd=airdata[:,2])

Nb = 3
Rtip = 3.054 / 2.0
Rhub = 0.375

rs = np.array([0.525, 0.675, 0.825, 0.975, 1.125, 1.275, 1.425])
chords = np.array([ 0.18, 0.225, 0.225, 0.21, 0.1875, 0.1425, 0.12])
pitchs = np.array([17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0,])

airfoils=[airmodel for _ in range(rs.shape[0])]

rotor=CCBlade(
    r=rs,
    chord=chords,
    theta=pitchs,
    af=airfoils,
    Rhub=Rhub,
    Rtip=Rtip,
    B=Nb,
    rho=1.225,
    mu=1.789e-5
)

vinf_list = np.linspace(1.0, 44.0, 20)
omega=np.full_like(vinf_list,1100.0)
pitch=np.full_like(vinf_list,0.0)

aero,derivs=rotor.evaluate(Uinf=vinf_list,Omega=omega,pitch=pitch,coefficients=False)
T=aero["T"]
Q=aero["Q"]
P=aero["P"]

D = Rtip * 2
n = 1100.0 / 60
CT=T / (1.225 * n**2 * D**4)
CQ = Q / (1.225 * n**2 * D**5)
CP = 2 * np.pi * CQ

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
        # print([np.rad2deg(i["phi"]) for i in ret_list])

        exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
        ax.plot(exp_data[:, 0], exp_data[:, 1], label="exp")
        ax.set_xlabel("J")
        ax.set_ylabel("CT")
        ax.legend(fontsize=15)
        # ax.grid(True,"both",linestyle="-")

        ax = fig.add_subplot(122)
        ax.plot(vinf_list / (1100 / 60 * 2 * Rtip), CP, label="My Bemt")

        exp_data = np.loadtxt("./BEM/propeller_dat.csv", skiprows=1, ndmin=2)
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
