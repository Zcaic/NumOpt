from NumOpt import Opti, ca
import numpy as np
import aerosandbox as asb
import gc
import subprocess
from pathlib import Path
import re as regex
import io
import pandas as pd
from scipy.interpolate import pchip_interpolate
import aerosandbox.numpy as anp


class Blade:
    def __init__(self, Rtip, radius_resolution, Nb, afmodel):
        self.rr = np.asarray(radius_resolution, dtype=float).reshape((1, -1))
        self.Rtip = Rtip
        self.Rhub = self.rr[0, 0] * self.Rtip
        self.Nb = Nb
        self.n_stator = self.rr.shape[1] - 2
        self.afmodel = afmodel
        self.n_af = len(self.afmodel.af_list) if hasattr(self.afmodel, "af_list") else 5

        self.residual, self.find_phi = self.gen_residual()
        self.residual = self.residual.map(self.n_stator, "openmp")
        self.find_phi = self.find_phi.map(self.n_stator, "openmp")

        # self.solve_batch_func = self.gen_solve_batch()

    def gen_residual(self):
        Rhub = self.Rhub
        Rtip = self.Rtip
        Nb = self.Nb
        n_af = self.n_af

        r = ca.MX.sym("r")
        phi = ca.MX.sym("phi")
        Vx = ca.MX.sym("Vx")
        omega = ca.MX.sym("omega")
        chord = ca.MX.sym("chord")
        twist = ca.MX.sym("twist")
        pitch = ca.MX.sym("pitch")
        rho = ca.MX.sym("rho")
        mu = ca.MX.sym("mu")
        sos = ca.MX.sym("sos")

        # 与 AirfoilModel 中的 5 个控制翼型截面维度保持一致 (8x5, 8x5, 1x5)
        cst_upper = ca.MX.sym("cst_upper", 8, n_af)
        cst_lower = ca.MX.sym("cst_lower", 8, n_af)
        cst_le = ca.MX.sym("cst_le", 1, n_af)

        Vy = omega * r

        W0 = ca.sqrt(Vx**2 + Vy**2)
        Re = rho * W0 * chord / mu
        Mach = W0 / sos

        alpha = twist + pitch - phi
        alpha_deg = alpha / np.pi * 180.0
        CL, CD = self.afmodel(
            Reynold=Re,
            Mach=Mach,
            Alpha=alpha_deg,
            r=r,
            Rtip=Rtip,
            cst_upper=cst_upper,
            cst_lower=cst_lower,
            cst_le=cst_le,
        )

        R, a, ap, Cn, Ct, F = self._residual(
            phi=phi, Vx=Vx, Vy=Vy, chord=chord, CL=CL, CD=CD, Rhub=Rhub, Rtip=Rtip, Nb=Nb, r=r
        )

        func = ca.Function(
            "residual_func",
            [r, phi, Vx, omega, chord, twist, pitch, rho, mu, sos, ca.vec(cst_upper), ca.vec(cst_lower), ca.vec(cst_le)],
            [R, a, ap, Cn, Ct, F],
            ["r", "phi", "Vx", "omega", "chord", "twist", "pitch", "rho", "mu", "sos", "cst_upper", "cst_lower", "cst_le"],
            ["R", "a", "ap", "Cn", "Ct", "F"],
        )

        p_all = ca.vertcat(
            r, Vx, omega, chord, twist, pitch, rho, mu, sos, ca.vec(cst_upper), ca.vec(cst_lower), ca.vec(cst_le)
        )

        # mingw_jit["max_step"] = 5.0 / 180.0 * np.pi
        func_find_phi = ca.rootfinder(
            "find_phi",
            "bisection",
            {"x": phi, "p": p_all, "g": R},
            {
                "max_search": 100,
                "max_iter": 100,
                "search_step": 5 / 180.0 * ca.pi,
                "error_on_fail": False,
                "lb": -0.5 * ca.pi + 1e-6,
                "ub": 0.5 * ca.pi - 1e-6,
                # "abstol": 1e-4,
            },
            # mingw_jit,
        )

        return func, func_find_phi

    def solve(self, Vx, omega, chord, twist, pitch, rho, mu, sos, cst_upper, cst_lower, cst_le):
        Vx_list = ca.repmat(Vx, 1, self.n_stator)
        omega_list = ca.repmat(omega, 1, self.n_stator)
        chord_list = chord[[0], 1:-1]
        twist_list = twist[[0], 1:-1]
        pitch_list = ca.repmat(pitch, 1, self.n_stator)
        rho_list = ca.repmat(rho, 1, self.n_stator)
        mu_list = ca.repmat(mu, 1, self.n_stator)
        sos_list = ca.repmat(sos, 1, self.n_stator)

        cst_upper_list = ca.repmat(ca.vec(cst_upper), 1, self.n_stator)
        cst_lower_list = ca.repmat(ca.vec(cst_lower), 1, self.n_stator)
        cst_le_list = ca.repmat(ca.vec(cst_le), 1, self.n_stator)

        r_val = self.rr[[0], 1:-1] * self.Rtip

        p_all = ca.vertcat(
            r_val,
            Vx_list,
            omega_list,
            chord_list,
            twist_list,
            pitch_list,
            rho_list,
            mu_list,
            sos_list,
            cst_upper_list,
            cst_lower_list,
            cst_le_list,
        )

        # 使用几何入流角 arctan2(Vx, omega * r) 作为初值，兼顾悬停与高速巡航大入流角工况
        phi_guess = ca.atan2(Vx_list, omega_list * r_val)
        phi_star = self.find_phi(x0=phi_guess, p=p_all)["x"]

        R = self.residual(
            r=r_val,
            phi=phi_star,
            Vx=Vx_list,
            omega=omega_list,
            chord=chord_list,
            twist=twist_list,
            pitch=pitch_list,
            rho=rho_list,
            mu=mu_list,
            sos=sos_list,
            cst_upper=cst_upper_list,
            cst_lower=cst_lower_list,
            cst_le=cst_le_list,
        )
        a = R["a"]
        ap = R["ap"]
        Cn = R["Cn"]
        Ct = R["Ct"]

        phi_star = ca.horzcat(0.0, phi_star, 0.0)
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
        P = Q * omega

        D = self.Rtip * 2
        n = omega / (2 * np.pi)
        CT = T / (rho * n**2 * D**4)
        CQ = Q / (rho * n**2 * D**5)
        CP = 2 * np.pi * CQ

        return {"phi": phi_star, "T": T, "Q": Q, "P": P, "CT": CT, "CQ": CQ, "CP": CP, "dT": dT}

    @staticmethod
    def integrate(x, f):
        dx = ca.diff(x, 1)
        f_center = (f[0, :-1] + f[0, 1:]) / 2.0
        T_tot = ca.dot(dx, f_center)
        return T_tot

    @staticmethod
    def _residual(phi, Vx, Vy, chord, CL, CD, Rhub, Rtip, Nb, r):
        def prandtl_loss(phi, r, Rhub, Rtip, Nb):
            # 使用光滑绝对值并限制 arccos 自变量范围，避免在 phi=0 或叶尖/叶根边界处导数出现 NaN
            sphi = ca.sqrt(ca.sin(phi) ** 2 + 1e-12)

            ftip = Nb / 2.0 * (Rtip - r) / (r * sphi)
            Ftip = (2.0 / np.pi) * ca.arccos(ca.fmin(ca.exp(-ftip), 1.0 - 1e-9))

            fhub = Nb / 2.0 * (r - Rhub) / (Rhub * sphi)
            Fhub = (2.0 / np.pi) * ca.arccos(ca.fmin(ca.exp(-fhub), 1.0 - 1e-9))

            F = ca.fmax(Ftip * Fhub, 1e-4)
            return F

        def axial_induction(k, phi):
            k_eff = ca.if_else(phi < 0.0, -k, k)
            # 保护 1 - k 分母避免在搜索过程中 k=1 处除零
            denom_normal = ca.if_else(ca.fabs(1.0 - k_eff) < 1e-7, 1e-7, 1.0 - k_eff)
            a_normal = k_eff / denom_normal

            g1 = 2.0 * k_eff + 1.0 / 9.0
            # 必须对 g2 进行非负截断，避免 CasADi 在 k >= -2/3 分支求值与求导时因 sqrt(负数) 产生 NaN
            g2 = ca.fmax(-2.0 * k_eff - 1.0 / 3.0, 1e-9)
            g3 = ca.if_else(ca.fabs(-2.0 * k_eff - 7.0 / 9.0) < 1e-7, -1e-7, -2.0 * k_eff - 7.0 / 9.0)
            a_buhl = (g1 + ca.sqrt(g2)) / g3

            a = ca.if_else(k_eff >= -2.0 / 3.0, a_normal, a_buhl)
            # 利用 1/(1 + a_normal) = 1 - k_eff 消除残差方程在 k = 1 处的奇异极点
            inv_one_plus_a = ca.if_else(k_eff >= -2.0 / 3.0, 1.0 - k_eff, 1.0 / (1.0 + a_buhl))
            return a, inv_one_plus_a

        def tangential_induction(kp, Vx):
            kp_eff = ca.if_else(Vx < 0.0, -kp, kp)
            denom_ap = ca.if_else(ca.fabs(1.0 + kp_eff) < 1e-7, 1e-7, 1.0 + kp_eff)
            ap = kp_eff / denom_ap
            # 利用 1/(1 - ap) = 1 + kp_eff 消除分母奇异性
            inv_one_minus_ap = 1.0 + kp_eff
            return ap, inv_one_minus_ap

        phi_sin = ca.sin(phi)
        phi_cos = ca.cos(phi)

        solidity = Nb * chord / (2 * np.pi * r)

        Cn = CL * phi_cos - CD * phi_sin
        Ct = CL * phi_sin + CD * phi_cos

        F = prandtl_loss(phi=phi, r=r, Rhub=Rhub, Rtip=Rtip, Nb=Nb)

        k = Cn * solidity / (4.0 * F * phi_sin**2 + 1e-9)
        kp = Ct * solidity / (4.0 * F * phi_sin * phi_cos + 1e-9)

        a, inv_one_plus_a = axial_induction(k=k, phi=phi)
        ap, inv_one_minus_ap = tangential_induction(kp=kp, Vx=Vx)

        R = phi_sin * inv_one_plus_a - (Vx / Vy) * phi_cos * inv_one_minus_ap

        return R, a, ap, Cn, Ct, F


class CoxDeBoorBasis:
    def __init__(self, ncpts: int, degree: int = 3):
        self.ncpts = ncpts
        self.degree = degree
        self.knots = self.quasi_uniform_knots(self.ncpts, self.degree)

        valid_spans = [i for i in range(len(self.knots) - 1) if self.knots[i + 1] > self.knots[i]]
        self.first_span = valid_spans[0]
        self.last_span = valid_spans[-1]

        self._func_get_basis = self.__gen_basis_func()

    @staticmethod
    def quasi_uniform_knots(ncpts, degree):
        middle = np.linspace(0, 1, ncpts - degree + 1)
        start = np.zeros(degree, dtype="f8")
        end = np.ones(degree, dtype="f8")
        return np.hstack([start, middle, end])

    def _cox_de_boor_recursive(self, u_sym, knots, i, p):
        if p == 0:
            cond = ca.logic_and(u_sym >= knots[i], u_sym < knots[i + 1])

            if i == self.first_span:
                cond = ca.logic_or(cond, u_sym < knots[i])

            if i == self.last_span:
                cond = ca.logic_or(cond, u_sym >= knots[i + 1])

            return ca.if_else(cond, 1.0, 0.0)
        else:
            val = 0.0
            denom1 = knots[i + p] - knots[i]
            if denom1 > 1e-14:
                term1 = ((u_sym - knots[i]) / denom1) * self._cox_de_boor_recursive(u_sym, knots, i, p - 1)
                val += term1

            denom2 = knots[i + p + 1] - knots[i + 1]
            if denom2 > 1e-14:
                term2 = ((knots[i + p + 1] - u_sym) / denom2) * self._cox_de_boor_recursive(u_sym, knots, i + 1, p - 1)
                val += term2

            return val

    def __gen_basis_func(self):
        u = ca.MX.sym("u")
        basis_expr = [self._cox_de_boor_recursive(u, self.knots, i, self.degree) for i in range(self.ncpts)]
        basis_vector = ca.vertcat(*basis_expr)
        return ca.Function("get_basis", [u], [basis_vector], ["u"], ["N"])

    def get_basis(self, u_val):
        if isinstance(u_val, (ca.MX, ca.SX, ca.DM)):
            u_val = ca.reshape(u_val, 1, u_val.numel())
        else:
            u_val = np.asarray(u_val, dtype=float).reshape((1, -1))
        N = self._func_get_basis(u_val)
        return N

    def get_curve_point(self, cpts, u_val=None, u=None):
        if u_val is None:
            u_val = u
        weights = self.get_basis(u_val)
        ret = ca.mtimes(weights.T, cpts)
        return ret

    def get_xy_from_x(self, x, cpts):
        x_arr = np.asarray(x, dtype=float).reshape((-1, 1))
        n_pts = x_arr.shape[0]
        u = ca.MX.sym("u", n_pts, 1)
        cpts_mx = ca.MX.sym("cpts", *cpts.shape)

        xy = self.get_curve_point(cpts=cpts_mx, u_val=u)
        res = xy[:, 0] - ca.DM(x_arr)
        try:
            rfp = ca.rootfinder("rfp", "fast_newton", {"x": u, "p": ca.vec(cpts_mx), "g": res}, {"max_step": 0.2})
        except Exception:
            rfp = ca.rootfinder("rfp", "fast_newton", {"x": u, "p": ca.vec(cpts_mx), "g": res})

        # 使用线性归一化坐标作为参数反求初值，避免在边界处固定初值 0.5 导致收敛困难
        u0 = np.clip((x_arr - x_arr[0]) / max(float(x_arr[-1] - x_arr[0]), 1e-9), 1e-4, 1.0 - 1e-4)
        u_star = rfp(x0=u0, p=ca.vec(cpts))["x"]
        xy_star = self.get_curve_point(cpts=cpts, u_val=u_star)
        return xy_star


class AirfoilModel:
    def __init__(self, stator=None):
        if stator is None:
            stator = np.array([0.10, 0.25, 0.50, 0.75, 1.00], dtype=float)
        self.stator = np.asarray(stator, dtype=float).reshape((1, -1))
        self.r_min = float(self.stator[0, 0])
        self.r_max = float(self.stator[0, -1])

        self.af_list = [
            asb.Airfoil("n63428").set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3),
            asb.Airfoil("n63428").set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3),
            asb.Airfoil("n63418").set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3),
            asb.Airfoil("n63412").set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3),
            asb.Airfoil("n63408").set_TE_thickness(0.0).to_kulfan_airfoil().set_TE_thickness(2.4e-3),
        ]

        # 预计算展向 B 样条插值基函数矩阵的逆矩阵，避免在符号图中重复求逆
        self.sp = CoxDeBoorBasis(ncpts=len(self.af_list), degree=3)
        u_stator = (self.stator - self.r_min) / max(self.r_max - self.r_min, 1e-9)
        w_dm = self.sp.get_basis(u_stator).T
        self.w_inv = ca.DM(np.linalg.inv(np.array(w_dm, dtype=float)))

    def __call__(self, Alpha=None, Reynold=None, Mach=None, r=None, Rtip=None, cst_upper=None, cst_lower=None, cst_le=None):
        rr = r / Rtip
        u_rr = (rr - self.r_min) / max(self.r_max - self.r_min, 1e-9)

        cst_list = []
        for i in range(len(self.af_list)):
            cu = self.af_list[i].upper_weights * (1.0 + cst_upper[:, i])
            cl = self.af_list[i].lower_weights * (1.0 + cst_lower[:, i])
            cle = self.af_list[i].leading_edge_weight * (1.0 + cst_le[0, i])
            cst_list.append(ca.vertcat(cu, cl, cle))
        cst_mat = ca.hcat(cst_list).T

        P = ca.mtimes(self.w_inv, cst_mat)
        weights = self.sp.get_basis(u_rr).T
        cst = ca.mtimes(weights, P)

        # 转置为 (8, 1) 列向量，确保与 AeroSandbox / NeuralFoil 的 CasADi 符号向量拼接维度一致
        af = asb.KulfanAirfoil(
            upper_weights=cst[0, :8].T,
            lower_weights=cst[0, 8:16].T,
            leading_edge_weight=cst[0, 16],
            TE_thickness=2.4e-3,
        )
        aero = af.get_aero_from_neuralfoil(alpha=Alpha, Re=Reynold, mach=Mach, model_size="xxsmall")
        CL = aero["CL"]
        CD = aero["CD"]

        return CL, CD


N = 20
Rtip = 2.1
r = anp.cosspace(0.1, 1.0, N)
stator_r = np.array([0.10, 0.25, 0.50, 0.75, 1.00], dtype=float)
af_model = AirfoilModel(stator=stator_r)
rotor = Blade(Rtip=Rtip, radius_resolution=r, Nb=3, afmodel=af_model)


opti = Opti()
# 使用 np.zeros 而非 np.zeros_like((8, 5))，确保变量维度为 (8, 5) 与 (1, 5)
cst_upper = opti.variable(init_guess=np.zeros((8, 5)), lower_bound=-0.2, upper_bound=0.2)
cst_lower = opti.variable(init_guess=np.zeros((8, 5)), lower_bound=-0.2, upper_bound=0.2)
cst_le = opti.variable(init_guess=np.zeros((1, 5)), lower_bound=-0.2, upper_bound=0.2)

af_list = af_model.af_list
af_list_new = [
    asb.KulfanAirfoil(
        lower_weights=af_list[i].lower_weights * (1.0 + cst_lower[:, i]),
        upper_weights=af_list[i].upper_weights * (1.0 + cst_upper[:, i]),
        leading_edge_weight=af_list[i].leading_edge_weight * (1.0 + cst_le[0, i]),
        TE_thickness=2.4e-3,
    )
    for i in range(len(af_list))
]

# 在内部弦向站位 [0.02, 0.98] 上采样厚度约束，避免 x/c=0 处 0>=0 恒等式导致 IPOPT 雅可比矩阵奇异
x_thick_check = np.linspace(0.02, 0.98, 50)
opti.subject_to(
    [
        af_list_new[0].max_thickness(x_over_c=x_thick_check) >= 0.28,
        af_list_new[1].max_thickness(x_over_c=x_thick_check) >= 0.28,
        af_list_new[2].max_thickness(x_over_c=x_thick_check) >= 0.18,
        af_list_new[3].max_thickness(x_over_c=x_thick_check) >= 0.12,
        af_list_new[4].max_thickness(x_over_c=x_thick_check) >= 0.08,
        af_list_new[0].local_thickness(x_over_c=x_thick_check) >= 0.0,
        af_list_new[1].local_thickness(x_over_c=x_thick_check) >= 0.0,
        af_list_new[2].local_thickness(x_over_c=x_thick_check) >= 0.0,
        af_list_new[3].local_thickness(x_over_c=x_thick_check) >= 0.0,
        af_list_new[4].local_thickness(x_over_c=x_thick_check) >= 0.0,
    ]
)

twist_cpts = opti.variable(
    init_guess=np.array([[0.0, 20.0], [0.75, 10.0], [1.0, 0.0]]),
    lower_bound=np.array([[0.0, 0.0], [0.2, -20.0], [1.0, -20.0]]),
    upper_bound=np.array([[0.0, 45.0], [0.8, 45.0], [1.0, 45.0]]),
)

chord_cpts = opti.variable(
    init_guess=np.array([[0.0, 0.315], [0.75, 0.30], [1.0, 0.20]]),
    lower_bound=np.array([[0.0, 0.30], [0.2, 0.20], [1.0, 0.15]]),
    upper_bound=np.array([[0.0, 0.35], [0.8, 0.50], [1.0, 0.20]]),
)

vinf_hover = 3.0
vinf_cruise = 400.0 / 3.6

pitch_hover = opti.variable(init_guess=0.0, lower_bound=-20.0, upper_bound=45.0)
pitch_cruise = opti.variable(init_guess=30.0, lower_bound=-20.0, upper_bound=50.0)

basis_geom = CoxDeBoorBasis(ncpts=3, degree=2)
chord_list = basis_geom.get_xy_from_x(r, chord_cpts)[:, 1].T
twist_list = basis_geom.get_xy_from_x(r, twist_cpts)[:, 1].T

omega_hover = 231.0 / Rtip
omega_cruise = 201.0 / Rtip

aero_hover = rotor.solve(
    Vx=vinf_hover,
    omega=omega_hover,
    chord=chord_list,
    twist=twist_list / 180.0 * np.pi,
    pitch=pitch_hover / 180.0 * np.pi,
    rho=1.225,
    mu=1.789e-5,
    sos=340.0,
    cst_upper=cst_upper,
    cst_lower=cst_lower,
    cst_le=cst_le,
)
aero_cruise = rotor.solve(
    Vx=vinf_cruise,
    omega=omega_cruise,
    chord=chord_list,
    twist=twist_list / 180.0 * np.pi,
    pitch=pitch_cruise / 180.0 * np.pi,
    rho=0.777,
    mu=1.64e-5,
    sos=322.6,
    cst_upper=cst_upper,
    cst_lower=cst_lower,
    cst_le=cst_le,
)


T_hover = aero_hover["T"]
P_hover = aero_hover["P"]

T_cruise = aero_cruise["T"]
P_cruise = aero_cruise["P"]

# 对大数量级的推力与功率约束进行无量纲化缩放，改善 IPOPT 求解收敛性
opti.subject_to(
    [
        T_cruise / 850.0 >= 1.0,
        T_hover / 15272.0 >= 1.0,
        P_hover / 450e3 <= 1.0,
    ]
)

opti.minimize(P_cruise / 1e5)

opti.ipopt_solver(mu_strategy="monotone", start_with_resto="yes")
sol = opti.solve()

np.set_printoptions(suppress=True)
print("Hover Thrust [N]:", sol.value(T_hover))
print("Hover Power [kW]:", sol.value(P_hover) / 1e3)
print("Cruise Thrust [N]:", sol.value(T_cruise))
print("Cruise Power [kW]:", sol.value(P_cruise) / 1e3)
print("Hover Pitch [deg]:", sol.value(pitch_hover))
print("Cruise Pitch [deg]:", sol.value(pitch_cruise))
print("Chord Control Points:\n", sol.value(chord_cpts))
print("Twist Control Points:\n", sol.value(twist_cpts))
