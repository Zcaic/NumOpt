import aerosandbox as asb
import aerosandbox.numpy as anp
import casadi as ca
from termcolor import cprint


class Blade:
    def __init__(
        self, airfoil: asb.KulfanAirfoil, radius: float, R: float, Rn: float, b, theta: float, Nb: int, opti: asb.Opti
    ):
        """blade element class

        Parameters
        ----------
        airfoil : asb.KulfanAirfoil
            CST airfoil
        radius : float
            the radius of the blade
        R :float
            the radius of the paddle disk
        Rn : float
            the radius of the hub
        b : float [m]
            the chord of the blade
        theta : float [rad]
            Mounting angle of the blade
        V0 : float [m/s]
            Incoming flow speed
        ns : float [rad/s]
            rotational speed
        Nb : int
            Number of propeller blades
        atmos : asb.Atmosphere
            Atmospheric environment
        opti : opti:asb.Opti

        """
        self.airfoil = airfoil
        self.radius = radius
        self.R = R
        self.Rn = Rn
        self.b = b
        self.theta = theta
        self.Nb = Nb
        self.opti = opti

    def run(self, V0: float, ns: float, atmos: asb.Atmosphere):
        Va = self.opti.variable(init_guess=1.0, n_vars=1,freeze=False)
        Vt = self.opti.variable(init_guess=1.0, n_vars=1,freeze=False)

        V_tau = 2 * anp.pi * ns * self.radius
        fai0 = anp.arctan(V0 / V_tau)
        fai = anp.arctan((V0 + Va) / (V_tau - Vt))
        beta = fai - fai0
        rho = atmos.density()
        r = self.radius

        # Momentum Theorem
        dT_dr_Momentum = 4 * anp.pi * r * rho * (V0 + Va) * Va
        dM_dr_Momentum = 4 * anp.pi * r**2 * rho * (V0 + Va) * Vt

        # blade element
        W = anp.sqrt((V0 + Va) ** 2 + (V_tau - Vt) ** 2)
        sos = atmos.speed_of_sound()
        alpha = self.theta - fai
        Ma = W / sos
        Re = rho * W * self.b / atmos.dynamic_viscosity()
        aero = self.get_aerodynamic(self.airfoil, alpha=alpha, Mach=Ma, Re=Re, method=1)
        CL = aero["CL"]
        CD = aero["CD"]
        q = 0.5 * rho * W**2 * self.b
        dL_dr = q * CL
        dD_dr = q * CD
        # gam = anp.arctan(dD_dr / dL_dr)
        # dR = anp.sqrt(dL_dr**2 + dD_dr**2)
        # dT_dr_blade = dR * anp.cos(fai + gam)
        # dF_dr_blade = dR * anp.sin(fai + gam)
        # dM_dr_blade = dF_dr_blade * r
        dT_dr_blade = dL_dr * anp.cos(fai) - dD_dr * anp.sin(fai)
        dF_dr_blade = dL_dr * anp.sin(fai) + dD_dr * anp.cos(fai)
        dM_dr_blade = dF_dr_blade * r

        correction_coefficient = self.correct(fai)
        # dT_dr_Momentum_correct = dT_dr_Momentum * correction_coefficient
        # dM_dr_Momentum_correct = dM_dr_Momentum * correction_coefficient

        self.opti.subject_to(
            [
                dT_dr_Momentum * correction_coefficient == dT_dr_blade,
                dM_dr_Momentum * correction_coefficient == dM_dr_blade,
            ]
        )

        return {
            "T": dT_dr_blade,
            "M": dM_dr_blade,
            "F": dF_dr_blade,
            "AoA": alpha,
            "Beta": beta,
            "Va": Va,
            "Vt": Vt,
            "CL": CL,
            "CD": CD,
            "fai":fai,
            "V_tau":V_tau,
            "Re":Re,
            "Ma":Ma,
            "W":W,
            "T_":dT_dr_Momentum * correction_coefficient,
            "M_":dT_dr_Momentum * correction_coefficient
        }

    def correct(
        self,
        fai,
    ):
        # c1=0.125
        # c2=21
        # g=anp.exp(-c1*(self.Nb))
        Ft = 2 / anp.pi * anp.arccos(anp.exp(-(self.Nb * (self.R - self.radius)) / (2 * self.radius * anp.sin(fai))))
        Fr = 2 / anp.pi * anp.arccos(anp.exp(-(self.Nb * (self.radius - self.Rn) / (2 * self.radius * anp.sin(fai)))))
        F = Ft * Fr
        return F

    @staticmethod
    def get_aerodynamic(airfoil: asb.KulfanAirfoil, alpha: float, Mach: float, Re: float, method=1):
        if method == 1:
            aero = airfoil.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=Mach,model_size="xxxlarge")
            CL = aero["CL"]
            CD = aero["CD"]
            CM = aero["CM"]
        else:
            cprint("the other method is not completed.", "red", attrs=["bold"])
            exit(-6)
        res = {"CL": CL, "CD": CD, "CM": CM}
        return res


class Rotor:
    def __init__(self, R: float, Rn: float, Nb: int, opti: asb.Opti = None):
        """Rotor class

        Parameters
        ----------
        R : float
            the radius of the rotor
        Rn :float
            the radius of the hub
        Nb : int
            the number of blades
        v0 : float
            the freestream velocity
        opti : asb.Opti
        """
        self.R = R
        self.Rn = Rn
        self.Nb = Nb
        # self.r = r
        self.blades: list[Blade] = []
        self.opti = asb.Opti() if opti is None else opti

    def add_blade(self, airfoil: asb.KulfanAirfoil, r: float, chord: float, theta: float):
        """add a blade element to rotor

        Parameters
        ----------
        airfoil : asb.KulfanAirfoil
            the airfoil of blade
        r : float
            the radial position of airfoil in [0,1]
        chord : float
            the chord of the airfoil
        theta : float
            the twist angle of the airfoil
        """
        blade = Blade(
            airfoil=airfoil, radius=r * self.R, R=self.R, Rn=self.Rn, b=chord, theta=theta, Nb=self.Nb, opti=self.opti
        )
        self.blades.append(blade)

    def get_aero(self, V0: float, ns: float, atmos: asb.Atmosphere):
        ns = ns / (2 * anp.pi)
        rho = atmos.density()

        nblades = len(self.blades)

        T = [0.0] * nblades
        F = [0.0] * nblades
        M = [0.0] * nblades
        AoA = [0.0] * nblades
        beta = [0.0] * nblades
        Va = [0.0] * nblades
        Vt = [0.0] * nblades
        CL = [0.0] * nblades
        CD = [0.0] * nblades
        fai=[0.0]*nblades
        V_tau=[0.0]*nblades
        Re=[0.0]*nblades
        Ma=[0.0]*nblades
        W=[0.0]*nblades
        T_=[0.0]*nblades
        M_=[0.0]*nblades

        rlist = [ir.radius for ir in self.blades]
        drs = anp.diff(rlist).tolist()
        drs = [0.0] + drs + [0.0]
        drs = anp.array(drs)

        for i, iv in enumerate(self.blades):
            res_aero = iv.run(V0, ns, atmos)
            dr = (drs[i] + drs[i + 1]) / 2.0
            T[i] = res_aero["T"] * dr
            M[i] = res_aero["M"] * dr
            F[i] = res_aero["F"] * dr
            AoA[i] = res_aero["AoA"]
            beta[i] = res_aero["Beta"]
            Va[i] = res_aero["Va"]
            Vt[i] = res_aero["Vt"]
            CD[i] = res_aero["CD"]
            CL[i] = res_aero["CL"]
            fai[i]=res_aero["fai"]
            V_tau[i]=res_aero["V_tau"]
            Re[i]=res_aero["Re"]
            Ma[i]=res_aero["Ma"]
            W[i]=res_aero["W"]
            T_[i]=res_aero["T_"]*dr
            M_[i]=res_aero["M_"]*dr

        T = anp.stack(T)
        F = anp.stack(F)
        M = anp.stack(M)
        AoA = anp.stack(AoA)
        beta = anp.stack(beta)
        Va = anp.stack(Va)
        Vt = anp.stack(Vt)

        Ttot = self.Nb * anp.sum(T)
        Ftot = self.Nb * anp.sum(F)
        Mtot = self.Nb * anp.sum(M)

        P_in = 2 * anp.pi * ns * Mtot
        A = anp.pi * self.R**2
        vi = anp.sqrt(Ttot / (2 * rho * A))
        P_out = Ttot * (V0 + vi)

        eta = P_out / P_in

        # sol = self.opti.solve(options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"})

        # cprint(anp.rad2deg(sol(AoA)), color="blue")
        # cprint(anp.rad2deg(sol(beta)), color="yellow")

        return {
            "Ttot": Ttot,
            "Ftot": Ftot,
            "Mtot": Mtot,
            "eta": eta,
            "AoA": AoA,
            "Beta": beta,
            "Va": Va,
            "Vt": Vt,
            "T": T,
            "F": F,
            "M": M,
            "CL": CL,
            "CD": CD,
            "fai":fai,
            "Vtau":V_tau,
            "Re":Re,
            "Ma":Ma,
            "W":W,
            "T_":T_,
            "M_":M_
        }

    def solve(self):
        sol = self.opti.solve(verbose=True,options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"})
        return sol


if __name__ == "__main__":
    rotor = Rotor(R=3, Rn=0.0,Nb=3)

    af = asb.KulfanAirfoil("naca0012").set_TE_thickness(0.0)
    rotor.add_blade(airfoil=af, r=0.1, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012").set_TE_thickness(0.0)
    rotor.add_blade(airfoil=af, r=0.25, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012").set_TE_thickness(0.0)
    rotor.add_blade(airfoil=af, r=0.5, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012").set_TE_thickness(0.0)
    rotor.add_blade(airfoil=af, r=0.75, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012").set_TE_thickness(0.0)
    rotor.add_blade(airfoil=af, r=0.99, chord=0.453, theta=anp.deg2rad(20))

    aero = rotor.get_aero(V0=60, ns=300 * 2 * anp.pi / 60, atmos=asb.Atmosphere(altitude=0.0))

    sol = rotor.solve()
    cprint(sol(aero), color="green", attrs=["bold"])
