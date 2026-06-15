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

    def setup(self, nprocess: int, mpi_mothod: str = "thread"):
        self.FUNC_blade = self.__run_blade()
        self.FUNC_blade_mpi = self.FUNC_blade.map(nprocess, mpi_mothod)

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

    @staticmethod
    def __run_blade():

        opti = asb.Opti()
        # varibles
        Va = opti.variable(init_guess=1.0, n_vars=1)
        Vt = opti.variable(init_guess=1.0, n_vars=1)

        # parameters
        r = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        R = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        Rn = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        V0 = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        ns = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        rho = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        sos = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        viscosity = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        theta = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        b = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        Au = opti.variable(init_guess=1.0, n_vars=8, freeze=True)
        Al = opti.variable(init_guess=1.0, n_vars=8, freeze=True)
        Le = opti.variable(init_guess=1.0, n_vars=1, freeze=True)
        Nb = opti.variable(init_guess=1.0, n_vars=1, freeze=True)

        # compute
        V_tau = 2 * anp.pi * ns * r
        fai0 = anp.arctan(V0 / V_tau)
        fai = anp.arctan((V0 + Va) / (V_tau - Vt))
        beta = fai - fai0

        # Momentum Theorem
        dT_dr_Momentum = 4 * anp.pi * r * rho * (V0 + Va) * Va
        dM_dr_Momentum = 4 * anp.pi * r**2 * rho * (V0 + Va) * Vt

        # blade element
        W = anp.sqrt((V0 + Va) ** 2 + (V_tau - Vt) ** 2)
        alpha = theta - fai
        Ma = W / sos
        Re = rho * W * b / viscosity

        # af = asb.KulfanAirfoil(lower_weights=Al, upper_weights=Au, leading_edge_weight=Le,TE_thickness=0.0)
        af = asb.KulfanAirfoil("naca0012").set_TE_thickness(0.0)
        aero = af.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=Ma, model_size="xxxlarge")
        CL = aero["CL"]
        CD = aero["CD"]
        # CM=aero["CM"]

        q = 0.5 * rho * W**2 * b
        dL_dr = q * CL
        dD_dr = q * CD
        dT_dr_blade = dL_dr * anp.cos(fai) - dD_dr * anp.sin(fai)
        dF_dr_blade = dL_dr * anp.sin(fai) + dD_dr * anp.cos(fai)
        dM_dr_blade = dF_dr_blade * r

        # correct coeffience
        Ft = 2 / anp.pi * anp.arccos(anp.exp(-(Nb * (R - r)) / (2 * r * anp.sin(fai))))
        Fr = 2 / anp.pi * anp.arccos(anp.exp(-(Nb * (r - Rn) / (2 * r * anp.sin(fai)))))
        F = Ft * Fr
        dT_dr_Momentum_correct = dT_dr_Momentum * F
        dM_dr_Momentum_correct = dM_dr_Momentum * F

        opti.subject_to([dT_dr_blade == dT_dr_Momentum_correct, dM_dr_blade == dM_dr_Momentum_correct])

        opti.solve(verbose=False, dry_run=True)

        INPUTS = ca.vcat([r, R, Rn, V0, ns, rho, sos, viscosity, theta, b, Au, Al, Le, Nb])
        OUTPUS = ca.vcat(
            [
                dT_dr_blade,
                dM_dr_blade,
                dT_dr_Momentum_correct,
                dM_dr_Momentum_correct,
                alpha,
                beta,
                CL,
                CD,
                fai,
                V_tau,
                V0,
                Re,
                Ma,
                W,
                Va,
                Vt,
            ]
        )

        FUNC_blade = opti.to_function("FUNC_blade", [INPUTS], [OUTPUS])
        return FUNC_blade

    def get_aero(self, V0: float, ns: float, atmos: asb.Atmosphere):
        if not hasattr(self, "FUNC_blade_mpi"):
            cprint("run `setup` method first before call `get_aero` method.", "light_red", attrs=["bold"])
            exit(-1)
        ns = ns / (2 * anp.pi)
        rho = atmos.density()

        nblades = len(self.blades)
        # Vas = self.opti.variable(init_guess=1.0, n_vars=nblades,freeze=False)
        # Vts= self.opti.variable(init_guess=1.0, n_vars=nblades,freeze=False)
        rs = [i.radius for i in self.blades]
        Rs = [self.R] * nblades
        Rns = [self.Rn] * nblades
        V0s = [V0] * nblades
        nss = [ns] * nblades
        rho = atmos.density()
        rhos = [rho] * nblades
        sos = atmos.speed_of_sound()
        soss = [sos] * nblades
        viscosity = atmos.dynamic_viscosity()
        viscositys = [viscosity] * nblades
        Nbs = [self.Nb] * nblades

        thetas = []
        bs = []
        Aus = []
        Als = []
        Les = []
        for i in self.blades:
            af = i.airfoil
            thetas.append(i.theta)
            bs.append(i.b)
            Aus.append(af.upper_weights)
            Als.append(af.lower_weights)
            Les.append(af.leading_edge_weight)
        Aus = anp.array(Aus)
        Als = anp.array(Als)

        INPUTS = ca.hcat([rs, Rs, Rns, V0s, nss, rhos, soss, viscositys, thetas, bs, Aus, Als, Les, Nbs]).T
        OUTPUTS = self.FUNC_blade_mpi(INPUTS)
        # for i in range(nblades):
        #     self.opti.subject_to([OUTPUTS[0, i] == OUTPUTS[2, i], OUTPUTS[1, i] == OUTPUTS[3, i]])

        drs = anp.diff(rs).tolist()
        drs = [0.0] + drs + [0.0]
        drs = anp.array(drs)
        dr = (drs[:-1] + drs[1:]) / 2.0
        T = OUTPUTS[0, :].T * dr
        M = OUTPUTS[1, :].T * dr
        AoA = OUTPUTS[4, :]
        beta = OUTPUTS[5, :]
        CL = OUTPUTS[6, :]
        CD = OUTPUTS[7, :]

        Ttot = self.Nb * anp.sum(T)
        Mtot = self.Nb * anp.sum(M)

        P_in = 2 * anp.pi * ns * Mtot
        A = anp.pi * self.R**2
        vi = anp.sqrt(Ttot / (2 * rho * A))
        P_out = Ttot * (V0 + vi)

        eta = P_out / P_in

        return {
            "Ttot": Ttot,
            "Mtot": Mtot,
            "eta": eta,
            "AoA": AoA,
            "Beta": beta,
            "Va": OUTPUTS[-2, :],
            "Vt": OUTPUTS[-1, :],
            "T": T,
            "M": M,
            "CL": CL,
            "CD": CD,
            "fai": OUTPUTS[8, :],
            "Vtau": OUTPUTS[9, :],
            "V0": OUTPUTS[10, :],
            "Re": OUTPUTS[11, :],
            "Ma": OUTPUTS[12, :],
            "W": OUTPUTS[13, :],
            "T_": OUTPUTS[2, :],
            "M_": OUTPUTS[3, :],
        }

    def solve(self):
        sol = self.opti.solve(verbose=True, options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"})
        return sol


if __name__ == "__main__":
    rotor = Rotor(R=3, Rn=0.0, Nb=3)

    af = asb.KulfanAirfoil("naca0012")
    rotor.add_blade(airfoil=af, r=0.1, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012")
    rotor.add_blade(airfoil=af, r=0.25, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012")
    rotor.add_blade(airfoil=af, r=0.5, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012")
    rotor.add_blade(airfoil=af, r=0.75, chord=0.453, theta=anp.deg2rad(20))

    af = asb.KulfanAirfoil("naca0012")
    rotor.add_blade(airfoil=af, r=1.0, chord=0.453, theta=anp.deg2rad(20))

    rotor.setup(nprocess=5)
    aero = rotor.get_aero(V0=0.0, ns=300 * 2 * anp.pi / 60, atmos=asb.Atmosphere(altitude=0.0))

    sol = rotor.solve()
    cprint(sol(aero), color="green", attrs=["bold"])
