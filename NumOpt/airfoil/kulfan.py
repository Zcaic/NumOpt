import aerosandbox as asb
import aerosandbox.numpy as anp
from ..opti import cas
import numpy as np
from scipy.special import comb


class KulfanAirfoil:
    def __init__(self, Au, Al, N1=0.5, N2=1.0, Le=0.0, Te=0.0):
        self.Au = Au
        self.Al = Al
        self.N1 = N1
        self.N2 = N2
        self.Le = Le
        self.Te = Te

    def upper_coordinates(self, x):
        C = KulfanAirfoil.class_function(x, self.N1, self.N2)
        S = KulfanAirfoil.shape_function(x, self.Au)
        y = C * (S + self.Le * x**0.5 * (1 - x) ** (self.Au.shape[0] - 1.5)) + x * self.Te / 2.0

        coords = cas.hcat([x, y])
        return coords[::-1, :]

    def lower_coordinates(self, x):
        C = KulfanAirfoil.class_function(x, self.N1, self.N2)
        S = KulfanAirfoil.shape_function(x, self.Al)
        y = C * (S + self.Le * x**0.5 * (1 - x) ** (self.Al.shape[0] - 1.5)) - x * self.Te / 2.0

        coords = cas.hcat([x, y])
        return coords

    def coordinates(self, x):
        coordinates_upper = self.upper_coordinates(x)
        coordinates_lower = self.lower_coordinates(x)

        coordinates = cas.vcat([coordinates_upper[:-1, :], coordinates_lower])
        return coordinates

    @staticmethod
    def class_function(x, N1, N2):
        C = (x) ** N1 * (1 - x) ** N2
        return C

    @staticmethod
    def shape_function(x, w):
        N = w.shape[0] - 1  # Order of Bernstein polynomials

        K = comb(N, np.arange(N + 1))  # Bernstein polynomial coefficients

        B = []
        for i in range(w.shape[0]):
            tmp = K[i] * (x) ** i * (1 - x) ** (N - i)
            B.append(tmp)
        B = cas.hcat(B)
        S = cas.mtimes(B, w)

        return S

    @staticmethod
    def fit(upper_coordinates, lower_coordinates, nAu=8, nAl=8, N1=0.5, N2=1.0, symmetry=False):
        if not symmetry:
            opti = cas.Opti()
            Al = opti.variable(nAl)
            Au = opti.variable(nAu)
            leading_edge_weight = opti.variable(1)
            # N1 = opti.variable(1)
            # N2 = opti.variable(1)
            N1 = N1
            N2 = N2
            te = upper_coordinates[-1, 1] - lower_coordinates[-1, 1]

            xu = upper_coordinates[::-1, 0]
            yu = upper_coordinates[:, 1]
            xl = lower_coordinates[:, 0]
            yl = lower_coordinates[:, 1]

            af = KulfanAirfoil(Au=Au, Al=Al, N1=N1, N2=N2, Le=leading_edge_weight, Te=te)
            yu_ = af.upper_coordinates(xu)
            yl_ = af.lower_coordinates(xl)

            residual = cas.sum((yu_ - yu) ** 2) + cas.sum((yl_ - yl) ** 2)

            default_options = {
                "ipopt.sb": "yes",
                "ipopt.max_iter": 1000,
                "ipopt.max_cpu_time": 1e20,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.fast_step_computation": "yes",
                "detect_simple_bounds": False,
                "expand": True,
                "print_time": False,
                "ipopt.print_level": 0,
            }

            opti.minimize(residual)
            opti.subject_to(
                [
                    opti.bounded(-1.0, leading_edge_weight, 1.0),
                ]
            )

            opti.solver("ipopt", default_options)
            opti.set_initial(Au, 0.05)
            opti.set_initial(Al, -0.05)
            opti.set_initial(leading_edge_weight, 0.0)
            # opti.set_value(N1,0.5)
            # opti.set_value(N2,1.0)
            # opti.set_initial(N1, 0.5)
            # opti.set_initial(N2, 1.0)
            sol = opti.solve()

            Au_sol = sol.value(Au)
            Al_sol = sol.value(Al)
            le_sol = sol.value(leading_edge_weight)

        else:
            opti = cas.Opti()
            Al = opti.variable(nAl)
            Au = opti.variable(nAu)
            leading_edge_weight = 0.0
            # N1 = opti.variable(1)
            # N2 = opti.variable(1)
            N1 = N1
            N2 = N2
            te = upper_coordinates[-1, 1] - lower_coordinates[-1, 1]

            xu = upper_coordinates[::-1, 0]
            yu = upper_coordinates[:, 1]
            xl = lower_coordinates[:, 0]
            yl = lower_coordinates[:, 1]

            af = KulfanAirfoil(Au=Au, Al=Al, N1=N1, N2=N2, Le=leading_edge_weight, Te=te)
            yu_ = af.upper_coordinates(xu)
            yl_ = af.lower_coordinates(xl)

            residual = cas.sum((yu_ - yu) ** 2)

            default_options = {
                "ipopt.sb": "yes",
                "ipopt.max_iter": 1000,
                "ipopt.max_cpu_time": 1e20,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.fast_step_computation": "yes",
                "detect_simple_bounds": False,
                "expand": True,
                "print_time": False,
                "ipopt.print_level": 0,
            }

            opti.minimize(residual)

            opti.solver("ipopt", default_options)
            opti.set_initial(Au, 0.05)
            opti.set_initial(Al, -0.05)

            sol = opti.solve()

            Au_sol = sol.value(Au)
            Al_sol = -Au_sol
            le_sol = leading_edge_weight

        return KulfanAirfoil(Au=Au_sol, Al=Al_sol, N1=N1, N2=N2, Le=le_sol, Te=te)
