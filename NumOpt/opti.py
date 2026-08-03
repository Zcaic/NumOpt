# import aerosandbox.numpy as anp
# import aerosandbox as asb
# import numpy as np
# import casadi as cas
# from typing import Callable, Any, Dict,Literal
# from .cprint import cprint_yellow

# def trape(y, x):
#     y = anp.array(y)
#     x = anp.array(x)
#     mid_y = (y[:-1] + y[1:]) / 2.0
#     dx = anp.diff(x)

#     I = anp.sum(mid_y * dx)
#     return I


# class Opti(asb.Opti):
#     def ipopt_solver(
#         self,
#         max_iter: int = 1000,
#         max_runtime: float = 1e20,
#         callback: Callable[[int], Any] = None,
#         verbose: bool = True,
#         jit: bool = False,
#         detect_simple_bounds: bool = False,
#         expand: bool = True,
#         mu_strategy:Literal["monotone","adaptive"]="adaptive",
#         start_with_resto:Literal["yes","no"]="no",
#         options: Dict = None,
#     ):
#         if options is None:
#             options = {}
#         default_options = {
#             "ipopt.sb": "yes",  # Hide the IPOPT banner.
#             "ipopt.max_iter": max_iter,
#             "ipopt.max_cpu_time": max_runtime,
#             "ipopt.mu_strategy": "adaptive",
#             "ipopt.fast_step_computation": "yes",
#             "detect_simple_bounds": detect_simple_bounds,
#             "expand": expand,
#             "ipopt.mu_strategy": mu_strategy,
#             "ipopt.start_with_resto": start_with_resto
#         }
#         if jit:
#             default_options["jit"] = True
#             # options["compiler"] = "shell"  # Recommended by CasADi devs, but doesn't work on my machine
#             default_options["jit_options"] = {
#                 "flags": ["-O3"],
#                 # "verbose": True
#             }

#         if verbose:
#             default_options["ipopt.print_level"] = 5  # Verbose, per-iteration printing.
#         else:
#             default_options["print_time"] = False  # No time printing
#             default_options["ipopt.print_level"] = 0  # No printing from IPOPT

#         super().solver(
#             "ipopt",
#             {
#                 **default_options,
#                 **options,
#             },
#         )
#         if callback is not None:
#             self.callback(callback)

#     def solve(
#         self,
#         behavior_on_failure: str = "raise",
#     ):
#         if behavior_on_failure == "raise":
#             sol = asb.OptiSol(opti=self, cas_optisol=cas.Opti.solve(self))
#         elif behavior_on_failure == "return_last":
#             try:
#                 sol = asb.OptiSol(opti=self, cas_optisol=cas.Opti.solve(self))
#             except RuntimeError:
#                 import warnings

#                 warnings.warn("Optimization failed. Returning last solution.")

#                 sol = asb.OptiSol(opti=self, cas_optisol=self.debug)

#         if self.save_to_cache_on_solve:
#             self.save_solution()

#         return sol


import casadi as ca
from typing import Callable, Literal, Dict, Any
import numpy as np


class Opti(ca.Opti):
    def variable(self, init_guess, scale=1.0, lower_bound=None, upper_bound=None) -> ca.MX:
        init_guess = np.atleast_2d(init_guess)
        shape = init_guess.shape
        var = scale * super().variable(*shape)
        self.set_initial(var, init_guess)

        if lower_bound is not None:
            # if not np.shape(lower_bound):
            #     lower_bound=np.full(shape,lower_bound)
            self.subject_to(ca.vec(var / scale) >= ca.vec(lower_bound / scale))
        if upper_bound is not None:
            # if not np.shape(upper_bound):
            #     upper_bound=np.full(shape,upper_bound)
            self.subject_to(ca.vec(var / scale) <= ca.vec(upper_bound / scale))

        return var

    def parameter(self, value):
        value=np.atleast_2d(value)
        shape = value.shape
        param = super().parameter(*shape)
        self.set_value(param, value)
        return param

    def minimize(
        self,
        f: ca.MX,
    ) -> None:
        super().minimize(f)

    def maximize(self, f: ca.MX) -> None:
        super().minimize(-1 * f)

    def ipopt_solver(
        self,
        max_iter: int = 1000,
        max_runtime: float = 1e20,
        callback: Callable[[int], Any] = None,
        verbose: bool = True,
        jit: bool = False,
        detect_simple_bounds: bool = False,
        expand: bool = True,
        mu_strategy: Literal["monotone", "adaptive"] = "adaptive",
        start_with_resto: Literal["yes", "no"] = "no",
        options: Dict = None,
    ):
        if options is None:
            options = {}
        default_options = {
            "ipopt.sb": "yes",  # Hide the IPOPT banner.
            "ipopt.max_iter": max_iter,
            "ipopt.max_cpu_time": max_runtime,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.fast_step_computation": "yes",
            "detect_simple_bounds": detect_simple_bounds,
            "expand": expand,
            "ipopt.mu_strategy": mu_strategy,
            "ipopt.start_with_resto": start_with_resto,
        }
        if jit:
            default_options["jit"] = True
            # options["compiler"] = "shell"  # Recommended by CasADi devs, but doesn't work on my machine
            default_options["jit_options"] = {
                "flags": ["-O3"],
                # "verbose": True
            }

        if verbose:
            default_options["ipopt.print_level"] = 5  # Verbose, per-iteration printing.
        else:
            default_options["print_time"] = False  # No time printing
            default_options["ipopt.print_level"] = 0  # No printing from IPOPT

        super().solver(
            "ipopt",
            {
                **default_options,
                **options,
            },
        )
        if callback is not None:
            self.callback(callback)


if __name__ == "__main__":
    # cprint_yellow("it is ok")
    ...
