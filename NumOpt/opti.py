import aerosandbox.numpy as anp
import aerosandbox as asb 
import numpy as np 
import casadi as cas
from typing import Callable, Any, Dict
from .cprint import cprint_yellow

# def trape(y, x):
#     y = anp.array(y)
#     x = anp.array(x)
#     mid_y = (y[:-1] + y[1:]) / 2.0
#     dx = anp.diff(x)

#     I = anp.sum(mid_y * dx)
#     return I


class Opti(asb.Opti):
    def solver(
        self,
        max_iter: int = 1000,
        max_runtime: float = 1e20,
        callback: Callable[[int], Any] = None,
        verbose: bool = True,
        jit: bool = False,
        detect_simple_bounds: bool = False,
        expand: bool = True,
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

    def solve(
        self,
        behavior_on_failure: str = "raise",
    ):
        if behavior_on_failure == "raise":
            sol = asb.OptiSol(opti=self, cas_optisol=cas.Opti.solve(self))
        elif behavior_on_failure == "return_last":
            try:
                sol = asb.OptiSol(opti=self, cas_optisol=cas.Opti.solve(self))
            except RuntimeError:
                import warnings

                warnings.warn("Optimization failed. Returning last solution.")

                sol = asb.OptiSol(opti=self, cas_optisol=self.debug)

        if self.save_to_cache_on_solve:
            self.save_solution()

        return sol


if __name__=="__main__":
    cprint_yellow("it is ok")