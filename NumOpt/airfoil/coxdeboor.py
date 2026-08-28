import casadi as ca
import numpy as np


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
        u_val = u_val.reshape((1, -1))
        N = self._func_get_basis(u_val)
        return N

    def get_curve_point(self, cpts, u_val):
        weights = self.get_basis(u_val)
        ret = ca.mtimes(weights.T, cpts)
        return ret


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import scienceplots

    cpts = np.array([[0.0, 0.0], [0.5, 30.0], [0.6, 20.0], [1.0, 5.0]])
    sp = CoxDeBoorBasis(ncpts=cpts.shape[0], degree=3)

    u = np.linspace(-1, 2.0, 100)

    xy = sp.get_curve_point(cpts=cpts, u_val=u)

    print(xy[-1, :])

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
            fig = plt.figure(figsize=(6, 5))

            ax = fig.add_subplot(111)
            ax.plot(xy[:, 0], xy[:, 1], label="Bspline")
            ax.plot(cpts[:, 0], cpts[:, 1], "--o", markersize=10, label="cpts")

            ax.legend(fontsize=15)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.show()