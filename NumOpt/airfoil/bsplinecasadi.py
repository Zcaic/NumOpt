import casadi as ca
from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


class BsplineCasadi(ca.Callback):
    def __init__(self, name, cpts: np.ndarray, degree=3, extrapolate=False, opts={"enable_fd": False}):
        ca.Callback.__init__(self)
        self.construct(name, opts)
        self.cpts = cpts
        self.degree = degree
        self.knots = self.quasi_uniform_knots()
        self._sp = BSpline.construct_fast(t=self.knots, c=self.cpts, k=self.degree, extrapolate=extrapolate)
        self._sp_d1 = self._sp.derivative(1)
        self._sp_d2 = self._sp.derivative(2)

        self.get_xy_from_x=self.gen_get_y_from_x()

    def quasi_uniform_knots(self):
        middle = np.linspace(0, 1, self.cpts.shape[0] - self.degree + 1)
        start = np.zeros(self.degree, dtype="f8")
        end = np.ones(self.degree, dtype="f8")
        knots = np.hstack([start, middle, end])
        return knots

    def get_n_in(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(1, 1)

    def get_n_out(self):
        return 1

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(2,1)

    def eval(self, args):
        u = args[0]
        u = u.toarray().ravel()
        xy = self._sp(u).reshape((-1,1))
        return [xy]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        class JacFun(ca.Callback):
            def __init__(self_jac, opts={"enable_fd": False}):
                ca.Callback.__init__(self_jac)
                self_jac.construct(name, opts)

            def get_n_in(self_jac):
                return 2

            def get_n_out(self_jac):
                return 1

            def get_sparsity_in(self_jac, i):
                if i == 0:
                    return ca.Sparsity.dense(1, 1)
                elif i == 1:
                    return ca.Sparsity.dense(2,1)

            def get_sparsity_out(self_jac, i):
                return ca.Sparsity.dense(2,1)

            def eval(self_jac, args):
                u = args[0]
                u = u.toarray().ravel()
                jac = self._sp_d1(u).reshape((-1,1))
                return [jac]

            def has_jacobian(self_jac):
                return True

            def get_jacobian(self_jac, name, inames, onames, opts):
                class HessFun(ca.Callback):
                    def __init__(self_hess, opts={}):
                        ca.Callback.__init__(self_hess)
                        self_hess.construct(name, opts)

                    def get_n_in(self_hess):
                        return 3

                    def get_n_out(self_hess):
                        return 2

                    def get_sparsity_in(self_hess, i):
                        if i == 0:
                            return ca.Sparsity.dense(1, 1)
                        elif i == 1:
                            return ca.Sparsity.dense(2,1)
                        elif i == 2:
                            return ca.Sparsity.dense(2,1)

                    def get_sparsity_out(self_hess, i):
                        if i == 0:
                            return ca.Sparsity.dense(2,1)
                        elif i == 1:
                            return ca.Sparsity.dense(2,2)

                    def eval(self_hess, args):
                        u = args[0]
                        u = u.toarray().ravel()
                        hess_u = self._sp_d2(u).reshape((-1,1))
                        hess_xy = ca.GenDM_zeros(2,2)
                        return [hess_u, hess_xy]

                self_jac.jac_callback = HessFun()
                return self_jac.jac_callback

        self.jac_callback = JacFun()
        return self.jac_callback

    def gen_get_y_from_x(self):
        x=ca.SX.sym("x")
        u_star=self.__get_y_from_x()(0.5,x)
        xy_star=self(u_star)
        
        func=ca.Function("get_y_from_x",[x],[xy_star])
        return func


    def __get_y_from_x(self):
        u=ca.SX.sym("u")
        x=ca.SX.sym("x")
        xy=self(u)
        F1=xy[0,0]-x 

        g=ca.Function("g",[u,x],[F1])
        sol=ca.rootfinder("G","kinsol",g)
        return sol        


if __name__=="__main__":
    cpts = np.array([[0.0, 0.0], [0.5, 0.5], [0.6, 1.2], [2.0, 0.0]])
    sp = BsplineCasadi("sp", cpts=cpts, degree=3)
    # print(sp(0.5))

    u = np.linspace(0, 1, 1000)
    dumpy_x = ca.SX.sym("x")
    sp_func = ca.Function("sp_func", [dumpy_x], [sp(dumpy_x)])
    sp_func_jac=ca.Function("sp_func",[dumpy_x],[ca.jacobian(sp(dumpy_x),dumpy_x)])
    sp_func_hess=ca.Function("sp_func",[dumpy_x],[ca.hessian(sp(dumpy_x)[1,0],dumpy_x)[0]])

    print(sp_func_jac(0.5))
    print(sp_func_hess(0.5))
    sp_func_mpi = sp_func.map(u.shape[0])

    xy_list = sp_func_mpi(u).T


    # from NumOpt import Opti
    # opti=Opti()
    # u=opti.variable(init_guess=0.5,lower_bound=0.0,upper_bound=1.0)
    # xy=sp(u).T
    # opti.subject_to([
    #     xy[0,0]==0.6625
    # ])
    # opti.ipopt_solver()
    # sol=opti.solve()
    # print(sol(xy))

    print(sp.get_xy_from_x(0.6625))

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
            ax.plot(xy_list[:, 0], xy_list[:, 1], label="Bspline")
            ax.plot(cpts[:, 0], cpts[:, 1], "--o", label="cpts")

            ax.legend(fontsize=15)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.show()