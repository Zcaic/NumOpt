import numpy as np
import casadi as ca
from NumOpt import Opti


def cosspace(start: float = 0.0, stop: float = 1.0, num: int = 50):
    mean = (stop + start) / 2
    amp = (stop - start) / 2
    ones = 0 * start + 1
    spaced_array = mean + amp * np.cos(np.linspace(np.pi * ones, 0 * ones, num))

    # Fix the endpoints, which might not be exactly right due to floating-point error.
    spaced_array[0] = start
    spaced_array[-1] = stop

    return spaced_array

class Bspline:
    def __init__(self, cpts, degree=3):
        self.cpts = cpts
        self.degree = degree
        self.knots = self.quasi_uniform_knots(self.cpts.shape[0],self.degree)
        self.__get_xy_from_u_func=self.__gen_get_xy_from_u()

    @staticmethod
    def quasi_uniform_knots(ncpts, degree):
        middle = np.linspace(0, 1, ncpts - degree + 1)
        start = np.zeros(degree, dtype="f8")
        end = np.ones(degree, dtype="f8")
        knots = np.hstack([start, middle, end])
        return knots

    @staticmethod
    def coxdeBoor(u, i, p, knots):
        if p == 0:
            return ca.if_else(ca.logic_and(u >= knots[i], u < knots[i + 1]), 1.0, 0.0)
        else:
            term1 = term2 = 0.0
            delta_t1 = knots[i + p] - knots[i]
            delta_t2 = knots[i + p + 1] - knots[i + 1]

            if delta_t1 != 0.0:
                term1 = (u - knots[i]) / (delta_t1) * Bspline.coxdeBoor(u, i, p - 1, knots)

            if delta_t2 != 0.0:
                term2 = (knots[i + p + 1] - u) / (delta_t2) * Bspline.coxdeBoor(u, i + 1, p - 1, knots)

            return term1 + term2

    def __gen_get_xy_from_u(self):
        u=ca.MX.sym("u")
        C1=0.0
        for i in range(self.cpts.shape[0]):
            N=Bspline.coxdeBoor(u,i,self.degree,self.knots)
            C1+=N*self.cpts[i,:]

        C=ca.if_else(u == 1.0, self.cpts[-1,:], C1)

        func=ca.Function("get_xy_from_u",[u],[C])
        return func
        
    def get_xy_from_u(self, u=cosspace(0, 1, 100), to_numpy=True):
        func_map=self.__get_xy_from_u_func.map(u.shape[0])
        xy=func_map(u).T
        if to_numpy:
            xy=xy.toarray()
    
        return xy
    
    def get_xy_from_x(self,x,):
        ...

    @staticmethod
    def fit(data, ncpts, degree):
        def gen_fun(nu,ncpts):
            u=ca.MX.sym("u")
            cpts=ca.MX.sym("cpts",ncpts,2)
            knots=Bspline.quasi_uniform_knots(ncpts,degree)
            C1=0.0
            for i in range(ncpts):
                N=Bspline.coxdeBoor(u,i,degree,knots)
                C1+=N*cpts[i,:]

            C=ca.if_else(u == 1.0, cpts[-1,:], C1)

            func=ca.Function("get_xy_from_u",[u,cpts],[C.T])

            func_map=func.map(nu,[False,True],[False],{})

            return func_map


        opti = Opti()
        cpts_x = opti.variable(init_guess=np.linspace(data[0, 0], data[-1, 0], ncpts), freeze=True)
        cpts_y = opti.variable(init_guess=np.linspace(data[0, 1], data[-1, 1], ncpts))
        u = opti.variable(init_guess=0.5, n_vars=data.shape[0], lower_bound=0.0, upper_bound=1.0)
        cpts = ca.horzcat(cpts_x, cpts_y)

        func_map=gen_fun(data.shape[0],ncpts)


        data_fit = func_map(u.T,cpts).T

        y_dist = data_fit[:, 1] - data[:, 1]
        dist = ca.sum(y_dist**2)

        opti.subject_to(
            [
                data_fit[:, 0] == data[:, 0],
            ]
        )

        opti.minimize(dist)
        opti.ipopt_solver()
        sol = opti.solve()

        cpts = sol(cpts)
        data_fit = sol(data_fit)
        sp = Bspline(cpts=sol(cpts), degree=degree)
        return sp


class BsplineAirfoil:
    def __init__(self, ctu=None, ctl=None, degree=3):
        self.ctu = ctu
        self.ctl = ctl

        self.bspline_upper = Bspline(ctrlpts=self.ctu, degree=degree)
        self.bspline_lower = Bspline(ctrlpts=self.ctl, degree=degree)

        self.__te = self.ctu[-1, 1] - self.ctl[-1, 1]
        self.__symmetry = False

    def upper_coordinates(self, t):
        pts = self.bspline_upper(t)[::-1, :]
        return pts

    def lower_coordinates(self, t):
        pts = self.bspline_lower(t)
        return pts

    def coordinates(self, t):
        pts_upper = self.upper_coordinates(t)
        pts_lower = self.lower_coordinates(t)

        pts = ca.vcat([pts_upper[:-1, :], pts_lower])
        return pts

    @property
    def te(self):
        return self.__te

    @property
    def symmetry(self):
        return self.__symmetry

    @staticmethod
    def fit(upper_coordinates, lower_coordinates, nctu=9, nctl=9, symmetry=False):
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

        # ============================ upper ==========================
        opti = ca.Opti()

        ctu = opti.variable(nctu, 2)
        ctu_init = np.zeros(ctu.shape)
        ctu_init[1:, 0] = np.linspace(0, 1, nctu - 1)
        ctu_init[1:-1, 1] = 0.5
        ctu_init[-1, 1] = upper_coordinates[0, 1]

        ctl = opti.variable(nctl, 2)

        t = opti.variable(upper_coordinates.shape[0])
        t_init = np.linspace(0, 1, upper_coordinates.shape[0], dtype="f8")

        af = BsplineAirfoil(ctu=ctu, ctl=ctl)
        coords = af.upper_coordinates(t)

        dist = coords - upper_coordinates
        residual = ca.sum(ca.dot(dist, dist))

        opti.subject_to(
            [
                opti.bounded(0.0, t, 1.0),
                opti.bounded(0.0, ctu[:, 0], 1.0),
                ca.diff(t) > 0.0,
                ctu[:, 0] == ctu_init[:, 0],
                ctu[0, 1] == ctu_init[0, 1],
                ctu[-1, 1] == ctu_init[-1, 1],
                t[0] == 0.0,
                t[-1] == 1.0,
            ]
        )

        opti.minimize(residual)
        opti.solver("ipopt", default_options)
        opti.set_initial(ctu, ctu_init)
        opti.set_initial(t, t_init)

        sol = opti.solve()

        ctu_sol = sol.value(ctu)

        # ============================ lower ==========================
        if symmetry:
            ctl_sol = np.array(ctu_sol)
            ctl_sol[:, 1] = -ctu_sol[:, 1]
        else:
            opti = ca.Opti()

            ctu = opti.variable(nctu, 2)

            ctl = opti.variable(nctl, 2)
            ctl_init = np.zeros(ctl.shape)
            ctl_init[1:, 0] = np.linspace(0, 1, nctl - 1)
            ctl_init[1:-1, 1] = -0.5
            ctl_init[-1, 1] = lower_coordinates[-1, 1]

            t = opti.variable(lower_coordinates.shape[0])
            t_init = np.linspace(0, 1, lower_coordinates.shape[0], dtype="f8")

            af = BsplineAirfoil(ctu=ctu, ctl=ctl)
            coords = af.lower_coordinates(t)

            dist = coords - lower_coordinates
            residual = ca.sum(ca.dot(dist, dist))

            opti.subject_to(
                [
                    opti.bounded(0.0, t, 1.0),
                    opti.bounded(0.0, ctl[:, 0], 1.0),
                    ca.diff(t) > 0.0,
                    ctl[:, 0] == ctl_init[:, 0],
                    ctl[0, 1] == ctl_init[0, 1],
                    ctl[-1, 1] == ctl_init[-1, 1],
                    t[0] == 0.0,
                    t[-1] == 1.0,
                ]
            )

            opti.minimize(residual)
            opti.solver("ipopt", default_options)
            opti.set_initial(ctl, ctl_init)
            opti.set_initial(t, t_init)

            sol = opti.solve()

            ctl_sol = sol.value(ctl)

        af_fit = BsplineAirfoil(ctu=ctu_sol, ctl=ctl_sol)
        return af_fit


def test01():
    import matplotlib.pyplot as plt 
    import scienceplots

    cpts = np.array([[0.0, 0.0], [0.5, 0.5], [0.6, 1.2], [2.0, 0.0]])
    sp = Bspline(cpts=cpts, degree=3)
    print(sp.get_xy_from_u(np.array([0.1,0.5,0.6,0.8,0.9])))
    xy=sp.get_xy_from_u(np.linspace(0,1,100))

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
            ax.plot(sp.cpts[:, 0], sp.cpts[:, 1], "--o", markersize=10, label="cpts")

            ax.legend(fontsize=15)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.show()

def test04():
    import matplotlib.pyplot as plt
    import scienceplots

    data = np.array(
        [
            [1.0, 1.0],
            [1.5, 2.0],
            [2.0, 3.0],
            [2.5, 3.0],
            [3.5, 1.0],
            [4.0, 0.0],
            [5.0, -1.0],
        ]
    )

    sp = Bspline.fit(data=data, ncpts=5, degree=3)
    xy = sp.get_xy_from_u(np.linspace(0, 1, 100))

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
            ax.plot(sp.cpts[:, 0], sp.cpts[:, 1], "--o", markersize=10, label="cpts")
            ax.plot(data[:, 0], data[:, 1], "^", markersize=10, label="data")

            ax.legend(fontsize=15)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.show()


if __name__ == "__main__":
    # test01()
    test04()
