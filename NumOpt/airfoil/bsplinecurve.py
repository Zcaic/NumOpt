import casadi as ca
import numpy as np
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


class BsplineCurve:
    def __init__(self, cpts: np.ndarray, degree=3):
        self.cpts = cpts
        self.degree = degree
        self.knots = self.quasi_uniform_knots(self.cpts.shape[0], self.degree)
        self._func_get_xy_from_u = self.__gen_get_xy_from_u()

    @staticmethod
    def quasi_uniform_knots(ncpts, degree):
        middle = np.linspace(0, 1, ncpts - degree + 1)
        start = np.zeros(degree, dtype="f8")
        end = np.ones(degree, dtype="f8")
        knots = np.hstack([start, middle, end])
        return knots

    @staticmethod
    def deBoor(s, u, knots, cpts, degree):
        d = [cpts[s - degree + i,:].reshape((-1,1)) for i in range(degree + 1)]

        for r in range(1, degree + 1):
            for i in range(degree, r - 1, -1):
                idx = i + s - degree
                denom = knots[i + 1 + s - r] - knots[idx]
                if abs(denom) < 1e-14:
                    alpha = ca.DM(0.0)
                else:
                    alpha = (u - knots[idx]) / denom
                d[i] = (1 - alpha) * d[i - 1] + alpha * d[i]
        return d[degree]

    @staticmethod
    def __get_xy_from_u(u, cpts, knots, degree):
        span = range(degree, knots.shape[0] - degree - 1)
        C = ca.DM.zeros(cpts.shape[1],1)
        last_span = span[-1]
        for s in span:
            Cs = BsplineCurve.deBoor(s, u, knots, cpts, degree)
            if s == last_span:
                cond = ca.logic_and(u >= knots[s], u <= knots[s + 1])
            else:
                cond = ca.logic_and(u >= knots[s], u <= knots[s + 1])

            C = ca.if_else(cond, Cs, C)
        return C

    def __gen_get_xy_from_u(self):
        u = ca.GenMX_sym("u")
        C = self.__get_xy_from_u(u, self.cpts, self.knots, self.degree)
        func = ca.Function("get_xy_from_u", [u], [C])
        return func

    def get_xy_from_u(self, u=cosspace(0, 1, 100), to_numpy=True):
        func = self._func_get_xy_from_u.map(u.shape[0])
        xy = func(u).T
        if to_numpy:
            xy = xy.toarray()
        return xy

    def get_xy_from_x(self, x):
        opti = Opti()
        u = opti.variable(init_guess=0.5, n_vars=x.shape[0], lower_bound=0.0, upper_bound=1.0)
        xy = self.get_xy_from_u(u, False)
        opti.subject_to(xy[:, 0] == x)
        opti.ipopt_solver(verbose=False, options={"expand": False})
        sol = opti.solve()
        xy_star = sol(xy)
        return xy_star

    @staticmethod
    def fit(data, ncpts, degree):
        def gen_fun(ncpts,degree,nu):
            u = ca.MX.sym("u")
            cpts = ca.MX.sym("cpts", ncpts, 2)
            knots = BsplineCurve.quasi_uniform_knots(ncpts, degree)
            C = BsplineCurve.__get_xy_from_u(u, cpts, knots, degree)
            func = ca.Function("get_xy_from_u", [u, cpts], [C])
            func_mpi = func.map(nu,[False,True],[],{})
            return func_mpi
        
        
        # cpts = np.array([[0.0, 0.0], [0.5, 0.5], [0.6, 1.2], [2.0, 0.0]])
        # degree=3
        # knots=BsplineCurve.quasi_uniform_knots(cpts.shape[0],degree)
        # u=np.array([0.1,0.5,0.6,0.8,0.9])
        # func_mpi=gen_fun(cpts.shape[0],degree,u.shape[0])
        # print(func_mpi(u,cpts).T)


        opti = Opti()
        cpts_x = opti.variable(init_guess=np.linspace(data[0, 0], data[-1, 0], ncpts),freeze=True)
        cpts_y = opti.variable(init_guess=np.linspace(data[0, 1], data[-1, 1], ncpts))
        u = opti.variable(init_guess=0.5, n_vars=data.shape[0], lower_bound=0.0, upper_bound=1.0)
        cpts = ca.horzcat(cpts_x, cpts_y)

        func = gen_fun(ncpts,degree,data.shape[0])

        data_fit = func(u, cpts).T
        y_dist = data_fit[:, 1] - data[:, 1]
        dist = ca.sum(y_dist**2)

        opti.subject_to(
            [
                data_fit[:, 0] == data[:, 0],
                # data_fit[:,1]==data[:,1]
            ]
        )

        opti.minimize(dist)
        opti.ipopt_solver()
        sol = opti.solve()

        cpts = sol(cpts)
        data_fit = sol(data_fit)
        sp = BsplineCurve(cpts=cpts, degree=degree)
        return sp


class BsplineAirfoil:
    def __init__(self, cptus, cptls, degree):
        self.cptus = cptus
        self.cptls = cptls
        self.degree = degree
        self._spu = BsplineCurve(cpts=self.cptus, degree=self.degree)
        self._spl = BsplineCurve(cpts=self.cptls, degree=self.degree)

    def coordinates(self, u=cosspace(0, 1, 100), to_numpy=True):
        upper_coords = self._spu.get_xy_from_u(u, to_numpy)
        lower_coords = self._spl.get_xy_from_u(u, to_numpy)
        if to_numpy:
            coords = np.vstack([upper_coords[::-1], lower_coords[1:]])
        else:
            coords = ca.vertcat(upper_coords[::-1, :], lower_coords[1:, :])
        return coords

    def upper_coordinates(self, u=cosspace(0, 1, 100), to_numpy=True):
        coords = self._spu.get_xy_from_u(u, to_numpy)
        return coords

    def lower_coordinates(self, u=cosspace(0, 1, 100), to_numpy=True):
        coords = self._spl.get_xy_from_u(u, to_numpy)
        return coords

    @staticmethod
    def fit(upper_coords, lower_coords, TE=None, ncptus=9, ncptls=9, degree=3):
        def gen_sp_func(ncpts, knots, degree):
            u = ca.MX.sym("u")
            cpts = ca.MX.sym("cpts", ncpts, 2)
            sp = ca.bspline(u, cpts.T, [knots], [degree], 2, {})
            sp = ca.Function("sp", [u, cpts], [sp])
            return sp

        opti = Opti()

        cptus_x = np.zeros((ncptus,))
        cptus_x[1:] = cosspace(0, 1, ncptus - 1)

        cptls_x = np.zeros((ncptls,))
        cptls_x[1:] = cosspace(0, 1, ncptls - 1)

        cptus_y = opti.variable(init_guess=0.06, n_vars=ncptus - 2)
        cptls_y = opti.variable(init_guess=-0.06, n_vars=ncptls - 2)

        tu = opti.variable(init_guess=0.5, n_vars=upper_coords.shape[0], lower_bound=0.0, upper_bound=1.0)
        tl = opti.variable(init_guess=0.5, n_vars=lower_coords.shape[0], lower_bound=0.0, upper_bound=1.0)

        if TE:
            cptus_y = ca.vertcat([0.0], cptus_y, [TE / 2.0])
            cptls_y = ca.vertcat([0.0], cptls_y, [-TE / 2.0])
        else:
            cptus_y = ca.vertcat([0.0], cptus_y, [upper_coords[-1, 1]])
            cptls_y = ca.vertcat([0.0], cptls_y, [lower_coords[-1, 1]])

        cptus = ca.horzcat(cptus_x, cptus_y)
        cptls = ca.horzcat(cptls_x, cptls_y)

        knots_u = BsplineCurve.quasi_uniform_knots(ncptus, degree)
        knots_l = BsplineCurve.quasi_uniform_knots(ncptls, degree)

        spu = gen_sp_func(ncpts=ncptus, knots=knots_u, degree=degree)
        spl = gen_sp_func(ncpts=ncptls, knots=knots_l, degree=degree)

        spu_xy = spu(tu.T, cptus).T
        spl_xy = spl(tl.T, cptls).T

        dist_u = upper_coords - spu_xy
        dist_l = lower_coords - spl_xy

        dist = ca.dot(dist_u, dist_u) + ca.dot(dist_l, dist_l)

        opti.subject_to(
            [
                tu[0, 0] == 0.0,
                tu[-1, 0] == 1.0,
                tl[0, 0] == 0.0,
                tl[-1, 0] == 1.0,
            ]
        )

        opti.minimize(dist)
        opti.ipopt_solver(options={"expand": False})
        sol = opti.solve()
        print(sol(cptus), sol(cptls))


def test01():
    def find_valid_span(knots, degree):
        k = range(degree, knots.shape[0] - degree - 1)
        span = []
        for i in k:
            span.append([knots[i], knots[i + 1]])
        return k, span

    def __deBoor(s, u, knots, cpts, degree):
        d = [cpts[s - degree + i] for i in range(degree + 1)]

        for r in range(1, degree + 1):
            for i in range(degree, r - 1, -1):
                idx = i + s - degree
                denom = knots[i + 1 + s - r] - knots[idx]
                if abs(denom) < 1e-14:
                    alpha = ca.DM(0.0)
                else:
                    alpha = (u - knots[idx]) / denom
                d[i] = (1 - alpha) * d[i - 1] + alpha * d[i]
        return d[degree]
        # d=[cpts[s-degree+i] for i in range(degree+1)]
        # for r in range(1,degree+1):
        #     for j in range(degree,r-1,-1):
        #         i=s-degree+j
        #         denom=knots[i+degree-r+1]-knots[i]

        #         if ca.fabs(denom)<1e-12:
        #             alpha=ca.DM(0.0)
        #         else:
        #             alpha=(u-knots[i])/denom
        #         d[j]=(1.0-alpha)*d[j-1]+alpha*d[j]
        # return d[degree]

    def deBoor(u, cpts, knots, degree):
        span = range(degree, knots.shape[0] - degree - 1)
        C = ca.DM.zeros(cpts.shape[1], 1)
        last_span = span[-1]
        for s in span:
            Cs = __deBoor(s, u, knots, cpts, degree)
            if s == last_span:
                cond = ca.logic_and(u >= knots[s], u <= knots[s + 1])
            else:
                cond = ca.logic_and(u >= knots[s], u <= knots[s + 1])

            C = ca.if_else(cond, Cs, C)
        return C

    cpts = np.array([[0.0, 0.0], [0.5, 0.5], [0.6, 1.2], [2.0, 0.0]])
    degree = 3
    knots = BsplineCurve.quasi_uniform_knots(cpts.shape[0], degree)
    print(deBoor(0.6, cpts=cpts, knots=knots, degree=degree))


def test02():
    import matplotlib.pyplot as plt
    import scienceplots

    cpts = np.array([[0.0, 0.0], [0.5, 0.5], [0.6, 1.2], [2.0, 0.0]])
    sp = BsplineCurve(cpts=cpts, degree=3)
    print(sp.get_xy_from_u(np.array([0.1, 0.5, 0.6, 0.8, 0.9])))
    u = np.linspace(0, 1, 100)
    xy = sp.get_xy_from_u(u)
    print(sp.get_xy_from_x(sp.get_xy_from_u(np.array([0.1, 0.5, 0.6, 0.8, 0.9]))[:, 0]))

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


def test03():
    import aerosandbox as asb

    af_init = asb.Airfoil("n63415")
    BsplineAirfoil.fit(upper_coords=af_init.upper_coordinates()[::-1], lower_coords=af_init.lower_coordinates(), TE=None, ncptus=9, ncptls=6, degree=3)


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

    sp = BsplineCurve.fit(data=data, ncpts=5, degree=3)
    xy = sp.get_xy_from_u(u=np.linspace(0, 1, 100))

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
    # test02()
    # test03()
    test04()
