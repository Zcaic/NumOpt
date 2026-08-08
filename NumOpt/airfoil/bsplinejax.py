import jax
import jax.numpy as jnp 
from NumOpt import Opti

jax.config.update("jax_enable_x64",True)


def cosspace(start: float = 0.0, stop: float = 1.0, num: int = 50):
    mean = (stop + start) / 2
    amp = (stop - start) / 2
    ones = 0 * start + 1
    spaced_array = mean + amp * jnp.cos(jnp.linspace(jnp.pi * ones, 0 * ones, num))

    # Fix the endpoints, which might not be exactly right due to floating-point error.
    spaced_array[0] = start
    spaced_array[-1] = stop

    return spaced_array 

class BsplineCurve:
    def __init__(self, cpts: jnp.ndarray, degree=3):
        self.cpts = cpts
        self.degree = degree
        self.knots = self.quasi_uniform_knots(self.cpts.shape[0], self.degree)
        self._func_get_xy_from_u = self.__gen_get_xy_from_u()

    @staticmethod
    def quasi_uniform_knots(ncpts, degree):
        middle = jnp.linspace(0, 1, ncpts - degree + 1)
        start = jnp.zeros(degree, dtype="f8")
        end = jnp.ones(degree, dtype="f8")
        knots = jnp.hstack([start, middle, end])
        return knots

    @staticmethod
    def deBoor(s, u, knots, cpts, degree):
        p = degree
        
        idx0 = s - p
        idxs = idx0 + jnp.arange(p + 1, dtype=jnp.int32)
        d = cpts[idxs, :]   # (p+1, dim)

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = s - p + j
                denom = knots[i + p - r + 1] - knots[i]
                alpha = jnp.where(
                    jnp.abs(denom) < 1e-14,
                    0.0,
                    (u - knots[i]) / denom,
                )
                dj = (1.0 - alpha) * d[j - 1] + alpha * d[j]
                d = d.at[j].set(dj)

        return d[p]

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