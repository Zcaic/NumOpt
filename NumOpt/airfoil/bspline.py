from ..opti import asb, anp, cas, np


class Bspline:
    def __init__(self, ctrlpts, degree=3):
        self.ctrlpts = ctrlpts
        self.n = self.ctrlpts.shape[0] - 1
        self.degree = degree
        self.order = self.degree + 1
        self.segments = self.n - self.degree + 1
        self.knots = self.quasi_uniform_knots()

    def quasi_uniform_knots(self):
        middle = np.linspace(0, 1, self.segments + 1)
        start = np.zeros(self.degree, dtype="f8")
        end = np.ones(self.degree, dtype="f8")
        knots = np.hstack([start, middle, end])
        return knots

    @staticmethod
    def __deBoor(t, i, k, knots):
        if k == 1:
            return cas.if_else(cas.logic_and(t >= knots[i], t < knots[i + 1]), 1.0, 0.0)
        else:
            term1 = term2 = 0.0
            delta_t1 = knots[i + k - 1] - knots[i]
            delta_t2 = knots[i + k] - knots[i + 1]

            if delta_t1 != 0.0:
                term1 = (t - knots[i]) / (delta_t1) * Bspline.__deBoor(t, i, k - 1, knots)

            if delta_t2 != 0.0:
                term2 = (knots[i + k] - t) / (delta_t2) * Bspline.__deBoor(t, i + 1, k - 1, knots)

            return term1 + term2

    def N_coef(self, t, i):
        return Bspline.__deBoor(t, i, self.order, self.knots)

    def __call__(self, t):
        def func():
            tmp = 0.0
            for j in range(self.n + 1):
                N_coef = self.N_coef(u, j)
                tmp = tmp + N_coef * self.ctrlpts[j : j + 1, :]
            return tmp

        nts = t.shape[0]
        pts = [0.0] * nts

        for i in range(nts):
            u = t[i]
            pts[i] = cas.if_else(u == 1.0, self.ctrlpts[-1:, :], func())
            # if u == 1.0:
            #     pts[i] = self.ctrlpts[-1:, :]
            # else:
            #     for j in range(self.n + 1):
            #         N_coef = self.N_coef(u, j)
            #         pts[i] = pts[i] + N_coef * self.ctrlpts[j : j + 1, :]
        pts = cas.vcat(pts)
        return pts


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

        pts = cas.vcat([pts_upper[:-1, :], pts_lower])
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
        opti = cas.Opti()

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
        residual = cas.sum(cas.dot(dist, dist))

        opti.subject_to(
            [
                opti.bounded(0.0, t, 1.0),
                opti.bounded(0.0, ctu[:, 0], 1.0),
                cas.diff(t) > 0.0,
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
            opti = cas.Opti()

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
            residual = cas.sum(cas.dot(dist, dist))

            opti.subject_to(
                [
                    opti.bounded(0.0, t, 1.0),
                    opti.bounded(0.0, ctl[:, 0], 1.0),
                    cas.diff(t) > 0.0,
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
