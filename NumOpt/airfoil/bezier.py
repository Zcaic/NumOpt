from ..opti import asb, anp, cas, np
from scipy.special import comb
from ..cprint import cprint_green


class Bezier:
    def __init__(self, control_points):
        self.control_points = control_points
        self.setup()

    def setup(self):
        self.order = self.control_points.shape[0] - 1
        self.ndim = self.control_points.shape[1]
        self._K = comb(self.order, np.arange(self.order + 1))
        self._C = np.tile(self._K, (self.ndim, 1)).T * self.control_points

    def __call__(self, t):
        t = anp.reshape(t, (1, -1))
        ti = []
        t_ni = []
        t_ = 1 - t
        for i in range(self.order + 1):
            ti.append(t**i)
            t_ni.append(t_ ** (self.order - i))
        ti = anp.concatenate(ti, axis=0)
        t_ni = anp.concatenate(t_ni, axis=0)
        t_mat = (ti * t_ni).T
        pts = t_mat @ self._C
        return pts


class BezierAirfoil:
    def __init__(self, ctu, ctl):
        self.ctu = ctu
        self.ctl = ctl

        self.bezier_upper = Bezier(self.ctu)
        self.bezier_lower = Bezier(self.ctl)

        self.__thickness = self.ctu[-1, 1] - self.ctl[-1, 1]
        self.__symmetry = False

    def upper_coordinates(self, t):
        pts = self.bezier_upper(t)[::-1, :]
        return pts

    def lower_coordinates(self, t):
        pts = self.bezier_lower(t)
        return pts

    def coordinates(self, t):
        pts_u = self.upper_coordinates(t)
        pts_l = self.lower_coordinates(t)

        pts = anp.concatenate([pts_u[:-1, :], pts_l])
        return pts

    @property
    def thickness(self):
        return self.__thickness

    @property
    def symmetry(self):
        return self.__symmetry

    def set_thickness(self, thickness):
        thickness = thickness
        old_thickness = self.thickness
        delta_thickness = thickness - old_thickness

        t = anp.cosspace(0, 1, 100)

        upper_coords = self.upper_coordinates(t)
        upper_coords[:, 1] += upper_coords[:, 0] * delta_thickness / 2.0
        lower_coords = self.lower_coordinates(t)
        lower_coords[:, 1] -= lower_coords[:, 0] * delta_thickness / 2.0

        new_af = self.fit(upper_coords, lower_coords, nctu=self.ctu.shape[0], nctl=self.ctl.shape[0], symmetry=self.symmetry)
        return new_af

    @staticmethod
    def fit(upper_coordinates, lower_coordinates, nctu=7, nctl=7, symmetry=False):
        default_options = {
            "ipopt.sb": "yes",
            "ipopt.max_iter": 1000,
            "ipopt.max_cpu_time": 1e20,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.fast_step_computation": "yes",
            "detect_simple_bounds": False,
            "expand": True,
            # =======================
            # if verbose
            # "ipopt.print_level": 5,
            # =======================
            # =============================
            # if no verbose
            "print_time": False,
            "ipopt.print_level": 0,
            # =============================.
            # ============================
            # This is a heuristic that tends to result in more robust convergence on highly nonconvex problems.
            # On convex problems and larger problems, the default setting ("adaptive") tends to result in faster convergence.
            # ===========================
            # "ipopt.mu_strategy": "monotone",
            # ===============================
            # This is another heuristic that tells the optimizer to focus on finding the feasible space initially.
            # This can be good when you know your initial guess is very far from satisfying all constraints.
            # "ipopt.start_with_resto": "yes",
            # ===============================
        }

        # ------------------------ upper -------------------------------
        opti = cas.Opti()
        ctu = opti.variable(nctu, 2)
        ctl = opti.parameter(nctl, 2)

        af = BezierAirfoil(ctu=ctu, ctl=ctl)
        # t = anp.cosspace(0, 1, upper_coordinates.shape[0])
        t = opti.variable(upper_coordinates.shape[0])
        coords = af.upper_coordinates(t)

        # residual = cas.sum((coords[:, 1] - upper_coordinates[:, 1]) ** 2)
        dist = coords - upper_coordinates
        residual = cas.sum(cas.dot(dist, dist))

        opti.subject_to(
            [
                # coords[:, 0] == upper_coordinates[:, 0],
                ctu[0, 0] == 0.0,
                ctu[0, 1] == 0.0,
                ctu[1, 0] == 0.0,
                ctu[-1, 0] == 1.0,
                ctu[-1, 1] == upper_coordinates[0, 1],
                opti.bounded(0.0, t, 1.0),
                opti.bounded(0.0, ctu[:, 0], 1.0),
                cas.diff(t) >= 0.0,
                cas.diff(ctu[:, 0]) >= 0.0,
            ]
        )

        opti.minimize(residual)
        opti.solver("ipopt", default_options)
        opti.set_initial(ctu[:, 0], np.linspace(0, 1, nctu))
        opti.set_initial(ctu[:, 1], 0.05)
        opti.set_initial(t, np.linspace(0, 1, upper_coordinates.shape[0]))

        # cprint_green("fit upper...")
        sol = opti.solve()

        ctu_sol = sol.value(ctu)

        # --------------------------- lower ------------------------------------------
        if symmetry:
            ctl_sol = np.array(ctu_sol)
            ctl_sol[:, 1] = -ctl_sol[:, 1]
        else:
            opti = cas.Opti()
            ctu = opti.parameter(nctu, 2)
            ctl = opti.variable(nctl, 2)

            af = BezierAirfoil(ctu=ctu, ctl=ctl)
            # t = anp.cosspace(0, 1, upper_coordinates.shape[0])
            t = opti.variable(lower_coordinates.shape[0])
            coords = af.lower_coordinates(t)

            dist = coords - lower_coordinates
            # residual = cas.sum((coords[:, 1] - lower_coordinates[:, 1]) ** 2)
            residual = cas.sum(cas.dot(dist, dist))

            opti.subject_to(
                [
                    # coords[:, 0] == lower_coordinates[:, 0],
                    ctl[0, 0] == 0.0,
                    ctl[0, 1] == 0.0,
                    ctl[1, 0] == 0.0,
                    ctl[-1, 0] == 1.0,
                    ctl[-1, 1] == lower_coordinates[-1, 1],
                    opti.bounded(0.0, t, 1.0),
                    opti.bounded(0.0, ctl[:, 0], 1.0),
                    cas.diff(t) >= 0.0,
                    cas.diff(ctl[:, 0]) >= 0.0,
                ]
            )

            opti.minimize(residual)
            opti.solver("ipopt", default_options)
            opti.set_initial(ctl[:, 0], np.linspace(0, 1, nctl))
            opti.set_initial(ctl[:, 1], -0.1)
            opti.set_initial(t, np.linspace(0, 1, lower_coordinates.shape[0]))

            # cprint_green("fit lower...")
            sol = opti.solve()

            ctl_sol = sol.value(ctl)

        af = BezierAirfoil(ctu=ctu_sol, ctl=ctl_sol)
        af.__symmetry = symmetry
        return af
