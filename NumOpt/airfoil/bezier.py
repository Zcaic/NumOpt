from ..opti import asb, anp, cas, Opti

# import casadi as cas


class Bezier(object):
    def __init__(self, control_points):
        self.control_points = anp.array(control_points)
        self.setup()

    def setup(self):
        self.order = self.control_points.shape[0] - 1

    def __deCasteljau(self, t):
        npoint = t.shape[0]
        layer = [[] for _ in range(self.order + 1)]
        for i in range(self.order + 1):
            tmp = anp.tile(self.control_points[i, :], (npoint, 1))
            layer[0].append(tmp)
        for i in range(1, self.order + 1):
            pre_layer = layer[i - 1]
            for j in range(len(pre_layer) - 1):
                coord = (1 - t) * pre_layer[j] + t * pre_layer[j + 1]
                layer[i].append(coord)
        return layer

    def __call__(self, t):
        t = anp.array(t).reshape((-1, 1))
        t = anp.tile(t, [1, 3])
        layer = self.__deCasteljau(t)
        return layer[-1][0]


class Bezier2D:
    def __init__(self, control_points):
        self.control_points = anp.array(control_points)
        self.setup()

    def setup(self):
        self.order = self.control_points.shape[0] - 1

    def __deCasteljau(self, t):
        npoint = t.shape[0]
        layer = [[] for _ in range(self.order + 1)]
        for i in range(self.order + 1):
            tmp = anp.tile(self.control_points[i, :], (npoint, 1))
            layer[0].append(tmp)
        for i in range(1, self.order + 1):
            pre_layer = layer[i - 1]
            for j in range(len(pre_layer) - 1):
                coord = (1 - t) * pre_layer[j] + t * pre_layer[j + 1]
                layer[i].append(coord)
        return layer

    def __call__(self, t):
        t = anp.array(t).reshape((-1, 1))
        t = anp.tile(t, [1, 2])
        layer = self.__deCasteljau(t)
        return layer[-1][0]


class BezierAf:
    def __init__(self, ctu, ctl):
        self.ctu = anp.array(ctu)
        self.ctl = anp.array(ctl)

        self.setup()

    def setup(self):
        self.bezier_u = Bezier2D(self.ctu)
        self.bezier_l = Bezier2D(self.ctl)

    def upper_coordinates(self, npoints=100, disturbation=anp.cosspace):
        t = disturbation(num=npoints)
        coords = self.bezier_u(t[::-1])
        return coords

    def lower_coordinates(self, npoints=100, disturbation=anp.cosspace):
        t = disturbation(num=npoints)
        coords = self.bezier_l(t)
        return coords

    def coordinates(self, npoints_per_side=100, disturbation=anp.cosspace):
        upper_coords = self.upper_coordinates(npoints=npoints_per_side, disturbation=disturbation)
        lower_coords = self.lower_coordinates(npoints=npoints_per_side, disturbation=disturbation)

        coords = anp.concatenate([upper_coords[:-1, :], lower_coords], axis=0)
        return coords

    def __thickness(self, x):
        opti = Opti()
        nt = x.shape[0]
        nctu = self.ctu.shape[0]
        nctl = self.ctl.shape[0]
        # nt=opti.variable(init_guess=100,freeze=True)
        # nctu=opti.variable(init_guess=7,freeze=True)
        # nctl=opti.variable(init_guess=7,freeze=True)
        x = opti.variable(init_guess=anp.linspace(0.0, 1.0, nt), freeze=True)

        tu = opti.variable(init_guess=anp.linspace(0.0, 1.0, nt), lower_bound=0.0, upper_bound=1.0)
        tl = opti.variable(init_guess=anp.linspace(0.0, 1.0, nt), lower_bound=0.0, upper_bound=1.0)

        ctu = anp.concatenate([opti.variable(init_guess=anp.ones(2), freeze=True) for _ in range(nctu)], axis=1)
        ctl = anp.concatenate([opti.variable(init_guess=anp.ones(2), freeze=True) for _ in range(nctl)], axis=1)

        bezier_u = Bezier2D(ctu.T)
        bezier_l = Bezier2D(ctl.T)

        coords_u = bezier_u(tu)
        coords_l = bezier_l(tl)

        opti.subject_to([coords_u[:, 0] == x[:, 0], coords_l[:, 0] == x[:, 0], anp.diff(tu) > 0.0, anp.diff(tl) > 0.0])

        opti.solver(verbose=False)
        coords = opti.to_function("coords", [ctu, ctl, x], [coords_u, coords_l])
        return coords

    def thickness(self, x=anp.linspace(0, 1, 100)):
        # thick,coords_u,coords_l = self._thickness(x)(self.ctu.T, self.ctl.T, x)
        # return thick
        coords_u, coords_l = self.__thickness(x)(self.ctu.T, self.ctl.T, x)
        thickness = coords_u[:, 1] - coords_l[:, 1]
        thickness = anp.concatenate([x, thickness], axis=1)
        return thickness

    def max_thickness(self, x=anp.linspace(0, 1, 100)):
        thick = self.thickness(x)
        if not isinstance(thick, (cas.SX, cas.MX)):
            max_id = anp.argmax(thick[:, 1])
            return anp.array([[max_id, thick[max_id, 1]]])
        else:
            max_thick = anp.max(thick[:, 1])
            max_thick_idx = cas.find(max_thick == thick[:, 1])
            ret = anp.concatenate([max_thick_idx, max_thick], axis=1)
            return ret


def test1():
    R0 = [0, 0, 0]
    R1 = [0, 0.5, 0]
    R2 = [0.5, 1.0, 0]
    R3 = [1, 0.5, 0]
    R4 = [1.0, 0, 0]

    curve = Bezier([R0, R1, R2, R3, R4])
    t = anp.linspace(0, 1, 10)
    res = curve(t)
    print(res)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(curve.control_points[:, 0], curve.control_points[:, 1], "--o")
    ax.plot(res[:, 0], res[:, 1])
    plt.show()


def test2():
    R0 = [0.0, 0.0]
    R1 = [0.0, 0.5]
    R2 = [0.5, 1.0]
    R3 = [1.0, 0.5]
    R4 = [1.0, 0.0]

    curve = Bezier2D([R0, R1, R2, R3, R4])
    t = anp.linspace(0, 1, 10)
    res = curve(t)
    print(res)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(curve.control_points[:, 0], curve.control_points[:, 1], "--o")
    ax.plot(res[:, 0], res[:, 1])
    plt.show()


def test3():
    ctu = anp.array([[0.0, 0.0], [0.0, 26.8], [122.7, 52.9], [392.0, 110.2], [710.4, 41.6], [850.5, 12.9], [1000.0, 0.0]])
    ctl = anp.array([[0.0, 0.0], [0.0, -26.6], [160.1, -36.0], [421.5, -71.3], [716.2, -20.8], [850.8, 8.5], [1000.0, 0.0]])

    af = BezierAf(ctu=ctu, ctl=ctl)
    coords = af.coordinates()
    max_thick = af.max_thickness()
    print(max_thick)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(coords[:, 0], coords[:, 1])
    plt.show()


def test4():
    opti = Opti()
    x = opti.variable(init_guess=5.0)
    return


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    test4()
