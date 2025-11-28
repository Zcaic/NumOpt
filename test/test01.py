from NumOpt.airfoil.bezier import asb, anp, Opti, BezierAf, cas


def test1():
    ctu = anp.array([[0.0, 0.0], [0.0, 26.8], [122.7, 52.9], [392.0, 110.2], [710.4, 41.6], [850.5, 12.9], [1000.0, 0.0]])
    ctl = anp.array([[0.0, 0.0], [0.0, -26.6], [160.1, -36.0], [421.5, -71.3], [716.2, -20.8], [850.8, 8.5], [1000.0, 0.0]])

    af = BezierAf(ctu=ctu, ctl=ctl)
    coords = af.coordinates()
    max_thick = anp.float64(af.max_thickness(x=anp.linspace(0.0, 1000.0, 100)))
    print(max_thick)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(coords[:, 0], coords[:, 1])
    plt.show()


def test2():
    opti = Opti()
    ctu = anp.array(
        [
            [0.0, 0.0],
            [0.0, 26.8],
            [122.7, 52.9],
            [392.0, opti.variable(init_guess=110.0, lower_bound=50.0)],
            [710.4, 41.6],
            [850.5, 12.9],
            [1000.0, 0.0],
        ]
    )
    ctl = anp.array([[0.0, 0.0], [0.0, -26.6], [160.1, -36.0], [421.5, -71.3], [716.2, -20.8], [850.8, 8.5], [1000.0, 0.0]])

    af = BezierAf(ctu=ctu, ctl=ctl)
    coords = af.coordinates()
    max_thick = af.max_thickness(x=anp.linspace(0, 1000, 100))

    opti.subject_to([max_thick[0, 1] == 120.0])

    opti.solver()
    sol = opti.solve()
    print(sol(max_thick))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(sol(coords)[:, 0], sol(coords)[:, 1])
    plt.show()


def test3():
    nx = 5

    # inputs
    x = cas.GenMX.sym("x", nx, 1)

    max_x = anp.max(x)

    max_x_index = cas.find(max_x == x)

    ret = anp.concatenate([max_x_index, max_x], axis=1)

    func = cas.Function("func", [x], [ret])

    print(func(anp.array([1, 3, 3, 3, 5])))


if __name__ == "__main__":
    # test1()
    test2()
    # test3()
