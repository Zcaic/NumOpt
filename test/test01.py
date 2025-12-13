from NumOpt.airfoil.bezier import Bezier, BezierAirfoil
from NumOpt.opti import cas, Opti, anp
import numpy as np
import aerosandbox as asb
from NumOpt.cprint import nostd
from NumOpt.airfoil.export_cst2nx import cst2nx


def test1():
    opti = Opti()
    T = opti.variable(init_guess=2.8e3 * 9.8)

    A = np.pi * 3.5**2
    vi = (T / (2 * 1.225 * A)) ** 0.5

    Pin = 743e3

    eta = T * vi / Pin

    opti.subject_to([eta == 0.80])

    opti.solver()
    sol = opti.solve()
    print(sol(T))


def test2():
    af = asb.Airfoil(coordinates="C:/Users/Zcaic/Desktop/NACA 64-220 - Camberline=1.dat").normalize().set_TE_thickness(2.4e-3)
    # af=asb.Airfoil("n63415")
    af_new = BezierAirfoil.fit(
        upper_coordinates=af.upper_coordinates(), lower_coordinates=af.lower_coordinates(), nctu=7, nctl=7, symmetry=False
    )
    # af_new = af_new.set_thickness(5e-3)
    # print(af_new.thickness)
    print(af_new.ctu * 1000)
    print(af_new.ctl * 1000)
    coords = af_new.coordinates(anp.cosspace(0, 1, 100))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(af.coordinates[:, 0], af.coordinates[:, 1], label="ori")
    ax.plot(coords[:, 0], coords[:, 1], label="fit")
    ax.plot(af_new.ctu[:, 0], af_new.ctu[:, 1], "o--", label="ctu")
    ax.plot(af_new.ctl[:, 0], af_new.ctl[:, 1], "o--", label="ctl")
    ax.legend()
    plt.show()


def test3():
    af = asb.Airfoil(coordinates="C:/Users/Zcaic/Desktop/ONERA OA213.dat").normalize()
    af_new = af.to_kulfan_airfoil()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(af.coordinates[:, 0], af.coordinates[:, 1], label="ori")
    ax.plot(af_new.coordinates[:, 0], af_new.coordinates[:, 1], label="fit")
    # ax.plot(af_new.ctu[:, 0], af_new.ctu[:, 1], "o--", label="ctu")
    # ax.plot(af_new.ctl[:, 0], af_new.ctl[:, 1], "o--", label="ctl")
    ax.legend()
    plt.show()
    # inputs
    # x = cas.GenMX.sym("x", nx, 1)

    # max_x = anp.max(x)

    # max_x_index = cas.find(max_x == x)

    # ret = anp.concatenate([max_x_index, max_x], axis=1)

    # func = cas.Function("func", [x], [ret])

    # print(func(anp.array([1, 3, 3, 3, 5])))


def test4():
    opti = cas.Opti()

    # 声明变量K并设置为整数类型
    K = opti.variable()
    opti.set_domain(K, "integer")

    # 定义数组并转换为MX类型
    A = [3, 2, 1, 2, 3]
    A = cas.MX(A)

    # 构建目标函数：最小化A[K]
    opti.minimize(A[K])

    # 添加约束：确保K在数组有效范围内
    opti.subject_to(0 <= (K <= 4))

    # 设置求解器并求解
    opti.solver("bonmin")
    sol = opti.solve()

    # 输出结果
    print(sol.value(K))


def test5():
    import sympy as sp

    p = sp.symbols("p", complex=True)
    v = sp.symbols("v")
    m = sp.symbols("m")
    r = sp.symbols("r")
    b = sp.symbols("b")
    K_h = sp.symbols("K_h")
    K_alpha = sp.symbols("K_alpha")
    S_alpha = sp.symbols("S_alpha")
    I_alpha = sp.symbols("I_alpha")
    a = sp.symbols("a")
    pi = np.pi

    A = sp.Matrix(
        [
            [
                m * p**2 + 2 * pi * r * v * b * p + K_h,
                S_alpha * p**2 + (1 - 2 * a) * pi * r * v * b**2 * p + 2 * pi * r * v**2 * b,
            ],
            [
                S_alpha * p**2 - (2 * a + 1) * pi * r * v * b**2 * p,
                I_alpha * p**2 + (2 * a * a) * pi * r * v * b**3 * p + K_alpha - (2 * a + 1) * pi * r * v**2 * b**2,
            ],
        ]
    )

    detA = A.det()

    p_expr = sp.solve(detA, p)

    # p_expr=sp.lambdify([v,m,r,b,K_h,K_alpha,S_alpha,I_alpha,a],p_expr)
    return p_expr


def test6():
    ctu = np.array([[0.0, 0.0], [0.0, 26.8], [122.7, 52.9], [392.0, 110.2], [710.4, 41.6], [850.5, 12.9], [1000.0, 0.0]])
    ctl = np.array([[0.0, 0.0], [0.0, -26.6], [160.1, -36.0], [421.5, -71.3], [716.2, -20.8], [850.8, 8.5], [1000.0, 0.0]])

    af = BezierAirfoil(ctu=ctu, ctl=ctl)
    coords = af.coordinates(np.linspace(0, 1, 100))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(coords[:, 0], coords[:, 1])
    plt.show()


def test7():
    opti = cas.Opti()
    x = opti.variable(5)
    obj = cas.sum((x - 0.5) ** 2)

    opti.subject_to([opti.bounded(-10.0, x, 10.0)])
    opti.minimize(obj)
    # opti.solver("sqpmethod",{"qpsol":"qrqp"})
    # opti.solver(
    #     "sqpmethod", {"qpsol": "qpoases", "hessian_approximation": "limited-memory", "qpsol_options":{"print_problem": False}}
    # )
    opti.solver("sqpmethod", {"qpsol": "qpoases"})

    opti.set_initial(x, 5.0)

    with nostd():
        sol = opti.solve()
    print(sol.value(x))


def test8():
    cst2nx()


def test9():
    from NumOpt.airfoil.bspline import Bspline

    # ctrlpts=np.array([[0.0,0.16],[0.25,0.1],[0.5,0.2],[0.75,0.1],[2.0,0.0]])
    ctrlpts = np.array([[50, 50], [100, 300], [300, 100], [380, 200], [400, 600], [500, 400], [300, 600]])
    bs = Bspline(ctrlpts=ctrlpts, degree=3)
    pts = bs(np.linspace(0.0, 1.0, 100))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pts[:, 0], pts[:, 1])
    ax.plot(ctrlpts[:, 0], ctrlpts[:, 1], "--o")
    plt.show()


def test10():

    def uniform_node(control_points, k=3):
        num, _ = control_points.shape
        n = num - 1

        return np.linspace(0, 1, n + k + 2)

    def quasi_uniform_node(control_points, k=3):
        num, _ = control_points.shape
        n = num - 1
        mid = np.linspace(0, 1, n - k + 2)

        return np.r_[np.zeros(k), mid, np.ones(k)]

    def piecewise_node(control_points, k=3):
        num, _ = control_points.shape
        n = num - 1

        ## restricted condition
        assert (n - k) % k == 0, "input is valid."

        tmp = np.linspace(0, 1, int((n - k) / k + 2))
        mid = np.r_[tmp[0], np.repeat(tmp[1:-1], k), tmp[-1]]

        return np.r_[np.zeros(k), mid, np.ones(k)]

    def non_uniform_node(control_points, k=3):
        """
        Hartley-Judd algorithem
        """
        num, _ = control_points.shape
        n = num - 1

        l = np.sqrt(np.sum(np.diff(control_points, axis=0) ** 2, axis=1))
        ll = l[0 : len(l) - 1] + l[1::]
        L = np.sum(ll)

        mid_size = n - k
        mid = np.zeros(mid_size)

        for i in range(mid_size):
            mid[i] = np.sum(ll[0 : i + 1]) / L

        knots = np.r_[np.zeros(k + 1), mid, np.ones(k + 1)]

        return knots

    def cal_B(i, k, knots, u):
        """
        de Boor-Cox recursion
        Args:
            i (int): ith point idx
            k (int): degree of b-spline , equal to ord - 1
            knots (ndarray): 1 dim
            u : independent variable

        Returns:
            B: B_{i,k}
        """
        if k == 1:
            B = 1 if knots[i] <= u <= knots[i + 1] else 0  ##
        else:
            coef1 = coef2 = 0
            if knots[i + k - 1] - knots[i] != 0:
                coef1 = (u - knots[i]) / (knots[i + k - 1] - knots[i])
            if knots[i + k] - knots[i + 1] != 0:
                coef2 = (knots[i + k] - u) / (knots[i + k] - knots[i + 1])

            B = coef1 * cal_B(i, k - 1, knots, u) + coef2 * cal_B(i + 1, k - 1, knots, u)

        return B

    def cal_curve(control_points, knots, t):

        num, dims = control_points.shape
        n = num - 1
        k = len(knots) - n - 1  # degree of b-spline

        N = len(t)
        P = np.zeros((N, dims))

        for idx in range(N):
            u = t[idx]
            for i in range(0, num):
                P[idx, :] += control_points[i, :] * cal_B(i, k, knots, u)
        return P

    import matplotlib.pyplot as plt

    control_points = np.array([[50, 50], [100, 300], [300, 100], [380, 200], [400, 600], [500, 400], [300, 600]])

    N = 500
    t = np.linspace(0.0, 1.0, N)

    uniform_knots = uniform_node(control_points)
    quasi_knots = quasi_uniform_node(control_points)
    piecewise_knots = piecewise_node(control_points)
    non_knots = non_uniform_node(control_points)

    P_uniform = cal_curve(control_points, uniform_knots, t)
    P_quasi = cal_curve(control_points, quasi_knots, t)
    P_piecewise = cal_curve(control_points, piecewise_knots, t)
    P_non = cal_curve(control_points, non_knots, t)

    fig, axs = plt.subplots(2, 2)

    ## 均匀 会闭合
    axs[0, 0].scatter(control_points[:, 0], control_points[:, 1], color="C0", facecolors="none", label="control points")
    axs[0, 0].plot(control_points[:, 0], control_points[:, 1], color="C0", linewidth=0.2)
    axs[0, 0].plot(P_uniform[:, 0], P_uniform[:, 1], color="C3", label="uniform")
    axs[0, 0].legend()

    ## 准均匀
    axs[0, 1].scatter(control_points[:, 0], control_points[:, 1], color="C0", facecolors="none", label="control points")
    axs[0, 1].plot(control_points[:, 0], control_points[:, 1], color="C0", linewidth=0.2)
    axs[0, 1].plot(P_quasi[:, 0], P_quasi[:, 1], color="C3", label="quasi")
    axs[0, 1].legend()

    ## 分段
    axs[1, 0].scatter(control_points[:, 0], control_points[:, 1], color="C0", facecolors="none", label="control points")
    axs[1, 0].plot(control_points[:, 0], control_points[:, 1], color="C0", linewidth=0.2)
    axs[1, 0].plot(P_piecewise[:, 0], P_piecewise[:, 1], color="C3", label="piecewise")
    axs[1, 0].legend()

    ## HJ 非均匀
    axs[1, 1].scatter(control_points[:, 0], control_points[:, 1], color="C0", facecolors="none", label="control points")
    axs[1, 1].plot(control_points[:, 0], control_points[:, 1], color="C0", linewidth=0.2)
    axs[1, 1].plot(P_non[:, 0], P_non[:, 1], color="C3", label="non uniform")
    axs[1, 1].legend()
    plt.show()


def test11():
    from NumOpt.airfoil.bspline import Bspline

    ctrlpts = np.array([[50, 50], [100, 300], [300, 100], [380, 200], [400, 600], [500, 400], [300, 600]])
    bs = Bspline(ctrlpts=ctrlpts, degree=3)

    opti = cas.Opti()
    t = opti.variable()
    opti.subject_to(opti.bounded(0.0, t, 1.0))

    pts = bs(t)
    opti.subject_to([pts[0, 0] == 100.0])

    default_options = {
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000,
        "ipopt.max_cpu_time": 1e20,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.fast_step_computation": "yes",
        "detect_simple_bounds": False,
        "expand": True,
        "ipopt.print_level": 5,
    }

    opti.solver("ipopt", default_options)

    opti.set_initial(t, 0.0)
    sol = opti.solve()
    print(sol.value(pts))


def test12():
    opti = cas.Opti()
    x = opti.variable()
    y = cas.if_else(cas.logic_and(x[0] <= 0.0, x[0] >= -1.0), -cas.cos(x) + 1, x**2)

    opti.subject_to([opti.bounded(-1.0, x, 1.0)])
    opti.minimize(y)
    default_options = {
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000,
        "ipopt.max_cpu_time": 1e20,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.fast_step_computation": "yes",
        "detect_simple_bounds": False,
        "expand": True,
        "ipopt.print_level": 5,
    }
    opti.solver("ipopt", default_options)

    opti.set_initial(x, 1.0)
    sol = opti.solve()
    print(sol.value(y))


def test13():
    from NumOpt.airfoil.bspline import Bspline

    af = asb.Airfoil("naca0012")
    coords = af.lower_coordinates()

    opti = cas.Opti()

    ctrlpts = opti.variable(9, 2)
    ctrlpts_init=np.zeros(ctrlpts.shape)
    ctrlpts_init[1:,0]=np.linspace(0,1,8)
    ctrlpts_init[1:,1]=0.5

    t = opti.variable(coords.shape[0])
    t_init=np.linspace(0,1,coords.shape[0],dtype="f8")

    spline = Bspline(ctrlpts=ctrlpts, degree=3)
    pts = spline(t)

    dist = pts - coords
    residual = cas.sum(cas.dot(dist, dist))

    opti.subject_to(
        [
            opti.bounded(0.0, t, 1.0),
            opti.bounded(0.0, ctrlpts[:, 0], 1.0),
            cas.diff(t) > 0.0,
            cas.diff(ctrlpts[:, 0]) > 0.0,
            ctrlpts[0, 0] == 0.0,
            ctrlpts[1:,0]==ctrlpts_init[1:,0],
            ctrlpts[-1,1]==coords[-1,1],
            t[0] == 0.0,
            t[-1] == 1.0,
        ]
    )
    opti.minimize(residual)
    default_options = {
        "ipopt.sb": "yes",
        "ipopt.max_iter": 1000,
        "ipopt.max_cpu_time": 1e20,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.fast_step_computation": "yes",
        "detect_simple_bounds": False,
        "expand": True,
        "ipopt.print_level": 5,
    }
    opti.solver("ipopt", default_options)

    opti.set_initial(t,t_init)
    opti.set_initial(ctrlpts,ctrlpts_init)

    sol = opti.solve()

    pts = sol.value(pts)
    ctrlpts = sol.value(ctrlpts)
    print(ctrlpts)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(coords[:, 0], coords[:, 1], label="old")
    ax.plot(pts[:, 0], pts[:, 1], label="fit")
    ax.plot(ctrlpts[:, 0], ctrlpts[:, 1], "--o",label="ctrlpts")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    # test9()
    # test10()
    # test11()
    # test12()
    test13()