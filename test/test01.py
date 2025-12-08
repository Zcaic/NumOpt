from NumOpt.airfoil.bezier import Bezier, BezierAirfoil
from NumOpt.opti import cas,Opti
import numpy as np
import aerosandbox as asb


def test1():
    opti=Opti()
    T=opti.variable(init_guess=2.8e3*9.8)

    A=np.pi*3.5**2
    vi=(T/(2*1.225*A))**0.5

    Pin=743e3

    eta=T*vi/Pin 

    opti.subject_to([
        eta==0.80
    ])

    opti.solver()
    sol=opti.solve()
    print(sol(T))

def test2():
    af=asb.Airfoil("naca0012")
    af_new=BezierAirfoil.fit(upper_coordinates=af.upper_coordinates(),lower_coordinates=af.lower_coordinates(),symmetry=True)
    af_new=af_new.set_thickness(5e-3)
    print(af_new.thickness)
    coords = af_new.coordinates(np.linspace(0, 1, 100))

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(coords[:, 0], coords[:, 1],label="fit")
    ax.plot(af.coordinates[:,0],af.coordinates[:,1],label="ori")
    ax.plot(af_new.ctu[:,0],af_new.ctu[:,1],"o--",label="ctu")
    ax.plot(af_new.ctl[:,0],af_new.ctl[:,1],"o--",label="ctl")
    ax.legend()
    plt.show()



def test3():
    nx = 5

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
                I_alpha * p**2 + (2 * a *a) * pi * r * v * b**3 * p + K_alpha - (2 * a + 1) * pi * r * v**2 * b**2,
            ],
        ]
    )

    detA=A.det()

    p_expr=sp.solve(detA,p)

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


if __name__ == "__main__":
    # test1()
    test2()
    # test3()
    # test4()
    # test5()
    # test6()
