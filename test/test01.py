from NumOpt.airfoil.bezier import BezierAirfoil
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


def test14():
    from NumOpt.airfoil.bspline import BsplineAirfoil

    af=asb.Airfoil("n63415")

    af_fit=BsplineAirfoil.fit(upper_coordinates=af.upper_coordinates(),lower_coordinates=af.lower_coordinates(),symmetry=False)

    t=np.linspace(0,1,100)
    coords=af_fit.coordinates(t)

    import matplotlib.pyplot as plt 
    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(af.coordinates[:,0],af.coordinates[:,1],label="old")
    ax.plot(coords[:,0],coords[:,1],label="fit")
    ax.plot(af_fit.ctu[:,0],af_fit.ctu[:,1],"--o",label="ctu")
    ax.plot(af_fit.ctl[:,0],af_fit.ctl[:,1],"--o",label="ctl")
    ax.legend()
    plt.show()


def test15():
    from NumOpt.airfoil.kulfan import KulfanAirfoil 
    af=asb.Airfoil("naca0012").normalize()

    af_fit:KulfanAirfoil=KulfanAirfoil.fit(upper_coordinates=af.upper_coordinates(),lower_coordinates=af.lower_coordinates(),symmetry=True)

    t=np.linspace(0,1,100)
    coords=af_fit.coordinates(t)

    import matplotlib.pyplot as plt 
    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(af.coordinates[:,0],af.coordinates[:,1],label="old")
    ax.plot(coords[:,0],coords[:,1],label="fit")
    ax.legend()
    plt.show()

def test16():
    from NumOpt.airfoil.kulfan import KulfanAirfoil 
    af=asb.KulfanAirfoil("n63415")
    af_fit=KulfanAirfoil(Au=af.upper_weights,Al=af.lower_weights,N1=af.N1,N2=af.N2,Le=af.leading_edge_weight,Te=af.TE_thickness)
    coords=af_fit.coordinates(np.linspace(0,1,100))

    import matplotlib.pyplot as plt 
    fig=plt.figure()
    ax=fig.add_subplot()
    ax.plot(af.coordinates[:,0],af.coordinates[:,1],label="old")
    ax.plot(coords[:,0],coords[:,1],label="fit")
    ax.legend()
    plt.show()

def test17():
    opti=cas.Opti()
    N1=opti.variable(1)
    N2=opti.variable(1)

    x=0.5
    y=x**N1*(1-x)*N2

    opti.minimize(y)
    opti.subject_to([
        opti.bounded(0.0,N1,1.0),
        opti.bounded(0.0,N2,1.0)
    ])

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

    opti.solver("ipopt",default_options)

    sol=opti.solve()



def test18():
    import jax 
    import jax.numpy as jnp 

    # @jax.grad
    def func(x):
        return jnp.sin(x)
    
    y=func(0.0)
    print(y)

def test19():
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize
    import diffrax
    import optimistix as optx

    # -----------------------------
    # 1️⃣ 物理参数
    # -----------------------------
    g = 9.81          # 重力加速度 (m/s^2)
    v0 = 20.0         # 初速度 (m/s)
    c_d = 0.05        # 阻力系数 (drag coefficient)

    # -----------------------------
    # 2️⃣ 定义动力学系统 (带空气阻力)
    # 状态 y = [x, y, vx, vy]
    # -----------------------------
    def projectile_dynamics(t, y, args):
        x, y_pos, vx, vy = y
        v = jnp.sqrt(vx**2 + vy**2)
        dxdt = vx
        dydt = vy
        dvxdt = -c_d * v * vx
        dvydt = -g - c_d * v * vy
        return jnp.array([dxdt, dydt, dvxdt, dvydt])

    # -----------------------------
    # 3️⃣ 模拟轨迹直到落地
    # -----------------------------
    def simulate(theta):
        # 初始条件
        y0 = jnp.array([
            0.0,                         # x
            0.0,                         # y
            v0 * jnp.cos(theta),         # vx
            v0 * jnp.sin(theta)          # vy
        ])

        # 定义事件函数（落地条件 y = 0）
        def event_fn(t, y, args,**kwargs):
            return y[1]  # 高度 y

        event = diffrax.Event(event_fn, optx.Newton(1e-5, 1e-5, optx.rms_norm))

        # 使用Tsit5积分器（高精度 Runge-Kutta）
        solver = diffrax.Tsit5()
        term = diffrax.ODETerm(projectile_dynamics)
        saveat = diffrax.SaveAt(t1=True, dense=True)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=10.0,          # 最大模拟时间
            dt0=0.01,
            y0=y0,
            event=event,
            saveat=saveat,
            max_steps=4096
        )

        final_state = sol.ys
        x_final = final_state[0,0]  # 落地时水平位移
        return x_final

    # -----------------------------
    # 4️⃣ 定义目标函数（最小化 -射程）
    # -----------------------------
    def objective(theta_array):
        theta = theta_array[0]
        R = simulate(theta)
        return -R  # 最大化 R 等价于最小化 -R

    # -----------------------------
    # 5️⃣ 自动微分梯度
    # -----------------------------
    grad_objective = jax.jit(jax.grad(lambda t: objective(t)))

    # -----------------------------
    # 6️⃣ 使用 SLSQP 优化发射角度
    # -----------------------------
    res = minimize(
        objective,
        x0=jnp.array([0.5]),        # 初始猜测（约 30°）
        method="SLSQP",
        jac=grad_objective,
        bounds=[(0, jnp.pi / 2)]    # 限制角度在 [0°, 90°]
    )

    theta_opt = res.x[0]
    R_opt = simulate(theta_opt)

    # -----------------------------
    # 7️⃣ 输出结果
    # -----------------------------
    print("✅ 优化结果（含空气阻力）")
    print(f"最优发射角 θ = {theta_opt:.4f} rad = {theta_opt * 180 / jnp.pi:.2f}°")
    print(f"对应射程 R = {R_opt:.4f} m")
    print(f"优化是否成功: {res.success}, 迭代次数: {res.nit}")

def test20():
    from scipy.optimize import minimize,OptimizeResult 
    import jax.numpy as jnp 
    import jax 
    import numpy as np 

    class UDP():
        def __init__(self):
            self._func_grad=jax.jacobian(self._func)
        def _func(self,x):
            return jnp.sin(x)
        
        def fun(self,x):
            ret=np.array(self._func(x),dtype="f8")
            return ret
        
        def grad(self,x):
            ret=np.array(self._func_grad(x),dtype="f8")
            return ret[0]

    udp=UDP()
    sol:OptimizeResult=minimize(
        fun=udp.fun,
        x0=0.0,
        method="SLSQP",
        jac=udp.grad,
        bounds=[(0.0,jnp.pi*2.0),],
        options={"disp":True}
    )

    print(sol.x,sol.fun)


def test21():
    import jax
    import jax.numpy as jnp
    import optimistix as optx

    def func(x,args):
        x0=x[0]
        a=args[0]
        return x0**2-a,
    
    def solve_x(a):
        solver=optx.Newton(rtol=1e-6,atol=1e-6)
        sol=optx.root_find(func,solver,(1.0,),args=(a,))
        return sol.value[0]
    # print(sol.value)
    dxda=jax.grad(solve_x)
    print(dxda(4.0))
    # solver=optx.Newton(rtol=1e-6,atol=1e-6)
    # sol=optx.root_find(func,solver,(1.0,),args=(4.0,))
    # print(sol.value)

def test22():
    import numpy as np

    class ECGEO:
        """
        Entropy–Covariance Guided Evolutionary Optimization (EC-GEO)
        """
        def __init__(self, func, dim, bounds, pop_size=50, max_iter=200, 
                    sigma0=0.2, eta_g=0.3, eta_c=0.1, eta_e=0.1, top_k_ratio=0.3):
            self.func = func
            self.dim = dim
            self.bounds = np.array(bounds)
            self.pop_size = pop_size
            self.max_iter = max_iter
            self.sigma = sigma0
            self.eta_g = eta_g
            self.eta_c = eta_c
            self.eta_e = eta_e
            self.top_k_ratio = top_k_ratio

            # 初始化群体
            self.X = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], 
                                    size=(self.pop_size, self.dim))
            self.fitness = np.apply_along_axis(self.func, 1, self.X)
            self.x_best = self.X[np.argmin(self.fitness)]
            self.f_best = np.min(self.fitness)
            self.mean = np.mean(self.X, axis=0)
            self.C = np.cov(self.X.T)
            self.H_prev = 0.5 * np.log((2 * np.pi * np.e) ** self.dim * np.linalg.det(self.C + 1e-12 * np.eye(self.dim)))

        def step(self):
            # 计算协方差矩阵与特征分解
            self.mean = np.mean(self.X, axis=0)
            self.C = np.cov(self.X.T)
            eigvals, eigvecs = np.linalg.eigh(self.C)
            idx = np.argsort(eigvals)[::-1]
            eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

            # 计算当前信息熵
            H = 0.5 * np.log((2 * np.pi * np.e) ** self.dim * np.linalg.det(self.C + 1e-12 * np.eye(self.dim)))
            delta_H = H - self.H_prev
            self.H_prev = H

            # 计算自然梯度方向
            try:
                g_nat = np.linalg.solve(self.C + 1e-8 * np.eye(self.dim), (self.x_best - self.mean))
            except np.linalg.LinAlgError:
                g_nat = (self.x_best - self.mean)

            # 自适应学习率与步长
            eta_eff = self.eta_g * np.exp(-self.eta_e * delta_H)
            self.sigma *= np.exp(self.eta_c * delta_H)

            # 生成新群体
            K = max(1, int(self.dim * self.top_k_ratio))
            new_X = []
            for i in range(self.pop_size):
                noise = np.zeros(self.dim)
                for k in range(K):
                    noise += np.sqrt(abs(eigvals[k])) * eigvecs[:, k] * np.random.randn()
                x_new = self.X[i] - eta_eff * g_nat + self.sigma * noise
                # 边界处理
                x_new = np.clip(x_new, self.bounds[:, 0], self.bounds[:, 1])
                new_X.append(x_new)
            self.X = np.array(new_X)

            # 计算新适应度
            self.fitness = np.apply_along_axis(self.func, 1, self.X)
            idx_best = np.argmin(self.fitness)
            if self.fitness[idx_best] < self.f_best:
                self.f_best = self.fitness[idx_best]
                self.x_best = self.X[idx_best]

        def optimize(self, verbose=True):
            for t in range(self.max_iter):
                self.step()
                if verbose and (t % 10 == 0 or t == self.max_iter - 1):
                    print(f"Iter {t:3d} | Best fitness: {self.f_best:.6e} | σ={self.sigma:.4f}")
            return self.x_best, self.f_best

    # ----------------------------
    # 测试函数定义
    # ----------------------------

    def sphere(x):
        """Sphere test function"""
        return np.sum(x ** 2)

    def rosenbrock(x):
        """Rosenbrock function"""
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    # ----------------------------
    # 示例运行
    # ----------------------------
    np.random.seed(42)
    dim = 10
    bounds = [(-5, 5)] * dim

    print("=== Testing on Sphere function ===")
    optimizer = ECGEO(func=sphere, dim=dim, bounds=bounds, pop_size=60, max_iter=150)
    best_x, best_f = optimizer.optimize()
    print(f"Final best fitness (Sphere): {best_f:.6e}\n")

    print("=== Testing on Rosenbrock function ===")
    optimizer = ECGEO(func=rosenbrock, dim=dim, bounds=bounds, pop_size=60, max_iter=200)
    best_x, best_f = optimizer.optimize()
    print(f"Final best fitness (Rosenbrock): {best_f:.6e}")
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
    # test13()
    # test14()
    # test15()
    # test16()
    # test17()
    # test18()
    # test19()
    # test20()
    # test21()
    test22()