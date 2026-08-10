import casadi as ca 
from NumOpt import Opti

class Root(ca.Callback):
    def __init__(self, name,x0, g:ca.Function, opts={"enable_fd": False}):
        ca.Callback.__init__(self)

        self.x0=x0
        self.Rfunc=g
        self.x_shape=g.size_in("x")
        self.p_shape=g.size_in("p")
        self.R_shape=g.size_out("R")
        self.dR_dx,self.dR_dp,self.dx_dp,self.ddx_dp2,self.x_star_func=self.__gen()

        if ca.DM(x0).shape!=self.x_shape:
            raise ValueError("x0 shape must be equal x...")

        self.construct(name, opts)

    def __gen(self):
        dumpy_x=ca.MX.sym("dumpy_x",*self.x_shape)
        dumpy_p=ca.MX.sym("dumpy_p",*self.p_shape)
        R=self.Rfunc(x=dumpy_x,p=dumpy_p)["R"]

        dR_dx=ca.jacobian(R,dumpy_x)
        dR_dp=ca.jacobian(R,dumpy_p)
        dx_dp=-ca.solve(dR_dx,dR_dp)

        ddxdp_dp=ca.jacobian(dx_dp,dumpy_p)
        ddxdp_dx=ca.jacobian(dx_dp,dumpy_x)

        ddx_dp2=ddxdp_dp+ddxdp_dx@dx_dp

        dR_dx=ca.Function("dR_dx",[dumpy_x,dumpy_p],[dR_dx])
        dR_dp=ca.Function("dR_dy",[dumpy_x,dumpy_p],[dR_dp])
        dx_dp=ca.Function("dx_dp",[dumpy_x,dumpy_p],[dx_dp])
        ddx_dp2=ca.Function("ddx_dp2",[dumpy_x,dumpy_p],[ddx_dp2])

        opti=Opti()
        x=opti.variable(init_guess=self.x0)
        p=opti.parameter(value=ca.DM(*self.p_shape))
        R=self.Rfunc(x=x,p=p)["R"]
        opti.subject_to(R==0.0)
        opti.ipopt_solver(verbose=False)
        x_star_func=opti.to_function("x_star_func",[x,p],[x],["x0","p"],["x_star"])
        del opti

        return dR_dx,dR_dp,dx_dp,ddx_dp2,x_star_func

    def get_n_in(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(*self.p_shape)

    def get_n_out(self):
        return 1

    def get_sparsity_out(self, i):
        return ca.Sparsity.dense(*self.x_shape)
    
    def eval(self, args):
        # opti=Opti()
        p=args[0]

        # x=opti.variable(init_guess=self.x0)
        # p=opti.parameter(value=p0)
        # R=self.Rfunc(x=x,p=p)["R"]
        # opti.subject_to(R==0.0)
        # opti.ipopt_solver(verbose=False)
        # sol=opti.solve()

        # self.x_star=sol.value(x)
        # self.R_star=sol.value(R)
        # self.P_star=sol.value(p)
        x_star=self.x_star_func(x0=self.x0,p=p)["x_star"]
        
        return [x_star]

    def has_jacobian(self):
        return True

    def get_jacobian(self, name, inames, onames, opts):
        class JacFun(ca.Callback):
            def __init__(self_jac, opts={"enable_fd": False}):
                ca.Callback.__init__(self_jac)
                self_jac.construct(name, opts)

            def get_n_in(self_jac):
                return 2

            def get_n_out(self_jac):
                return 1

            def get_sparsity_in(self_jac, i):
                if i == 0:
                    return ca.Sparsity.dense(*self.p_shape)
                elif i == 1:
                    return ca.Sparsity.dense(*self.x_shape)

            def get_sparsity_out(self_jac, i):
                xsize=self.x_shape[0]*self.x_shape[1]
                psize=self.p_shape[0]*self.p_shape[1]
                return ca.Sparsity.dense(psize,xsize)

            def eval(self_jac, args):
                p=args[0]
                x_star=args[1]

                dx_dp=self.dx_dp(x_star,p)

                # dR_dx=self.dR_dx(x_star,p)
                # dR_dp=self.dR_dp(x_star,p)

                # dx_star_dp=-np.linalg.solve(dR_dx,dR_dp)

                return [dx_dp]

            def has_jacobian(self_jac):
                return True

            def get_jacobian(self_jac, name, inames, onames, opts):
                class HessFun(ca.Callback):
                    def __init__(self_hess, opts={}):
                        ca.Callback.__init__(self_hess)
                        self_hess.construct(name, opts)

                    def get_n_in(self_hess):
                        return 3

                    def get_n_out(self_hess):
                        return 2

                    def get_sparsity_in(self_hess, i):
                        if i == 0:
                            return ca.Sparsity.dense(*self.p_shape) #p
                        elif i == 1:  
                            return ca.Sparsity.dense(*self.x_shape) #x
                        elif i == 2:
                            xsize=self.x_shape[0]*self.x_shape[1]
                            psize=self.p_shape[0]*self.p_shape[1]
                            return ca.Sparsity.dense(xsize,psize) # dx_dp

                    def get_sparsity_out(self_hess, i):
                        xsize=self.x_shape[0]*self.x_shape[1]
                        psize=self.p_shape[0]*self.p_shape[1]
                        if i == 0:
                            return ca.Sparsity.dense(xsize*psize,psize) # ddx_dp2
                        elif i == 1:
                            return ca.Sparsity.dense(xsize*psize,xsize) # ddx_dpx

                    def eval(self_hess, args):
                        xsize=self.x_shape[0]*self.x_shape[1]
                        psize=self.p_shape[0]*self.p_shape[1]
                        p = args[0]
                        x_star=args[1]

                        ddx_dp2=self.ddx_dp2(x_star,p)
                        ddx_dpx=ca.DM.zeros(xsize*psize,xsize)

                        return [ddx_dp2, ddx_dpx]

                self_jac.jac_callback = HessFun()
                return self_jac.jac_callback

        self.jac_callback = JacFun()
        return self.jac_callback 
    
def test01():
    x=ca.MX.sym('x')
    p=ca.MX.sym("p")
    g=ca.Function("g",[x,p],[x**2-p],["x","p"],["R"]) 

    root=Root("root",x0=0.1,g=g)

    print(root(10))


    dx_dp=ca.jacobian(root(p),p)
    dx_dp=ca.Function("dx_dp",[p],[dx_dp])

    ddx_dp2=ca.jacobian(dx_dp(p),p)
    ddx_dp2=ca.Function("ddx_dp2",[p],[ddx_dp2])

    print(dx_dp(10.0))
    # print(-0.5*10.0**(-0.5))

    print(ddx_dp2(10.0))
    # print(0.25*10**(-1.5))

def test02():
    def imp():
        x=ca.MX.sym('x')
        p=ca.MX.sym("p")
        g=ca.Function("g",[x,p],[x**2-p],["x","p"],["R"])
        root=Root("root",x0=2.0,g=g)
        return root

    root=imp()

    opti=Opti()
    p=opti.variable(init_guess=5.0,lower_bound=1.0,upper_bound=20.0)
    x_star=root(p)
    # print(x_star.shape)
    opti.minimize(x_star)
    opti.ipopt_solver()
    sol=opti.solve()
    print(sol.value(p))
if __name__=="__main__":
    # test01()
    test02()