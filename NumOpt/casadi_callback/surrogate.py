import casadi as ca
import aerosandbox.numpy as anp
from typing import Literal
from smt.surrogate_models.krg import KRG


class Surrogate:
    def __init__(self, sm: KRG):
        self.sm = sm

    @property
    def nx(self):
        return self.sm.nx

    def predict(self, x):
        return self.sm.predict_values(x)

    def predict_derivate(self, x):
        gradient = anp.array([self.sm.predict_derivatives(x, kx) for kx in range(self.nx)])
        return gradient

    def predict_variances(self, x):
        return self.sm.predict_variances(x)

    def predict_variance_derivatives(self, x):
        gradient = anp.array([self.sm.predict_variance_derivatives(x, kx) for kx in range(self.nx)])
        return gradient


class Surrogate2Callback(ca.Callback):
    def __init__(self, name, surrogate: Surrogate,nsamples=1, output_kind: Literal["predict", "predict_variances"] = "predict", opts={}):
        ca.Callback.__init__(self)
        self.surrogate = surrogate
        self.nsamples=nsamples
        if output_kind == "predict":
            self.output = self.surrogate.predict
            self.output_der = self.surrogate.predict_derivate
        else:
            self.output = self.surrogate.predict_variances
            self.output_der = self.surrogate.predict_variance_derivatives
        self.construct(name, opts)
        self.jac_callback = self._jac(name)

    def get_n_in(self):
        return 1

    def get_sparsity_in(self, i):
        # return ca.Sparsity.dense(self.surrogate.nx, 1)
        return ca.Sparsity.dense(self.nsamples,self.surrogate.nx)

    def get_n_out(self):
        return 1

    def get_sparsity_out(self, i):
        # return ca.Sparsity.dense(1, 1)
        return ca.Sparsity.dense(self.nsamples, 1)

    def eval(self, args):
        x = args[0]
        x = x.toarray()
        y = self.output(x)
        return [y]

    def has_jacobian(self):
        return True
    
    def _jac(self,name):
        class JacFun(ca.Callback):
            def __init__(self_jac, opts={"enable_fd": False}):
                ca.Callback.__init__(self_jac)
                self_jac.construct(f"jac_{name}", opts)

            def get_n_in(self_jac):
                return 2

            def get_n_out(self_jac):
                return 1

            def get_sparsity_in(self_jac, i):
                if i == 0:
                    return ca.Sparsity.dense(self.surrogate.nx, 1)
                elif i == 1:
                    return ca.Sparsity.dense(1, 1)

            def get_sparsity_out(self_jac, i):
                return ca.Sparsity.dense(1, self.surrogate.nx)

            def eval(self_jac, args):
                x = args[0]
                x = x.toarray()
                y_jac = self.output_der(x)
                return [y_jac]
        return JacFun()

    def get_jacobian(self, name, inames, onames, opts):
        # class JacFun(ca.Callback):
        #     def __init__(self_jac, opts={"enable_fd": False}):
        #         ca.Callback.__init__(self_jac)
        #         self_jac.construct(name, opts)

        #     def get_n_in(self_jac):
        #         return 2

        #     def get_n_out(self_jac):
        #         return 1

        #     def get_sparsity_in(self_jac, i):
        #         if i == 0:
        #             return ca.Sparsity.dense(self.surrogate.nx, 1)
        #         elif i == 1:
        #             return ca.Sparsity.dense(1, 1)

        #     def get_sparsity_out(self_jac, i):
        #         return ca.Sparsity.dense(1, self.surrogate.nx)

        #     def eval(self_jac, args):
        #         x = args[0]
        #         x = x.toarray()
        #         y_jac = self.output_der(x)
        #         return [y_jac]

        # self.jac_callback = JacFun()
        return self.jac_callback


if __name__ == "__main__":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import aerosandbox as asb 
    import time

    xt = anp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    yt = anp.array([0.0, 1.0, 1.5, 0.9, 1.0])

    sm = KRG(theta0=[1e-2], print_global=False)
    sm.set_training_values(xt, yt)
    sm.train()

    N=10000

    sm = Surrogate(sm)
    sm_predict = Surrogate2Callback("sm", surrogate=sm,nsamples=1)
    # dumpy=ca.GenSX_sym("dumpy",1,1)
    # sm_predict_mpi=ca.Function("mpi",[dumpy],[sm_predict(dumpy)]).map(N,"thread")
    # print(sm.predict_derivate(0.0))


    # X=anp.linspace(0,4,N).reshape((-1,1))
    # time_s=time.time()
    # y=sm.predict(X)
    # print(time.time()-time_s,"s")
    # # print(y)

    # time_s=time.time()
    # y=sm_predict_mpi(X.T)
    # print(time.time()-time_s,"s")

    # time_s=time.time()
    # y=sm_predict(X)
    # print(time.time()-time_s,"s")

    opti = asb.Opti()
    x = opti.variable(init_guess=1.0, lower_bound=0.0, upper_bound=4.0)
    obj = sm_predict(x)
    opti.maximize(obj)
    sol = opti.solve(options={"ipopt.hessian_approximation": "limited-memory"})
    xopt = sol(x)
    yopt = sol(obj)
    print(sol(x), sol(obj))

    # num = 100
    # x = anp.linspace(0.0, 4.0, num)
    # y=sm_predict(x).toarray()
    # print(y)

    # fig=make_subplots(1,1)
    # fig.add_scatter(x=x,y=y[:,0],mode="lines+markers",row=1,col=1,line={"shape":"spline","width":2})
    # fig.add_scatter(x=[xopt],y=[yopt],mode="markers",row=1,col=1,marker={"symbol":"star","size":20})
    # fig.update_layout()
    # fig.show(config={"scrollZoom":True})
    # fig.write_html("./fig.html",config={"scrollZoom":True},auto_open=True)
