from scipy.special import comb

class NxCST:
    def __init__(self, order=8):
        self.order = order

    def C_func(self, t, N1, N2):
        return t**N1 * (1 - t) ** N2

    def S_func(self, t, A):
        N = len(A) - 1
        S = 0.0
        for i, ai in enumerate(A):
            b = comb(N, i)
            si = b * t**i * (1 - t) ** (N - i)
            S += si * ai
        return S

    def write_exp(self, expfile, N1, N2, Au, Al, Te, Le, yu, yl, scaler, prefix):
        with open(expfile, "w") as fout:
            fout.write(f"{prefix}N1={N1}\n")
            fout.write(f"{prefix}N2={N2}\n")
            for idx, a in enumerate(Au):
                fout.write(f"{prefix}Au{idx}={a:.6f}\n")
            for idx, a in enumerate(Al):
                fout.write(f"{prefix}Al{idx}={a:.6f}\n")
            fout.write(f"{prefix}Te={Te:.6f}\n")
            fout.write(f"{prefix}Le={Le:.6f}\n")
            fout.write("t=1.0\n")
            fout.write(f"{prefix}scaler={scaler:.6f}\n")
            fout.write(f"[MilliMeter]{prefix}x={prefix}scaler*t\n")
            fout.write(f"[MilliMeter]{prefix}yu={prefix}scaler*({yu})\n")
            fout.write(f"[MilliMeter]{prefix}yl={prefix}scaler*({yl})\n")

    def __call__(self, prefix, expfile, N1_value, N2_value, Au_value, Al_value, Te_value, Le_value, scaler_value):
        import sympy as sp

        Au = sp.symbols(f"{prefix}Au:{self.order}")
        Al = sp.symbols(f"{prefix}Al:{self.order}")
        N1 = sp.symbols(f"{prefix}N1")
        N2 = sp.symbols(f"{prefix}N2")
        Te = sp.symbols(f"{prefix}Te")
        Le = sp.symbols(f"{prefix}Le")

        t = sp.symbols("t")

        C = self.C_func(t, N1, N2)

        Su = self.S_func(t, Au)
        yu = C * Su + t * Te / 2.0
        yu += Le * t * (1 - t) ** (self.order + 0.5)

        Sl = self.S_func(t, Al)
        yl = C * Sl - t * Te / 2.0
        yl += Le * t * (1 - t) ** (self.order + 0.5)

        yu_expr = str(yu).replace("**", "^")
        yl_expr = str(yl).replace("**", "^")

        print(yu_expr)
        print(yl_expr)

        self.write_exp(
            expfile=expfile,
            N1=N1_value,
            N2=N2_value,
            Au=Au_value,
            Al=Al_value,
            Te=Te_value,
            Le=Le_value,
            yu=yu_expr,
            yl=yl_expr,
            scaler=scaler_value,
            prefix=prefix,
        )


def exp():
    import aerosandbox as asb

    af = asb.Airfoil("naca0012").normalize()
    order = 8
    af = af.to_kulfan_airfoil(n_weights_per_side=order).set_TE_thickness(2.4e-3)

    exper = NxCST(order=order)
    exper(
        prefix="af1_",
        expfile="C:/Users/Zcaic/Desktop/cst.exp",
        N1_value=af.N1,
        N2_value=af.N2,
        Au_value=af.upper_weights,
        Al_value=af.lower_weights,
        Te_value=af.TE_thickness,
        Le_value=af.leading_edge_weight,
        scaler_value=1000.0,
    )

if __name__=="__main__":
    exp()