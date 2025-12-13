import sympy as sp
from sympy.printing.str import StrPrinter
import aerosandbox as asb
from scipy.special import comb 
import aerosandbox.numpy as anp


class CustomerPrinter(StrPrinter):
    def _print_Pow(self, expr):
        return f"{expr.base}^{expr.exp}"


# def comb(N, i):
#     return (sp.factorial(N)) / (sp.factorial(i) * sp.factorial(N - i))


def C_func(t, N1, N2):
    return t**N1 * (1 - t) ** N2


def S_func(t, A):
    N = len(A) - 1
    S = 0.0
    for i, ai in enumerate(A):
        b = comb(N, i)
        si = b * t**i * (1 - t) ** (N - i)
        S += si * ai
    return S


def write_exp(file, N1, N2, Au, Al, Thickness,Le, yu, yl,prefix):
    with open(file, "w") as fout:
        fout.write(f"{prefix}N1={N1}\n")
        fout.write(f"{prefix}N2={N2}\n")
        for idx, a in enumerate(Au):
            fout.write(f"{prefix}Au{idx}={a:.6f}\n")
        for idx, a in enumerate(Al):
            fout.write(f"{prefix}Al{idx}={a:.6f}\n")
        fout.write(f"{prefix}Te={Thickness:.6f}\n")
        fout.write(f"{prefix}Le={Le:.6f}\n")
        fout.write(f"t=1.0\n")
        fout.write(f"{prefix}scaler=1000.0\n")
        fout.write(f"[MilliMeter]{prefix}x={prefix}scaler*t\n")
        fout.write(f"[MilliMeter]{prefix}yu={prefix}scaler*({yu})\n")
        fout.write(f"[MilliMeter]{prefix}yl={prefix}scaler*({yl})\n")


def cst2nx():
    order = 8
    prefix="af0_"
    Au = sp.symbols(f"{prefix}Au:{order}")
    Al = sp.symbols(f"{prefix}Al:{order}")
    N1 = sp.symbols(f"{prefix}N1")
    N2 = sp.symbols(f"{prefix}N2")
    Te = sp.symbols(f"{prefix}Te")
    Le = sp.symbols(f"{prefix}Le")
    # expr=x**2

    t = sp.symbols("t")

    C = C_func(t,N1,N2)

    Su = S_func(t, Au)+Le*t**0.5*(1-t)**(order-0.5)
    yu = C * Su + t * Te / 2.0

    Sl = S_func(t, Al)+Le*t**0.5*(1-t)**(order-0.5)
    yl = C * Sl - t * Te / 2.0

    # print(yu.subs(t,1))

    # print(yu)
    yu_expr = str(yu).replace("**","^")
    yl_expr = str(yl).replace("**","^")

    print(yu_expr)
    print(yl_expr)

    af = asb.Airfoil("n63415").normalize().set_TE_thickness(0.0).to_kulfan_airfoil(n_weights_per_side=order)
    # af.leading_edge_weight=1.0
    # print(af.lower_coordinates(anp.cosspace(0,1,100)))
    write_exp(
        "C:/Users/Zcaic/Desktop/cst.exp",
        N1=af.N1,
        N2=af.N2,
        Au=af.upper_weights,
        Al=af.lower_weights,
        Thickness=af.TE_thickness,
        Le=af.leading_edge_weight,
        yu=yu_expr,
        yl=yl_expr,
        prefix=prefix
    )
