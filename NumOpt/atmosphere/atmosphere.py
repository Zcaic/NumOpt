def sutherland(Temperature, T_ref=273.15, mu_ref=1.716e-5, S=110.0):
    """_summary_

    Parameters
    ----------
    Temperature : float

    T_ref : float, optional
        reference Temperature, by default 273.15
    mu_ref : float, optional
        reference viscosity, by default 1.716e-5
    S : float, optional
        the Sutherland Constant, by default 110.0
    """

    mu = mu_ref * (Temperature / T_ref) ** 1.5 * ((T_ref + S) / (Temperature + S))
    return mu
