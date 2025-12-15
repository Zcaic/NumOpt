from ..opti import asb, anp, cas, np


class Bspline:
    def __init__(self, ctrlpts, degree=3):
        self.ctrlpts = ctrlpts
        self.n = self.ctrlpts.shape[0] - 1
        self.degree = degree
        self.order = self.degree + 1
        self.segments = self.n - self.degree + 1
        self.knots = self.quasi_uniform_knots()

    def quasi_uniform_knots(self):
        middle = np.linspace(0, 1, self.segments + 1)
        start = np.zeros(self.degree, dtype="f8")
        end = np.ones(self.degree, dtype="f8")
        knots = np.hstack([start, middle, end])
        return knots

    @staticmethod
    def __deBoor(t, i, k, knots):
        if k == 1:
            return cas.if_else(cas.logic_and(t >= knots[i], t < knots[i + 1]), 1.0, 0.0)
        else:
            term1 = term2 = 0.0
            delta_t1 = knots[i + k - 1] - knots[i]
            delta_t2 = knots[i + k] - knots[i + 1]

            if delta_t1 != 0.0:
                term1 = (t - knots[i]) / (delta_t1) * Bspline.__deBoor(t, i, k - 1, knots)

            if delta_t2 != 0.0:
                term2 = (knots[i + k] - t) / (delta_t2) * Bspline.__deBoor(t, i + 1, k - 1, knots)

            return term1 + term2

    def N_coef(self, t, i):
        return Bspline.__deBoor(t, i, self.order, self.knots)

    def __call__(self, t):
        def func():
            tmp = 0.0
            for j in range(self.n + 1):
                N_coef = self.N_coef(u, j)
                tmp = tmp + N_coef * self.ctrlpts[j : j + 1, :]
            return tmp

        nts = t.shape[0]
        pts = [0.0] * nts

        for i in range(nts):
            u = t[i]
            pts[i] = cas.if_else(u == 1.0, self.ctrlpts[-1:, :], func())
            # if u == 1.0:
            #     pts[i] = self.ctrlpts[-1:, :]
            # else:
            #     for j in range(self.n + 1):
            #         N_coef = self.N_coef(u, j)
            #         pts[i] = pts[i] + N_coef * self.ctrlpts[j : j + 1, :]
        pts = cas.vcat(pts)
        return pts
