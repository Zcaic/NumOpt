import numpy as np
from scipy.spatial.distance import pdist, squareform

from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.repair import NoRepair
from pymoo.core.initialization import Initialization
from pymoo.core.survival import Survival
from pymoo.core.population import Population
from pymoo.operators.repair.bounds_repair import repair_random_init
from pymoo.core.replacement import ImprovementReplacement


class FitnessSurvival(Survival):

    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        F, cv = pop.get("F", "cv")
        assert F.shape[1] == 1, "FitnessSurvival can only used for single objective single!"
        S = np.lexsort([F[:, 0], cv])
        pop.set("rank", np.argsort(S))
        return pop[S[:n_survive]]


class UDA(Algorithm):
    def __init__(
        self,
        pop_size=100,
        sampling=LHS(),
        repair=NoRepair(),
        termination=None,
        output=None,
        display=None,
        callback=None,
        archive=None,
        return_least_infeasible=False,
        save_history=False,
        verbose=False,
        seed=None,
        evaluator=None,
        **kwargs,
    ):
        super().__init__(
            termination,
            output,
            display,
            callback,
            archive,
            return_least_infeasible,
            save_history,
            verbose,
            seed,
            evaluator,
            **kwargs,
        )
        self.repair = repair
        self.initialization = Initialization(sampling)
        self.pop_size = pop_size
        self.survial = FitnessSurvival()

    def _setup(self, problem, **kwargs):
        return super()._setup(problem, **kwargs)

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    @staticmethod
    def cov(X, Y, idx, mean=True):
        xi = X - X[[idx]]
        yi = Y - Y[[idx]]
        ret = xi.T @ yi
        if mean:
            n_samples = X.shape[0]
            ret = ret / (n_samples - 1)
        return ret

    def _initialize_advance(self, infills=None, **kwargs):
        # self.pop = self.survial.do(self.problem, infills)
        infills = self.survial.do(self.problem, infills)
        self.pop = infills
        # X=infills.get("X")
        # F=infills.get("Y")
        # pop_size=infills.shape[0]

    def _infill(self):
        X = self.pop.get("X")
        F = self.pop.get("F")
        ndim = self.problem.n_var
        dist = squareform(pdist(X))

        new_X = []
        for i in range(self.pop_size):
            iX = X[[i], :]
            iF = F[[i], :]

            delta_F = F - iF
            delta_X = X - iX

            mask = (delta_F < 0.0).flatten()
            better_dX = delta_X[mask, :]
            better_dF = delta_F[mask, :]

            mask = (delta_F > 0.0).flatten()
            worse_dX = delta_X[mask, :]
            worse_dF = delta_F[mask, :]

            if better_dX.shape[0] != 0:
                main_better_eigvals, main_better_vec = self._get_main_vector(better_dX, include=0.99)
            else:
                main_better_eigvals = np.empty((0,))
                main_better_vec = np.empty((0, ndim))
            if worse_dX.shape[0] != 0:
                main_worse_eigvals, main_worse_vec = self._get_main_vector(worse_dX, include=0.99)
            else:
                main_worse_eigvals = np.empty((0,))
                main_worse_vec = np.empty((0, ndim))

            better_vec_or_worse = self.select_better_or_worse(better_num=better_dX.shape[0], worse_num=worse_dX.shape[0])
            if better_vec_or_worse:  # better
                ...
            else:  # worse
                num_vec = main_worse_vec.shape[0]
                weights=self.gen_weights(num_vec)
                tot_vec=np.sum(main_worse_vec*weights,axis=0,keepdims=True)
                project_step=worse_dX@tot_vec
                



            project_step = delta_X @ main_better_vec
            project_step = np.sort(project_step, axis=0)[::-1]
            idx = int(np.floor(0.1 * delta_X.shape[0]))
            main_step = project_step[idx, :]

            selecte_id = self.roulette_wheel_selection(main_better_eigvals)
            selecte_vec = main_better_vec[:, selecte_id]
            selecte_step = main_step[selecte_id]

            dx = self.random_state.random() * selecte_step * selecte_vec
            new_ix = ix + dx
            new_X.append(new_ix)

        new_X = np.array(new_X)

        if self.problem.has_bounds():
            # off = set_to_bounds_if_outside(off, *self.problem.bounds())
            off = repair_random_init(off, X, *self.problem.bounds(), random_state=self.random_state)

        off = Population.new(X=off)

        off = self.repair.do(self.problem, off)
        return off

        # norm_better_dX = better_dX / (np.linalg.norm(better_dX, axis=1, keepdims=True) + 1e-12)
        # norm_worse_dX = worse_dX / (np.linalg.norm(worse_dX, axis=1, keepdims=True) + 1e-12)

    def _advance(self, infills=None, **kwargs):
        off = infills
        has_improved = ImprovementReplacement().do(self.problem, self.pop, off, return_indices=True)

        self.pop[has_improved] = off[has_improved]
        self.survial.do(self.problem, self.pop)

    def gen_weights(self, num, method="sort"):
        if method == "sort":
            if num == 1:
                return 1.0
            else:
                r = np.random.random(num - 1)
                r = np.hstack([[0.0], r, [1.0]])
                ret = np.diff(r)
                return ret
        else:
            r = np.random.random(num)
            ret = r / np.sum(r)
            return ret

    def select_better_or_worse(self, better_num, worse_num):
        prob = np.array([better_num, worse_num])
        idx = self.roulette_wheel_selection(prob)
        if idx == 0:
            return True
        else:
            return False

    def roulette_wheel_selection(self, prob) -> int:
        prob = prob / np.sum(prob)
        cum_prob = np.cumsum(prob)
        cum_prob = np.insert(cum_prob, 0, 0.0)
        idx_selection = self.random_state.random()
        idx = np.argmax(idx_selection < cum_prob) - 1
        return idx

    def _normalization(self, vec):
        std = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / (std + 1e-12)
        return vec

    def _get_main_vector(self, vectors, include=0.99):
        # dF=np.abs(dF)
        # dF_weight=dF/np.sum(dF).reshape(-1,1)
        vectors = self._normalization(vectors)
        # vectors=np.sqrt(dF_weight)*vectors
        cov = vectors.T @ vectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sort_idx = np.argsort(eigenvalues)[::-1]
        sort_eigenvalues = eigenvalues[sort_idx]
        sort_eigenvectors = eigenvectors[:, sort_idx]

        total_energy = np.sum(sort_eigenvalues)
        k = np.argmax(np.cumsum(sort_eigenvalues) > include * total_energy) + 1
        main_vec = sort_eigenvectors[:, :k]
        main_eigvals = sort_eigenvalues[:k]
        return main_eigvals, main_vec
