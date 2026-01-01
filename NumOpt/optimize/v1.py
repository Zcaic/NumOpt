import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.repair import NoRepair
from pymoo.core.initialization import Initialization
from pymoo.core.survival import Survival


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
        K=2,
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
        self.K=K
        self.repair = repair
        self.initialization = Initialization(sampling)
        self.pop_size = pop_size

    def _setup(self, problem, **kwargs):
        return super()._setup(problem, **kwargs)

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        # self.pop = self.survial.do(self.problem, infills)
        x = infills.get("X")
        C = np.cov(m=x, ddof=1)
        eigenvalues, eigenvectors = np.linalg.eigh(C)
