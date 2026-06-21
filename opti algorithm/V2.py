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
        self.gamma=1.0
        self.p_per=0.0

        self.history=Population.empty()

    def _setup(self, problem, **kwargs):
        super()._setup(problem, **kwargs)
        self.ndim=self.problem.n_var

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size, algorithm=self, random_state=self.random_state)
        # return self.Chebyshev_map_init()
    
    def _initialize_advance(self, infills=None, **kwargs):
        infills = self.survial.do(self.problem, infills)
        self.pop = infills

        self.history=Population.merge(self.history,self.pop)


    def _infill(self):
        X = self.pop.get("X")
        F = self.pop.get("F")
        # dist = squareform(pdist(X))
        hist_X,hist_F=self.history.get("X","F")

        new_X = []
        for i in range(self.pop_size):
            iX = X[[i], :]
            iF = F[[i], :]

            delta_F = hist_F - iF
            delta_X = hist_X - iX

            mask = (delta_F < 0.0).flatten()
            better_dX = delta_X[mask, :]
            better_dF = delta_F[mask, :]

            mask = ~mask
            worse_dX = delta_X[mask, :]
            worse_dF = delta_F[mask, :]

            if better_dX.shape[0] != 0:
                ind=np.argsort(better_dF*-1,axis=0).flatten()
                weight=np.arange(1,better_dF.shape[0]+1)[ind]
                main_better_eigvals, main_better_vec = self.obtain_subspace_orthonormal_basis(better_dX, weight=weight,tau=0.95)
            else:
                main_better_eigvals = np.empty((0,))
                main_better_vec = np.empty((0, self.ndim))
            if worse_dX.shape[0] != 0:
                ind=np.argsort(worse_dF,axis=0).flatten()
                weight=np.arange(1,worse_dF.shape[0]+1)[ind]
                main_worse_eigvals, main_worse_vec = self.obtain_subspace_orthonormal_basis(worse_dX, weight=weight,tau=0.95)
            else:
                main_worse_eigvals = np.empty((0,))
                main_worse_vec = np.empty((0, self.ndim))


            num_better_lambda=main_better_eigvals.shape[0]
            num_worse_lambda=main_worse_eigvals.shape[0]

            if num_better_lambda==0:
                z=self.random_state.random(size=(num_worse_lambda,1))
                vec_w=main_worse_vec@(np.diag(np.sqrt(main_worse_eigvals))@z)
                vec_final=vec_w/(np.linalg.norm(vec_w,axis=0)+1e-12)
                sss=np.sum(vec_final**2)
                L=worse_dX@vec_final
                step=-self.random_state.random()*np.max(np.abs(L))


            else:
                z=self.random_state.random(size=(num_better_lambda,1))
                vec_b=main_better_vec@(np.diag(np.sqrt(main_better_eigvals))@z)
                if num_worse_lambda==0:
                    vec_w=0.0
                else:
                    vec_w=main_worse_vec@(main_worse_vec.T@vec_b)
                vec_final=vec_b-self.gamma*vec_w
                vec_final=vec_final/(np.linalg.norm(vec_final,axis=0)+1e-12)
                sss=np.sum(vec_final**2)

                L=better_dX@vec_final
                step=self.random_state.random()*np.max(np.abs(L))

            if self.p_per>self.random_state.random():
                ...
            else:
                new_iX=iX+step*vec_final.T

                # fig=plt.figure()
                # ax=fig.add_subplot(111)
                # ax.plot(hist_X[:,0],hist_X[:,1],"o",label="pop")
                # ax.plot(iX[:,0],iX[:,1],"*",markersize=10,label="current pt")
                # ax.plot(5.5,5.5,"^",markersize=10,label="opt")
                # ax.plot(new_iX[:,0],new_iX[:,1],"+",markersize=10,label="next")
                # ax.legend()
                # plt.show()
                # plt.close()

            new_X.append(new_iX)

        new_X = np.vstack(new_X)

        if self.problem.has_bounds():
            # off = set_to_bounds_if_outside(off, *self.problem.bounds())
            new_X = repair_random_init(new_X, X, *self.problem.bounds(), random_state=self.random_state)

        off = Population.new(X=new_X)

        off = self.repair.do(self.problem, off)
        return off

        # norm_better_dX = better_dX / (np.linalg.norm(better_dX, axis=1, keepdims=True) + 1e-12)
        # norm_worse_dX = worse_dX / (np.linalg.norm(worse_dX, axis=1, keepdims=True) + 1e-12)

    def _advance(self, infills=None, **kwargs):
        off = infills
        has_improved = ImprovementReplacement().do(self.problem, self.pop, off, return_indices=True)

        self.pop[has_improved] = off[has_improved]
        self.survial.do(self.problem, self.pop)

        self.history=Population.merge(self.history,off[has_improved])

    def Chebyshev_map_init(self):
        p=self.random_state.random(size=(self.pop_size,self.ndim))
        for i in range(1,self.pop_size):
            for j in range(self.ndim):
                if p[i-1,j]<0.5:
                    p[i,j]=2*p[i-1,j]
                else:
                    p[i,j]=2*(1-p[i-1,j])
        xl, xu = self.problem.bounds()
        popX=xl+p*(xu-xl)
        pop=Population.new(X=popX)
        return pop

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
    
    def correct_orientation(self,dx_scale,eigenvectors):
        proj=np.sum(dx_scale@eigenvectors,axis=0)
        sign=np.sign(proj)
        sign[sign==0.0]=1.0
        eigenvectors_correct=eigenvectors*sign
        return eigenvectors_correct

    def pca(self,dx,weight,correct_orientation=False):
        std = np.linalg.norm(dx, axis=1, keepdims=True)
        dx = dx / (std + 1e-12)
        sss=np.sum(dx**2,axis=1)
        weight=weight/np.sum(weight)
        dx_scale=np.diag(np.sqrt(weight))@dx 
        cov=dx_scale.T@dx_scale 
        eigenvalues, eigenvectors=np.linalg.eigh(cov)

        if correct_orientation:
            eigenvectors_correct=self.correct_orientation(dx_scale,eigenvectors)
            return eigenvalues,eigenvectors_correct 
        else:
            return eigenvalues,eigenvectors
    
    def obtain_subspace_orthonormal_basis(self,dx,weight,tau=0.95):
        eigenvalues,eigenvectors_correct =self.pca(dx,weight,correct_orientation=True)
        sort_idx = np.argsort(eigenvalues)[::-1]
        sort_eigenvalues = eigenvalues[sort_idx]
        sort_eigenvectors = eigenvectors_correct[:, sort_idx]

        total_energy = np.sum(sort_eigenvalues)
        k = np.argmax(np.cumsum(sort_eigenvalues) > tau * total_energy) + 1
        k=np.minimum(k,self.ndim-1)
        
        basis_eigvals = sort_eigenvalues[:k]
        basis_vec = sort_eigenvectors[:, :k]
        return basis_eigvals,basis_vec

    def gen_orthogonal_complement(self,U):
        xi=self.random_state.random(size=(self.ndim,1))
        proj_xi=xi-U@U.T@xi
        return proj_xi


if __name__=="__main__":
    import matplotlib.pyplot as plt
    plt.ioff()

    class UDP(Problem):
        def __init__(self, n_var=2, n_obj=1, n_ieq_constr=0, n_eq_constr=0, xl=-10.0, xu=10.0, **kwargs):
            super().__init__(n_var, n_obj, n_ieq_constr, n_eq_constr, xl, xu, **kwargs)
        def _evaluate(self, x, out, *args, **kwargs):
            out["F"]=np.sum((x-5.5)**2,axis=1,keepdims=True)
    
    uda=UDA(pop_size=50)
    uda.setup(problem=UDP(),termination=("n_gen",100),seed=10)
    while uda.has_next():
        pop=uda.ask()
        uda.evaluator.eval(problem=uda.problem,pop=pop)
        uda.tell(infills=pop)
        print(uda.opt.get("X"))