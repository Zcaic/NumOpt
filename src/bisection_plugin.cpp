
#define WITH_DL

#define NOMINMAX

#include "casadi/casadi.hpp"
#include "casadi/core/casadi_os.hpp"
#include "casadi/mem.h"

#include "casadi/core/rootfinder_impl.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace casadi
{
    struct BisectionMemory : public RootfinderMemory
    {
        casadi_real *x_L;
        casadi_real *x_R;
        casadi_real *x_mid;
        casadi_real *R_L;
        casadi_real *R_R;
        casadi_real *R_mid;
        casadi_int *bracket_status;
    };

    class Bisection : public Rootfinder
    {
    public:
        int max_search_;
        int max_iter_;
        Bisection(const std::string &name, const Function &f) : Rootfinder(name, f)
        {
            // max_step_ = 5.0 * casadi::pi / 180.0;
            casadi::uout() << "it is a test---zcc" << std::endl;
        }
        ~Bisection() override
        {
            clear_mem();
        }

        const char *plugin_name() const override { return "bisection"; }
        std::string class_name() const override { return "Bisection"; }

        static Rootfinder *creator(const std::string &name, const Function &f)
        {
            casadi::uout() << "it is a test---zcc" << std::endl;
            return new Bisection(name, f);
        }

        static const Options options_;
        const Options &get_options() const override { return options_; }

        // Dict get_stats(void *mem) const override
        // {
        //     Dict stats = Rootfinder::get_stats(mem);
        //     return stats;
        // }

        void init(const Dict &opts) override
        {
            casadi::uout() << "it is a test---zcc" << std::endl;
            // Rootfinder::init(opts);
            // =====================================================
            Dict linear_solver_options;
            std::string linear_solver = "qr";
            Function jac; // Jacobian of f with respect to z

            // Read options
            for (auto &&op : opts)
            {
                if (op.first == "implicit_input")
                {
                    iin_ = op.second;
                }
                else if (op.first == "implicit_output")
                {
                    iout_ = op.second;
                }
                else if (op.first == "jacobian_function")
                {
                    jac = op.second;
                }
                else if (op.first == "linear_solver_options")
                {
                    linear_solver_options = op.second;
                }
                else if (op.first == "linear_solver")
                {
                    linear_solver = op.second.to_string();
                }
                else if (op.first == "constraints")
                {
                    u_c_ = op.second;
                }
            }
            casadi::uout() << "it is a test---zccdffffffffffff" << std::endl;
            casadi::uout() << oracle_.name_in(0) << std::endl;
            // Get the number of equations and check consistency
            casadi_assert(iin_ >= 0 && iin_ < oracle_.n_in() && oracle_.n_in() > 0,
                          "Implicit input not in range");
            casadi::uout() << "11111111111111111111111111" << std::endl;
            casadi_assert(iout_ >= 0 && iout_ < oracle_.n_out() && oracle_.n_out() > 0,
                          "Implicit output not in range");
            casadi::uout() << "222222222222222222222222222222222" << std::endl;
            casadi_assert(oracle_.sparsity_out(iout_).is_dense() && oracle_.sparsity_out(iout_).is_column(),
                          "Residual must be a dense vector");
            casadi::uout() << "3333333333333333333333333333333" << std::endl;
            casadi_assert(oracle_.sparsity_in(iin_).is_dense() && oracle_.sparsity_in(iin_).is_column(),
                          "Unknown must be a dense vector");
            casadi::uout() << "444444444444444444444444444444444444444444444444" << std::endl;
            n_ = oracle_.nnz_out(iout_);
            casadi::uout() << "5555555555555555555555555555555555555" << std::endl;
            casadi_assert(n_ == oracle_.nnz_in(iin_),
                          "Dimension mismatch. Input size is " + str(oracle_.nnz_in(iin_)) + ", "
                                                                                             "while output size is " +
                              str(oracle_.nnz_out(iout_)));

            // Call the base class initializer
            casadi::uout() << "it is a test---zcaaaaaa" << std::endl;
            OracleFunction::init(opts);
            casadi::uout() << "it is a test---zccbbbbb" << std::endl;

            // Generate Jacobian if not provided
            if (jac.is_null())
            {
                std::vector<std::string> s_in = oracle_.name_in();
                std::vector<std::string> s_out = oracle_.name_out();
                s_out.insert(s_out.begin(), "jac:" + oracle_.name_out(iout_) + ":" + oracle_.name_in(iin_));
                jac = oracle_.factory(oracle_.name() + "_jac", s_in, s_out);
            }
            set_function(jac, "jac_g_x");
            sp_jac_ = jac.sparsity_out(0);
            // Check for structural singularity in the Jacobian
            casadi_assert(!sp_jac_.is_singular(),
                          "Rootfinder::init: singularity - the jacobian is structurally rank-deficient. "
                          "sprank(J)=" +
                              str(sprank(sp_jac_)) + " (instead of " + str(sp_jac_.size1()) + ")");

            // Get the linear solver creator function
            linsol_ = Linsol("linsol", linear_solver, sp_jac_, linear_solver_options);

            // Constraints
            casadi_assert(u_c_.size() == n_ || u_c_.empty(),
                          "Constraint vector if supplied, must be of length n, but got " + str(u_c_.size()) + " and n = " + str(n_));

            // Allocate sufficiently large work vectors
            alloc(oracle_);
            size_t sz_w = oracle_.sz_w();
            if (!jac.is_null())
            {
                sz_w = std::max(sz_w, jac.sz_w());
            }
            alloc_w(sz_w + 2 * static_cast<size_t>(n_));

            // =====================================================

            max_search_ = 20;
            max_iter_ = 25;
            casadi::uout() << "it is a test---zccxxxxxx" << std::endl;

            for (auto &&op : opts)
            {
                if (op.first == "max_search")
                {
                    max_search_ = op.second;
                }
                else if (op.first == "max_iter")
                {
                    max_iter_ = op.second;
                }
            }

            alloc_w(n_, true);  //  x_L
            alloc_w(n_, true);  //  x_R
            alloc_w(n_, true);  // x_mid
            alloc_w(n_, true);  // R_L
            alloc_w(n_, true);  // R_R
            alloc_w(n_, true);  // R_mid
            alloc_iw(n_, true); // bracket_status

            casadi::uout() << "it is a test---zccxxxxxxx" << std::endl;
            casadi::uout() << n_ << std::endl;
        }

        void *alloc_mem() const override { return new BisectionMemory(); }

        int init_mem(void *mem) const override
        {
            casadi::uout() << "it is a test---zccaaaaaaaaaa" << std::endl;
            // if (Rootfinder::init_mem(mem)) return 1;

            auto m = static_cast<BisectionMemory *>(mem);

            casadi::uout() << n_ << std::endl;
            // for (casadi_int i = 0; i < n_; ++i)
            // {
            //     casadi::uout()<<i<<std::endl;
            //     // m->bracket_status[i] = 0;
            // }

            casadi::uout() << "it is a test---zccaaaaaaaaaa" << std::endl;
            return 0;
        }

        void free_mem(void *mem) const override
        {
            delete static_cast<BisectionMemory *>(mem);
        }

        void set_work(void *mem, const double **&arg, double **&res, casadi_int *&iw, double *&w) const override
        {
            Rootfinder::set_work(mem, arg, res, iw, w);
            auto m = static_cast<BisectionMemory *>(mem);
            casadi_int n = nnz_in(0);

            // clang-format off
            m->x_L = w;w += n;
            m->x_R = w;w += n;
            m->x_mid = w;w += n;
            m->R_L = w;w += n;
            m->R_R = w; w += n;
            m->R_mid = w; w += n;

            m->bracket_status = iw;iw += n;
            // clang-format on
        }

        int solve(void *mem) const override
        {
            auto m = static_cast<BisectionMemory *>(mem);
            casadi_int n = nnz_in(0);

            const casadi_real *x0 = m->arg[0];
            const casadi_real *p = m->arg[1];
            casadi_real *x_sol = m->res[0];

            const casadi_real *eval_arg[2];
            casadi_real *eval_res[1];
            eval_arg[1] = p;

            casadi_real step = casadi::pi / max_search_;

            for (casadi_int i = 0; i < n; ++i)
            {
                m->x_L[i] = x0[i] - step;
                m->x_R[i] = x0[i] + step;
                m->bracket_status[i] = 0;
            }

            eval_arg[0] = m->x_L;
            eval_res[0] = m->R_L;
            oracle_(eval_arg, eval_res, m->iw, m->w, 0);

            eval_arg[0] = m->x_R;
            eval_res[0] = m->R_R;
            oracle_(eval_arg, eval_res, m->iw, m->w, 0);

            // int max_bracket_iter = 15;
            bool all_bracketed = false;

            for (int iter = 0; iter < max_search_ && !all_bracketed; ++iter)
            {
                all_bracketed = true;
                for (casadi_int i = 0; i < n; ++i)
                {
                    if (m->bracket_status[i] == 0)
                    {
                        if (m->R_L[i] * m->R_R[i] <= 0.0)
                        {
                            m->bracket_status[i] = 1;
                        }
                        else
                        {
                            all_bracketed = false;
                            m->x_L[i] = std::max(-85.0 * casadi::pi / 180.0, m->x_L[i] - step);
                            m->x_R[i] = std::min(85.0 * casadi::pi / 180.0, m->x_R[i] + step);
                        }
                    }
                }

                if (!all_bracketed)
                {
                    eval_arg[0] = m->x_L;
                    eval_res[0] = m->R_L;
                    oracle_(eval_arg, eval_res, m->iw, m->w, 0);

                    eval_arg[0] = m->x_R;
                    eval_res[0] = m->R_R;
                    oracle_(eval_arg, eval_res, m->iw, m->w, 0);
                }
            }

            if (!all_bracketed)
            {
                for (casadi_int i = 0; i < n; ++i)
                {
                    if (m->bracket_status[i] == 0)
                    {
                        casadi::uout() << "[Bisection Error] Section " << i
                                       << " failed to bracket root! "
                                       << "x_L = " << m->x_L[i] << ", R_L = " << m->R_L[i]
                                       << " | x_R = " << m->x_R[i] << ", R_R = " << m->R_R[i]
                                       << std::endl;
                    }
                }
                return 1;
            }

            // int bisection_iters = 40;
            for (int iter = 0; iter < max_iter_; ++iter)
            {
                for (casadi_int i = 0; i < n; ++i)
                {
                    m->x_mid[i] = 0.5 * (m->x_L[i] + m->x_R[i]);
                }

                eval_arg[0] = m->x_mid;
                eval_res[0] = m->R_mid;
                oracle_(eval_arg, eval_res, m->iw, m->w, 0);

                for (casadi_int i = 0; i < n; ++i)
                {
                    if (m->R_L[i] * m->R_mid[i] > 0.0)
                    {
                        m->x_L[i] = m->x_mid[i];
                        m->R_L[i] = m->R_mid[i];
                    }
                    else
                    {
                        m->x_R[i] = m->x_mid[i];
                        m->R_R[i] = m->R_mid[i];
                    }
                }
            }

            for (casadi_int i = 0; i < n; ++i)
            {
                x_sol[i] = 0.5 * (m->x_L[i] + m->x_R[i]);
            }

            return 0;
        }

        void serialize_body(SerializingStream &s) const override
        {
            Rootfinder::serialize_body(s);
            s.version("Bisection", 1);
            s.pack("Bisection::max_search", max_search_);
            s.pack("Bisection::max_iter", max_iter_);
        }

        static ProtoFunction *deserialize(DeserializingStream &s) { return new Bisection(s); }

    protected:
        explicit Bisection(DeserializingStream &s) : Rootfinder(s)
        {
            s.version("Bisection", 1);
            s.unpack("Bisection::max_search", max_search_);
            s.unpack("Bisection::max_iter", max_iter_);
        }
    };

    const Options Bisection::options_ = {{&Rootfinder::options_},
                                         {{"max_search", {OT_INT, "Maximum search numbers"}},
                                          {"max_iter", {OT_INT, "Maximum iter numbers"}}}

    };

#ifndef CASADI_SYMBOL_EXPORT
#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#define CASADI_SYMBOL_EXPORT __declspec(dllexport)
#else
#define CASADI_SYMBOL_EXPORT __attribute__((visibility("default")))
#endif
#endif
    // =================================================================

    // =================================================================
    extern "C"
    {
        CASADI_SYMBOL_EXPORT int casadi_register_rootfinder_bisection(Rootfinder::Plugin *plugin)
        {
            plugin->creator = Bisection::creator;
            plugin->name = "bisection";
            plugin->doc = "Custom Bisection Rootfinder Plugin";
            plugin->version = 31;
            plugin->options = &Bisection::options_;
            plugin->deserialize = &Bisection::deserialize;
            return 0;
        }

        CASADI_SYMBOL_EXPORT void casadi_load_rootfinder_bisection() {}
    }

} // namespace casadi