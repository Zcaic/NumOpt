
#define WITH_DL
#define NOMINMAX


#include "casadi/casadi.hpp"
#include "casadi/core/rootfinder_impl.hpp"
#include <cmath>
#include <limits>
#include <string>

namespace casadi
{

    // =================================================================
    // Memory结构体
    // =================================================================
    struct BisectionMemory : public RootfinderMemory
    {
        // 返回状态码：
        //  0  : 达到最大迭代次数
        //  1  : 区间宽度收敛 |b-a| < abstol_step
        //  2  : 残差收敛     |f(mid)| < abstol
        // -1  : 求值返回NaN
        // -2  : 初始bracket不合法 f(lb)*f(ub) > 0
        int return_status;

        // 实际执行的迭代次数
        casadi_int iter;

        // 统计用：最后一次中点的函数值和区间宽度
        double f_mid;
        double bracket_width;
    };

    // =================================================================
    // 主类声明 + 实现
    // =================================================================
    class Bisection : public Rootfinder
    {
    public:
        // ---------------------------------------------------------------
        // 构造 / 析构
        // ---------------------------------------------------------------
        explicit Bisection(const std::string &name, const Function &f)
            : Rootfinder(name, f) {
                casadi::uout()<<"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"<<std::endl;
                casadi::uout()<<f.name_in(0)<<std::endl;
            }

        ~Bisection() override { clear_mem(); }

        // ---------------------------------------------------------------
        // 插件标识接口
        // ---------------------------------------------------------------
        const char *plugin_name() const override { return "bisection"; }
        std::string class_name() const override { return "Bisection"; }

        static Rootfinder *creator(const std::string &name, const Function &f)
        {
            return new Bisection(name, f);
        }

        // ---------------------------------------------------------------
        // 选项 / 文档
        // ---------------------------------------------------------------
        static const Options options_;
        const Options &get_options() const override { return options_; }
        static const std::string meta_doc;

        // ---------------------------------------------------------------
        // 初始化
        // ---------------------------------------------------------------
        void init(const Dict &opts) override
        {
            // 1. 父类init（初始化oracle_, iin_, iout_, n_等）
            casadi::uout()<<"xxxxxxxxxxxxxxxxxx"<<std::endl;

            casadi::uout()<<oracle_.name_in(0)<<std::endl;
    
            casadi::uout()<<"xxxxxxxxxxxxxxxxxx"<<std::endl;

            Rootfinder::init(opts);
            // 2. 默认参数
            max_iter_ = 1000;
            abstol_ = 1e-12;
            abstol_step_ = 1e-12;
            lb_ = -1.0;
            ub_ = 1.0;

            // 3. 读取用户选项
            for (auto &&op : opts)
            {
                if (op.first == "max_iter")
                    max_iter_ = static_cast<casadi_int>(op.second);
                else if (op.first == "abstol")
                    abstol_ = op.second;
                else if (op.first == "abstol_step")
                    abstol_step_ = op.second;
                else if (op.first == "lb")
                    lb_ = op.second;
                else if (op.first == "ub")
                    ub_ = op.second;
            }

            // 4. 合法性检查
            casadi_assert(n_ == 1,
                          "Bisection rootfinder only supports scalar equations (n=1), "
                          "but got n=" +
                              str(n_) + ".");

            casadi_assert(oracle_.n_in() > 0,
                          "Bisection: the supplied function f must have at least one input.");

            casadi_assert(lb_ < ub_,
                          "Bisection: lb (" + str(lb_) + ") must be strictly less than ub (" + str(ub_) + ").");

            // 5. 申请工作向量
            alloc_w(1, true); // x_mid
            alloc_w(1, true); // f_val
        }

        // ---------------------------------------------------------------
        // 内存管理
        // ---------------------------------------------------------------
        void *alloc_mem() const override { return new BisectionMemory(); }

        int init_mem(void *mem) const override
        {
            if (Rootfinder::init_mem(mem)) return 1;
            auto m = static_cast<BisectionMemory *>(mem);
            m->return_status = 0;
            m->iter = 0;
            m->f_mid = 0.0;
            m->bracket_width = ub_ - lb_;
            return 0;
        }

        void free_mem(void *mem) const override
        {
            delete static_cast<BisectionMemory *>(mem);
        }

        // ---------------------------------------------------------------
        // 工作向量切分
        // ---------------------------------------------------------------
        void set_work(void *mem, const double **&arg, double **&res,
                      casadi_int *&iw, double *&w) const override
        {
            Rootfinder::set_work(mem, arg, res, iw, w);
            w += 1; // x_mid
            w += 1; // f_val
        }

        // ---------------------------------------------------------------
        // 求解核心
        // ---------------------------------------------------------------
        int solve(void *mem) const override
        {
            auto m = static_cast<BisectionMemory *>(mem);

            double f_val = 0.0;

            // ---- 内部求值函数 ------------------------------------------
            auto eval_f = [&](double x) -> double
            {
                for (casadi_int i = 0; i < n_in_; ++i)
                    m->arg[i] = m->iarg[i];
                m->arg[iin_] = &x;
                for (casadi_int i = 0; i < n_out_; ++i)
                    m->res[i] = nullptr;
                m->res[iout_] = &f_val;
                if (oracle_(m->arg, m->res, m->iw, m->w, 0))
                {
                    f_val = std::numeric_limits<double>::quiet_NaN();
                }
                return f_val;
            };

            // ---- 初始化区间 --------------------------------------------
            double a = lb_, b = ub_;
            double fa = eval_f(a), fb = eval_f(b);

            // NaN检查
            if (std::isnan(fa) || std::isnan(fb))
            {
                casadi_warning("Bisection: f(lb) or f(ub) returned NaN.");
                return finish(m, a, 0.0, b - a, -1, false, SOLVER_RET_UNKNOWN);
            }

            // bracket合法性检查
            if (fa * fb > 0.0)
            {
                casadi_warning("Bisection: f(lb)*f(ub) > 0, bracket does not contain a root. "
                               "lb=" +
                               str(a) + " f(lb)=" + str(fa) +
                               " ub=" + str(b) + " f(ub)=" + str(fb));
                return finish(m, a, fa, b - a, -2, false, SOLVER_RET_UNKNOWN);
            }

            // 端点恰好是根
            if (fa == 0.0) return finish(m, a, 0.0, 0.0, 2, true, SOLVER_RET_SUCCESS);
            if (fb == 0.0) return finish(m, b, 0.0, 0.0, 2, true, SOLVER_RET_SUCCESS);

            // ---- 二分法主循环 ------------------------------------------
            double mid = a, f_mid_val = fa;

            for (m->iter = 0; m->iter < max_iter_; ++m->iter)
            {

                mid = 0.5 * (a + b);
                f_mid_val = eval_f(mid);

                // NaN保护
                if (std::isnan(f_mid_val))
                {
                    casadi_warning("Bisection: f(mid) is NaN at iter=" + str(m->iter));
                    return finish(m, mid, f_mid_val, b - a, -1, false, SOLVER_RET_UNKNOWN);
                }

                // 残差收敛
                if (std::fabs(f_mid_val) < abstol_)
                {
                    return finish(m, mid, f_mid_val, b - a, 2, true, SOLVER_RET_SUCCESS);
                }

                // 区间收敛
                if ((b - a) < abstol_step_)
                {
                    return finish(m, mid, f_mid_val, b - a, 1, true, SOLVER_RET_SUCCESS);
                }

                // 更新区间
                if (fa * f_mid_val < 0.0)
                {
                    b = mid;
                    fb = f_mid_val;
                }
                else
                {
                    a = mid;
                    fa = f_mid_val;
                }
            }

            // 达到最大迭代次数
            return finish(m, mid, f_mid_val, b - a, 0, false, SOLVER_RET_LIMITED);
        }

        // ---------------------------------------------------------------
        // 统计信息
        // ---------------------------------------------------------------
        Dict get_stats(void *mem) const override
        {
            Dict stats = Rootfinder::get_stats(mem);
            auto m = static_cast<BisectionMemory *>(mem);
            stats["return_status"] = status_str(m->return_status);
            stats["iter_count"] = m->iter;
            stats["f_mid"] = m->f_mid;
            stats["bracket_width"] = m->bracket_width;
            return stats;
        }

        // ---------------------------------------------------------------
        // 代码生成（暂不支持）
        // ---------------------------------------------------------------
        void codegen_declarations(CodeGenerator &g) const override {}
        void codegen_body(CodeGenerator &g) const override
        {
            casadi_error("Bisection::codegen_body is not implemented yet.");
        }

        // ---------------------------------------------------------------
        // 序列化 / 反序列化
        // ---------------------------------------------------------------
        void serialize_body(SerializingStream &s) const override
        {
            Rootfinder::serialize_body(s);
            s.version("Bisection", 1);
            s.pack("Bisection::max_iter", max_iter_);
            s.pack("Bisection::abstol", abstol_);
            s.pack("Bisection::abstol_step", abstol_step_);
            s.pack("Bisection::lb", lb_);
            s.pack("Bisection::ub", ub_);
        }

        static ProtoFunction *deserialize(DeserializingStream &s)
        {
            return new Bisection(s);
        }

    protected:
        explicit Bisection(DeserializingStream &s) : Rootfinder(s)
        {
            s.version("Bisection", 1);
            s.unpack("Bisection::max_iter", max_iter_);
            s.unpack("Bisection::abstol", abstol_);
            s.unpack("Bisection::abstol_step", abstol_step_);
            s.unpack("Bisection::lb", lb_);
            s.unpack("Bisection::ub", ub_);
        }

    private:
        // ---------------------------------------------------------------
        // 算法参数
        // ---------------------------------------------------------------
        casadi_int max_iter_;
        double abstol_;
        double abstol_step_;
        double lb_;
        double ub_;

        // ---------------------------------------------------------------
        // 辅助函数
        // ---------------------------------------------------------------
        int finish(BisectionMemory *m,
                   double x_sol, double f_sol, double width,
                   int status, bool success,
                   UnifiedReturnStatus urs) const
        {
            casadi_copy(&x_sol, 1, m->ires[iout_]);
            m->return_status = status;
            m->f_mid = f_sol;
            m->bracket_width = width;
            m->success = success;
            m->unified_return_status = urs;
            return 0;
        }

        static std::string status_str(int status)
        {
            switch (status)
            {
            case 0:
                return "max_iteration_reached";
            case 1:
                return "converged_bracket";
            case 2:
                return "converged_abstol";
            case -1:
                return "nan_encountered";
            case -2:
                return "invalid_bracket";
            default:
                return "unknown";
            }
        }
    };

    // =================================================================
    // 静态成员定义（.cpp中直接定义，无需inline）
    // =================================================================

    const std::string Bisection::meta_doc =
        "Bisection method rootfinder.\n"
        "Solves scalar equation f(x, p) = 0 using the bisection method.\n"
        "Requires options 'lb' and 'ub' such that f(lb) * f(ub) <= 0.\n"
        "Only supports scalar equations (n=1).\n";

    const Options Bisection::options_ = {{&Rootfinder::options_},
                                         {
                                             {"max_iter",
                                              {OT_INT,
                                               "Maximum number of bisection iterations (default: 1000)."}},
                                             {"abstol",
                                              {OT_DOUBLE,
                                               "Stopping tolerance on |f(mid)| (default: 1e-12)."}},
                                             {"abstol_step",
                                              {OT_DOUBLE,
                                               "Stopping tolerance on bracket width |b-a| (default: 1e-12)."}},
                                             {"lb",
                                              {OT_DOUBLE,
                                               "Initial lower bound of the bracket (default: -1.0)."}},
                                             {"ub",
                                              {OT_DOUBLE,
                                               "Initial upper bound of the bracket (default: 1.0)."}},
                                         }};

    // =================================================================
    // 插件注册函数
    // =================================================================

    extern "C" int casadi_register_rootfinder_bisection(Rootfinder::Plugin *plugin)
    {
        plugin->creator = Bisection::creator;
        plugin->name = "bisection";
        plugin->doc = Bisection::meta_doc.c_str();
        plugin->version = 3.7;
        plugin->options = &Bisection::options_;
        plugin->deserialize = &Bisection::deserialize;
        return 0;
    }

    extern "C" void casadi_load_rootfinder_bisection()
    {
        Rootfinder::registerPlugin(casadi_register_rootfinder_bisection);
    }

} // namespace casadi