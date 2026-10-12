#include "GridFunc.h"
#include "Laph2.h"
#include "Laph4.h"
#include "Laph6.h"
#include "MGmol_MPI.h"
#include "PCGSolver.h"
#include "PEenv.h"

#include "catch.hpp"

class Function
{
private:
    static double lattice_[3];
    static double coeff_[3];

    static double f_(const double x, const double y, const double z)
    {
        const double ax = coeff_[0] * 2. * M_PI / lattice_[0];
        const double ay = coeff_[1] * 2. * M_PI / lattice_[1];
        const double az = coeff_[2] * 2. * M_PI / lattice_[2];

        return sin(ax * x) + sin(ay * y) + sin(az * z);
    }

    static double del2f_(const double x, const double y, const double z)
    {
        const double ax = coeff_[0] * 2. * M_PI / lattice_[0];
        const double ay = coeff_[1] * 2. * M_PI / lattice_[1];
        const double az = coeff_[2] * 2. * M_PI / lattice_[2];

        return ax * ax * sin(ax * x) + ay * ay * sin(ay * y)
               + az * az * sin(az * z);
    }

    void set(pb::GridFunc<double>& gfu, double (*f)(double, double, double))
    {
        const pb::Grid& grid(gfu.grid());
        double* u1 = gfu.uu();

        const short nghosts = grid.ghost_pt();
        const int endx      = nghosts + grid.dim(0);
        const int endy      = nghosts + grid.dim(1);
        const int endz      = nghosts + grid.dim(2);
        double h[3]         = { grid.hgrid(0), grid.hgrid(1), grid.hgrid(2) };
        for (int ix = nghosts; ix < endx; ix++)
        {
            int iix  = ix * grid.inc(0);
            double x = grid.start(0) + ix * h[0];

            for (int iy = nghosts; iy < endy; iy++)
            {
                int iiy  = iy * grid.inc(1) + iix;
                double y = grid.start(1) + iy * h[1];

                for (int iz = nghosts; iz < endz; iz++)
                {
                    double z = grid.start(2) + iz * h[2];

                    u1[iiy + iz] = (*f)(x, y, z);
                }
            }
        }
        gfu.set_updated_boundaries(false);
    }

public:
    Function(const double lattice[3], const double coeff[3])
    {
        lattice_[0] = lattice[0];
        lattice_[1] = lattice[1];
        lattice_[2] = lattice[2];

        coeff_[0] = coeff[0];
        coeff_[1] = coeff[1];
        coeff_[2] = coeff[2];
    }

    void set(pb::GridFunc<double>& gfu) { set(gfu, this->f_); }

    void setLap(pb::GridFunc<double>& gfu) { set(gfu, this->del2f_); }
};

double Function::lattice_[3] = { -1, -1, -1 };
double Function::coeff_[3]   = { 0, 0, 0 };

// run test problems
template <class LapOperator>
void run_test()
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    const int nghosts = LapOperator::minNumberGhosts();
    const int order   = LapOperator::order();

    // specify computational domain
    const double origin[3]  = { 0., 0., 0. };
    const double ll         = 2.;
    const double lattice[3] = { ll, ll, ll };

    // mesh parameters
    const unsigned ngpts[3] = { 32, 24, 20 };

    // domain decomposition
    pb::PEenv mype_env(MPI_COMM_WORLD, ngpts[0], ngpts[1], ngpts[2]);

    // generate computaional mesh
    pb::Grid grid(origin, lattice, ngpts, mype_env, nghosts, 0);

    LapOperator lap(grid);

    // periodic GridFunc
    const short bc[3] = { 1, 1, 1 };
    pb::GridFunc<double> gfu(grid, bc[0], bc[1], bc[2]);
    pb::GridFunc<double> gff(grid, bc[0], bc[1], bc[2]);
    pb::GridFunc<double> gfv(grid, bc[0], bc[1], bc[2]);

    double coeff[3] = { 1., 3., 2. };
    Function f(lattice, coeff);

    // order factor
    double factor = pow(2, order);

    // initialize gfu
    f.set(gfu);

    // fill ghost values
    gfu.trade_boundaries();

    const short precond_lap_type = 1; // 2nd order
    PCGSolver<LapOperator, double, float> pcg(
        lap, precond_lap_type, bc[0], bc[1], bc[2]);
    const double tol    = 1.e-11;
    int maxits          = 30;
    const int max_level = 10;
    pcg.setup(1, 1, maxits, tol, max_level);

    // check PCG convergence
    {
        // apply FD (-Laplacian) operator to gfu, result in gff
        lap.apply(gfu, gff);

        const double nrhs = gff.norm2();

        bool converged = pcg.solve(gfv, gff);
        CHECK(converged);

        if (mmpi.instancePE0())
            std::cout << "Norm of RHS = " << nrhs << std::endl;

        pb::GridFunc<double> gfr(grid, bc[0], bc[1], bc[2]);
        lap.apply(gfv, gfr);
        gfr -= gff;
        const double nr = gfr.norm2();
        if (mmpi.instancePE0())
            std::cout << "Norm of final residual = " << nr << std::endl;
        CHECK(nr < tol * nrhs);

        // compare solution with data used to generate r.h.s.
        gfv -= gfu;

        double ne = gfv.norm2();

        if (mmpi.instancePE0())
            std::cout << "Norm of error = " << ne << std::endl;
        CHECK(ne < tol);
    }

    // check discretization error
    {
        // initialize rhs
        f.setLap(gff);

        // Poisson solve
        bool converged = pcg.solve(gfv, gff);
        CHECK(converged);

        // compare solution with exact solution
        gfv -= gfu;

        double ne = gfv.norm2();

        if (mmpi.instancePE0())
            std::cout << "Norm of discretization error = " << ne << std::endl;
        CHECK(ne < 0.6 / factor);

        // generate finer computaional mesh
        const unsigned ngpts_fine[3]
            = { 2 * ngpts[0], 2 * ngpts[1], 2 * ngpts[2] };
        pb::PEenv mype_env_fine(
            MPI_COMM_WORLD, ngpts_fine[0], ngpts_fine[1], ngpts_fine[2]);
        pb::Grid grid_fine(
            origin, lattice, ngpts_fine, mype_env_fine, nghosts, 0);

        LapOperator lap_fine(grid_fine);

        // periodic GridFunc
        pb::GridFunc<double> gfu_fine(grid_fine, bc[0], bc[1], bc[2]);
        pb::GridFunc<double> gfv_fine(grid_fine, bc[0], bc[1], bc[2]);
        pb::GridFunc<double> gff_fine(grid_fine, bc[0], bc[1], bc[2]);

        maxits = 50;
        PCGSolver<LapOperator, double, float> pcg_fine(
            lap_fine, precond_lap_type, bc[0], bc[1], bc[2]);
        pcg_fine.setup(1, 1, maxits, tol, max_level);

        // initialize exact solution and rhs on fine mesh
        f.set(gfu_fine);
        f.setLap(gff_fine);

        // solve fine mesh problem
        converged = pcg_fine.solve(gfv_fine, gff_fine);
        CHECK(converged);

        // compare solution with exact solution
        gfv_fine -= gfu_fine;

        double ne_fine = gfv_fine.norm2();
        if (mmpi.instancePE0())
        {
            std::cout << "Norm of discretization error = " << ne_fine
                      << std::endl;
            std::cout << "Expected : " << ne / factor << std::endl;
        }
        CHECK(ne_fine < ne);
        // check within 10% of expected convergence O(h^4)
        CHECK(std::abs(ne_fine - ne / factor) < ne_fine * 0.1);
    }
}

// run test problems
TEST_CASE("Poisson", "[poisson]")
{
    MGmol_MPI::setup(MPI_COMM_WORLD, std::cout);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    if (mmpi.instancePE0())
        std::cout << std::endl << "Run test for Laph2..." << std::endl;
    run_test<pb::Laph2<double>>();

    if (mmpi.instancePE0())
        std::cout << std::endl << "Run test for Laph4..." << std::endl;
    run_test<pb::Laph4<double>>();

    if (mmpi.instancePE0())
        std::cout << std::endl << "Run test for Laph6..." << std::endl;
    run_test<pb::Laph6<double>>();
}
