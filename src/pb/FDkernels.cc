#include "FDkernels.h"
#include "Timer.h"

static Timer del2_2nd_tm("del2_2nd");
static Timer del2_4th_tm("del2_4th");
static Timer del2_4th_Mehr_tm("del2_4th_Mehr");
static Timer rhs_4th_Mehr1_tm("rhs_4th_Mehr1");

const double inv12 = 1. / 12.;

namespace pb
{

void printFDkernelTimers(std::ostream& os) { del2_2nd_tm.print(os); }

// evaluate 2nd order FD Laplacian on uniform mesh defined by Grid
// object
template <typename ScalarType>
void FDkernelDel2_2nd(const Grid& grid, ScalarType* v, ScalarType* b,
    const size_t nfunc, MemorySpace::Host)
{
    assert(grid.ghost_pt() > 0);
    MemorySpace::assert_is_host_ptr(b);
    MemorySpace::assert_is_host_ptr(v);

    del2_2nd_tm.start();

    double inv_h2[3] = {
        1. / (grid.hgrid(0) * grid.hgrid(0)),
        1. / (grid.hgrid(1) * grid.hgrid(1)),
        1. / (grid.hgrid(2) * grid.hgrid(2)),
    };

    double cc        = inv_h2[0];
    const double c1x = -cc;

    cc               = inv_h2[1];
    const double c1y = -cc;

    cc               = inv_h2[2];
    const double c1z = -cc;
    const double c0  = -2. * (c1x + c1y + c1z);

    const int dim0 = grid.dim(0);
    const int dim1 = grid.dim(1);
    const int dim2 = grid.dim(2);

    const size_t ngpts = grid.sizeg();

    int incx     = grid.inc(0);
    int incy     = grid.inc(1);
    const int ng = grid.ghost_pt();

#pragma omp parallel for collapse(4)
    for (size_t ifunc = 0; ifunc < nfunc; ifunc++)
    {
        for (int ix = 0; ix < dim0; ix++)
        {
            for (int iy = 0; iy < dim1; iy++)
            {
                for (int iz = 0; iz < dim2; iz++)
                {
                    int iiz = ifunc * ngpts
                              + ((ix + ng) * incx + (iy + ng) * incy) + iz + ng;

                    b[iiz] = c0 * v[iiz] + c1x * (v[iiz - incx] + v[iiz + incx])
                             + c1y * (v[iiz - incy] + v[iiz + incy])
                             + c1z * (v[iiz - 1] + v[iiz + 1]);
                }
            }
        }
    }

    del2_2nd_tm.stop();
}

template <typename ScalarType>
void FDkernelDel2_4th(const Grid& grid, ScalarType* A, ScalarType* B,
    const size_t nfunc, MemorySpace::Host)
{
    MemorySpace::assert_is_host_ptr(A);
    MemorySpace::assert_is_host_ptr(B);

    del2_4th_tm.start();

    double inv_h2[3] = {
        1. / (grid.hgrid(0) * grid.hgrid(0)),
        1. / (grid.hgrid(1) * grid.hgrid(1)),
        1. / (grid.hgrid(2) * grid.hgrid(2)),
    };

    const double cc0 = inv12 * inv_h2[0];
    const double c1x = -16. * cc0;
    const double c2x = 1. * cc0;

    const double cc1 = inv12 * inv_h2[1];
    const double c1y = -16. * cc1;
    const double c2y = 1. * cc1;

    const double cc2 = inv12 * inv_h2[2];
    const double c1z = -16. * cc2;
    const double c2z = 1. * cc2;

    const double c0 = -2. * (c1x + c2x + c1y + c2y + c1z + c2z);

    const int gpt = grid.ghost_pt();

    const int incx = grid.inc(0);
    const int incy = grid.inc(1);

    const int incx2 = 2 * grid.inc(0);
    const int incy2 = 2 * grid.inc(1);

    int iix = gpt * incx;

    const int dim0 = grid.dim(0);
    const int dim1 = grid.dim(1);
    const int dim2 = grid.dim(2);

    const size_t ng = grid.sizeg();

#pragma omp parallel for collapse(4)
    for (size_t ifunc = 0; ifunc < nfunc; ifunc++)
    {
        for (int ix = 0; ix < dim0; ix++)
        {
            for (int iy = 0; iy < dim1; iy++)
            {
                for (int iz = 0; iz < dim2; iz++)
                {
                    int iiz = ifunc * ng
                              + (iix + ix * incx + gpt * incy + iy * incy)
                              + gpt;

                    const ScalarType* __restrict__ v0   = A + iiz;
                    const ScalarType* __restrict__ vmx  = A + (iiz - incx);
                    const ScalarType* __restrict__ vpx  = A + (iiz + incx);
                    const ScalarType* __restrict__ vmx2 = A + (iiz - incx2);
                    const ScalarType* __restrict__ vpx2 = A + (iiz + incx2);
                    const ScalarType* __restrict__ vmy  = A + (iiz - incy);
                    const ScalarType* __restrict__ vpy  = A + (iiz + incy);
                    const ScalarType* __restrict__ vmy2 = A + (iiz - incy2);
                    const ScalarType* __restrict__ vpy2 = A + (iiz + incy2);

                    ScalarType* __restrict__ u = B + iiz;

                    u[iz] = c0 * (double)v0[iz]

                            + c1x * ((double)vmx[iz] + (double)vpx[iz])
                            + c1y * ((double)vmy[iz] + (double)vpy[iz])
                            + c1z * ((double)v0[iz - 1] + (double)v0[iz + 1])

                            + c2x * ((double)vmx2[iz] + (double)vpx2[iz])
                            + c2y * ((double)vmy2[iz] + (double)vpy2[iz])
                            + c2z * ((double)v0[iz - 2] + (double)v0[iz + 2]);
                }
            }
        }
    }

    del2_4th_tm.stop();
}

#ifdef HAVE_MAGMA
template <typename ScalarType>
void FDkernelDel2_4th(const Grid& grid, ScalarType* A, ScalarType* B,
    const size_t nfunc, MemorySpace::Device)
{
    MemorySpace::assert_is_dev_ptr(A);
    MemorySpace::assert_is_dev_ptr(B);

    assert(grid.ghost_pt() > 1);

    del2_4th_tm.start();

    double inv_h2[3] = {
        1. / (grid.hgrid(0) * grid.hgrid(0)),
        1. / (grid.hgrid(1) * grid.hgrid(1)),
        1. / (grid.hgrid(2) * grid.hgrid(2)),
    };

    const double cc0 = inv12 * inv_h2[0];
    const double c1x = -16. * cc0;
    const double c2x = 1. * cc0;

    const double cc1 = inv12 * inv_h2[1];
    const double c1y = -16. * cc1;
    const double c2y = 1. * cc1;

    const double cc2 = inv12 * inv_h2[2];
    const double c1z = -16. * cc2;
    const double c2z = 1. * cc2;

    const double c0 = -2. * (c1x + c2x + c1y + c2y + c1z + c2z);

    const int gpt = grid.ghost_pt();

    const int incx = grid.inc(0);
    const int incy = grid.inc(1);

    const int incx2 = 2 * grid.inc(0);
    const int incy2 = 2 * grid.inc(1);

    int iix = gpt * incx;

    const int dim0 = grid.dim(0);
    const int dim1 = grid.dim(1);
    const int dim2 = grid.dim(2);

    const size_t ng = grid.sizeg();

    MGMOL_PARALLEL_FOR_COLLAPSE(2, A, B) // 0.037 s 3000x32^3 0.0156 1000x32^3
    for (size_t ifunc = 0; ifunc < nfunc; ifunc++)
    {
        for (int ix = 0; ix < dim0; ix++)
        {
            for (int iy = 0; iy < dim1; iy++)
            {
                for (int iz = 0; iz < dim2; iz++)
                {
                    int iiz = ifunc * ng
                              + (iix + ix * incx + gpt * incy + iy * incy)
                              + gpt;

                    const ScalarType* __restrict__ v0   = A + iiz;
                    const ScalarType* __restrict__ vmx  = A + (iiz - incx);
                    const ScalarType* __restrict__ vpx  = A + (iiz + incx);
                    const ScalarType* __restrict__ vmx2 = A + (iiz - incx2);
                    const ScalarType* __restrict__ vpx2 = A + (iiz + incx2);
                    const ScalarType* __restrict__ vmy  = A + (iiz - incy);
                    const ScalarType* __restrict__ vpy  = A + (iiz + incy);
                    const ScalarType* __restrict__ vmy2 = A + (iiz - incy2);
                    const ScalarType* __restrict__ vpy2 = A + (iiz + incy2);

                    ScalarType* __restrict__ u = B + iiz;

                    u[iz] = c0 * (double)v0[iz]

                            + c1x * ((double)vmx[iz] + (double)vpx[iz])
                            + c1y * ((double)vmy[iz] + (double)vpy[iz])
                            + c1z * ((double)v0[iz - 1] + (double)v0[iz + 1])

                            + c2x * ((double)vmx2[iz] + (double)vpx2[iz])
                            + c2y * ((double)vmy2[iz] + (double)vpy2[iz])
                            + c2z * ((double)v0[iz - 2] + (double)v0[iz + 2]);
                }
            }
        }
    }

    del2_4th_tm.stop();
}
#endif

template <typename ScalarType>
void FDkernelDel2_6th(const Grid& grid, ScalarType* v, ScalarType* u,
    const size_t nfunc, MemorySpace::Host)
{
    assert(grid.ghost_pt() > 2);

    double inv_h2[3] = { 1. / (grid.hgrid(0) * grid.hgrid(0)),
        1. / (grid.hgrid(1) * grid.hgrid(1)),
        1. / (grid.hgrid(2) * grid.hgrid(2)) };

    double cc        = (1. / 180.) * inv_h2[0];
    const double c1x = -270. * cc;
    const double c2x = 27. * cc;
    const double c3x = -2. * cc;

    cc               = (1. / 180.) * inv_h2[1];
    const double c1y = -270. * cc;
    const double c2y = 27. * cc;
    const double c3y = -2. * cc;

    cc               = (1. / 180.) * inv_h2[2];
    const double c1z = -270. * cc;
    const double c2z = 27. * cc;
    const double c3z = -2. * cc;

    const double c0
        = -2. * (c1x + c2x + c3x + c1y + c2y + c3y + c1z + c2z + c3z);

    const int dim0 = grid.dim(0);
    const int dim1 = grid.dim(1);
    const int dim2 = grid.dim(2);

    const size_t ngpts = grid.sizeg();

    const int incx = grid.inc(0);
    const int incy = grid.inc(1);

    const int incx2 = 2 * grid.inc(0);
    const int incy2 = 2 * grid.inc(1);
    const int incx3 = 3 * grid.inc(0);
    const int incy3 = 3 * grid.inc(1);

    const int ng = grid.ghost_pt();

    for (size_t ifunc = 0; ifunc < nfunc; ifunc++)
    {
        for (int ix = 0; ix < dim0; ix++)
        {
            for (int iy = 0; iy < dim1; iy++)
            {
                for (int iz = 0; iz < dim2; iz++)
                {
                    int i = ifunc * ngpts
                            + ((ix + ng) * incx + (iy + ng) * incy) + iz + ng;

                    u[i] = (ScalarType)(c0 * (double)v[i]
                                        + c1x
                                              * ((double)v[i - incx]
                                                  + (double)v[i + incx])
                                        + c1y
                                              * ((double)v[i - incy]
                                                  + (double)v[i + incy])
                                        + c1z
                                              * ((double)v[i - 1]
                                                  + (double)v[i + 1])
                                        + c2x
                                              * ((double)v[i - incx2]
                                                  + (double)v[i + incx2])
                                        + c2y
                                              * ((double)v[i - incy2]
                                                  + (double)v[i + incy2])
                                        + c2z
                                              * ((double)v[i - 2]
                                                  + (double)v[i + 2])

                                        + c3x
                                              * ((double)v[i - incx3]
                                                  + (double)v[i + incx3])
                                        + c3y
                                              * ((double)v[i - incy3]
                                                  + (double)v[i + incy3])
                                        + c3z
                                              * ((double)v[i - 3]
                                                  + (double)v[i + 3]));
                }
            }
        }
    }
}

template <typename ScalarType>
void FDkernelDel2_8th(const Grid& grid, ScalarType* v, ScalarType* u,
    const size_t nfunc, MemorySpace::Host)
{
    double inv_h2[3] = { 1. / (grid.hgrid(0) * grid.hgrid(0)),
        1. / (grid.hgrid(1) * grid.hgrid(1)),
        1. / (grid.hgrid(2) * grid.hgrid(2)) };

    double cc        = (1. / 5040.) * inv_h2[0];
    const double c1x = -8064. * cc;
    const double c2x = 1008. * cc;
    const double c3x = -128. * cc;
    const double c4x = 9. * cc;

    cc               = (1. / 5040.) * inv_h2[1];
    const double c1y = -8064. * cc;
    const double c2y = 1008. * cc;
    const double c3y = -128. * cc;
    const double c4y = 9. * cc;

    cc               = (1. / 5040.) * inv_h2[2];
    const double c1z = -8064. * cc;
    const double c2z = 1008. * cc;
    const double c3z = -128. * cc;
    const double c4z = 9. * cc;

    const double c0 = -2.
                      * (c1x + c2x + c3x + c4x + c1y + c2y + c3y + c4y + c1z
                          + c2z + c3z + c4z);

    const int dim0 = grid.dim(0);
    const int dim1 = grid.dim(1);
    const int dim2 = grid.dim(2);

    const size_t ngpts = grid.sizeg();

    const int incx  = grid.inc(0);
    const int incy  = grid.inc(1);
    const int incx2 = 2 * incx;
    const int incy2 = 2 * incy;
    const int incx3 = 3 * incx;
    const int incy3 = 3 * incy;
    const int incx4 = 4 * incx;
    const int incy4 = 4 * incy;

    const int ng = grid.ghost_pt();

    for (size_t ifunc = 0; ifunc < nfunc; ifunc++)
    {
        for (int ix = 0; ix < dim0; ix++)
        {
            for (int iy = 0; iy < dim1; iy++)
            {
                for (int iz = 0; iz < dim2; iz++)
                {
                    const int i = ifunc * ngpts
                                  + ((ix + ng) * incx + (iy + ng) * incy) + iz
                                  + ng;

                    u[i] = (ScalarType)(c0 * (double)v[i]

                                        + c1x
                                              * ((double)v[i - incx]
                                                  + (double)v[i + incx])
                                        + c1y
                                              * ((double)v[i - incy]
                                                  + (double)v[i + incy])
                                        + c1z
                                              * ((double)v[i - 1]
                                                  + (double)v[i + 1])

                                        + c2x
                                              * ((double)v[i - incx2]
                                                  + (double)v[i + incx2])
                                        + c2y
                                              * ((double)v[i - incy2]
                                                  + (double)v[i + incy2])
                                        + c2z
                                              * ((double)v[i - 2]
                                                  + (double)v[i + 2])

                                        + c3x
                                              * ((double)v[i - incx3]
                                                  + (double)v[i + incx3])
                                        + c3y
                                              * ((double)v[i - incy3]
                                                  + (double)v[i + incy3])
                                        + c3z
                                              * ((double)v[i - 3]
                                                  + (double)v[i + 3])

                                        + c4x
                                              * ((double)v[i - incx4]
                                                  + (double)v[i + incx4])
                                        + c4y
                                              * ((double)v[i - incy4]
                                                  + (double)v[i + incy4])
                                        + c4z
                                              * ((double)v[i - 4]
                                                  + (double)v[i + 4]));
                }
            }
        }
    }
}

template <typename ScalarType>
void FDkernelDel2_4th_Mehr(const Grid& grid, ScalarType* v, ScalarType* u,
    const size_t nfunc, MemorySpace::Host)
{
    del2_4th_Mehr_tm.start();

    double inv_h2[3] = { 1. / (grid.hgrid(0) * grid.hgrid(0)),
        1. / (grid.hgrid(1) * grid.hgrid(1)),
        1. / (grid.hgrid(2) * grid.hgrid(2)) };

    const double c0mehr4  = 16. * inv12 * (inv_h2[0] + inv_h2[1] + inv_h2[2]);
    const double cxmehr4  = -10. * inv12 * inv_h2[0] + 0.125 * c0mehr4;
    const double cymehr4  = -10. * inv12 * inv_h2[1] + 0.125 * c0mehr4;
    const double czmehr4  = -10. * inv12 * inv_h2[2] + 0.125 * c0mehr4;
    const double cxymehr4 = -inv12 * (inv_h2[0] + inv_h2[1]);
    const double cyzmehr4 = -inv12 * (inv_h2[2] + inv_h2[1]);
    const double cxzmehr4 = -inv12 * (inv_h2[0] + inv_h2[2]);

    const int shift = grid.ghost_pt();
    const int dim0  = grid.dim(0);
    const int dim1  = grid.dim(1);
    const int dim2  = grid.dim(2);
    const int ngpts = grid.sizeg();
    const int incx  = grid.inc(0);
    const int incy  = grid.inc(1);

    const int iix0 = shift * incx;

    for (size_t ifunc = 0; ifunc < nfunc; ifunc++)
    {
        for (int ix = 0; ix < dim0; ix++)
        {
            int iix = iix0 + ix * incx + ifunc * ngpts;
            int iiy = iix + shift * incy;

            for (int iy = 0; iy < dim1; iy++)
            {
                const int iiz = iiy + shift;

                ScalarType* const u0          = u + iiz;
                const ScalarType* const v0    = v + iiz;
                const ScalarType* const vmx   = v0 - incx;
                const ScalarType* const vpx   = v0 + incx;
                const ScalarType* const vmxmy = vmx - incy;
                const ScalarType* const vpxmy = vpx - incy;
                const ScalarType* const vmy   = v0 - incy;
                const ScalarType* const vpy   = v0 + incy;
                const ScalarType* const vmxpy = vmx + incy;
                const ScalarType* const vpxpy = vpx + incy;

                for (int iz = 0; iz < dim2; iz++)
                {
                    u0[iz]
                        = (ScalarType)(c0mehr4 * (double)v0[iz]
                                       + czmehr4
                                             * (double)(v0[iz - 1] + v0[iz + 1])
                                       + cymehr4 * (double)(vmy[iz] + vpy[iz])
                                       + cxmehr4 * (double)(vmx[iz] + vpx[iz])
                                       + cxzmehr4
                                             * (double)(vmx[iz - 1]
                                                        + vmx[iz + 1]
                                                        + vpx[iz - 1]
                                                        + vpx[iz + 1])
                                       + cyzmehr4
                                             * (double)(vmy[iz - 1]
                                                        + vmy[iz + 1]
                                                        + vpy[iz - 1]
                                                        + vpy[iz + 1])
                                       + cxymehr4
                                             * (double)(vmxmy[iz] + vpxmy[iz]
                                                        + vmxpy[iz]
                                                        + vpxpy[iz]));
                }

                iiy += incy;
            }
        }
    }
    del2_4th_Mehr_tm.stop();
}

template <typename ScalarType>
void FDkernelRHS_4th_Mehr1(const Grid& grid, ScalarType* v, ScalarType* rhs,
    const short rhs_ghosts, const size_t nfunc, MemorySpace::Host)
{
    rhs_4th_Mehr1_tm.start();

    const double c0 = 0.5;
    const double c1 = inv12;

    const int shift = grid.ghost_pt();
    int incx        = grid.inc(0);
    int incy        = grid.inc(1);
    const int iix0  = shift * incx;

    const int dim0 = grid.dim(0);
    const int dim1 = grid.dim(1);
    const int dim2 = grid.dim(2);

    const size_t ngpts = grid.sizeg();

    const int incx_rhs  = (dim2 + 2 * rhs_ghosts) * (dim1 + 2 * rhs_ghosts);
    const int incy_rhs  = (dim2 + 2 * rhs_ghosts);
    const int ngpts_rhs = (dim2 + 2 * rhs_ghosts) * (dim1 + 2 * rhs_ghosts)
                          * (dim0 + 2 * rhs_ghosts);

    const int offset_rhs
        = rhs_ghosts * incx_rhs + rhs_ghosts * incy_rhs + rhs_ghosts;

    for (size_t ifunc = 0; ifunc < nfunc; ifunc++)
    {
        for (int ix = 0; ix < dim0; ix++)
        {
            int iix = iix0 + ix * incx + ifunc * ngpts;
            int iiy = iix + shift * incy;

            for (int iy = 0; iy < dim1; iy++)
            {
                int iiz = iiy + shift;

                ScalarType* const u0 = rhs + ix * incx_rhs + iy * incy_rhs
                                       + offset_rhs + ifunc * ngpts_rhs;
                const ScalarType* const v0  = v + iiz;
                const ScalarType* const vmx = v0 - incx;
                const ScalarType* const vpx = v0 + incx;
                const ScalarType* const vmy = v0 - incy;
                const ScalarType* const vpy = v0 + incy;

                for (int iz = 0; iz < dim2; iz++)
                {
                    u0[iz] = (ScalarType)(c0 * (double)v0[iz]
                                          + c1
                                                * (double)(vmx[iz] + vpx[iz]
                                                           + vmy[iz] + vpy[iz]
                                                           + v0[iz - 1]
                                                           + v0[iz + 1]));
                }

                iiy += incy;
            }
        }
    }

    rhs_4th_Mehr1_tm.stop();
}

template void FDkernelDel2_2nd<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Host);
template void FDkernelDel2_2nd<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Host);

template void FDkernelDel2_4th<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Host);
template void FDkernelDel2_4th<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Host);

template void FDkernelDel2_4th_Mehr<double>(const Grid& grid, double* v,
    double* b, const size_t nfunc, MemorySpace::Host);
template void FDkernelDel2_4th_Mehr<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Host);

template void FDkernelDel2_6th<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Host);
template void FDkernelDel2_6th<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Host);

template void FDkernelDel2_8th<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Host);
template void FDkernelDel2_8th<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Host);

template void FDkernelRHS_4th_Mehr1<double>(const Grid& grid, double* v,
    double* b, const short nghosts, const size_t nfunc, MemorySpace::Host);
template void FDkernelRHS_4th_Mehr1<float>(const Grid& grid, float* v, float* b,
    const short nghosts, const size_t nfunc, MemorySpace::Host);

#ifdef HAVE_MAGMA
template void FDkernelDel2_2nd<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Device);
template void FDkernelDel2_2nd<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Device);

template void FDkernelDel2_4th<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Device);
template void FDkernelDel2_4th<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Device);

template void FDkernelDel2_6th<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Device);
template void FDkernelDel2_6th<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Device);

template void FDkernelDel2_8th<double>(const Grid& grid, double* v, double* b,
    const size_t nfunc, MemorySpace::Device);
template void FDkernelDel2_8th<float>(const Grid& grid, float* v, float* b,
    const size_t nfunc, MemorySpace::Device);
#endif
} // namespace pb
