// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "KBprojectorSparse.h"
#include "Mesh.h"
#include "Species.h"

#include "MGmol_abort.h"
#include <cstring>

const double rthreshold = 1.e-5;

std::vector<std::vector<KBPROJDTYPE>> KBprojectorSparse::work_nlindex_;
std::vector<std::vector<KBPROJDTYPE>> KBprojectorSparse::work_proj_;

KBprojectorSparse::KBprojectorSparse(const Species& sp) : KBprojector(sp)
{
    range_kbproj_ = species_.dim_nl();

    if (work_nlindex_.size() == 0) work_nlindex_.resize(omp_get_max_threads());
    if (work_proj_.size() == 0) work_proj_.resize(omp_get_max_threads());
    // cout<<"constructor: work_nlindex_.size()="<<work_nlindex_.size()<<endl;

    is_in_domain_ = nullptr;

    for (short i = 0; i < 3; i++)
    {
        kb_proj_start_index_[i] = 10000;
    }

    assert(range_kbproj_ >= 0);
    assert(range_kbproj_ < 256);
}

// copy constructor (without initialization of projectors)
KBprojectorSparse::KBprojectorSparse(const KBprojectorSparse& kb)
    : KBprojector(kb)
{
    range_kbproj_ = kb.range_kbproj_;

    is_in_domain_ = nullptr;

    for (short i = 0; i < 3; i++)
    {
        kb_proj_start_index_[i] = -1;
    }

    assert(range_kbproj_ >= 0);
    assert(range_kbproj_ < 256);
}

void KBprojectorSparse::setup(const double center[3])
{
    assert(subdivx_ > 0);

    // if(onpe0)cout<<"KBprojectorSparse::setup()..."<<endl;

    KBprojector::setup(center);

    for (short i = 0; i < 3; i++)
    {
        kb_proj_start_[i] = center_[i];
    }

    nlindex_.resize(subdivx_);
    size_nl_.assign(subdivx_, 0);

    if (maxl_ > 0)
    {
        setPtrProjectors();

        setIndexesAndProjectors();
    }

    Mesh* mymesh     = Mesh::instance();
    const int numloc = mymesh->locNumpt();
    for (short it = 0; it < omp_get_max_threads(); it++)
        work_proj_[it].resize(numloc);
}

void KBprojectorSparse::setNLindex(
    const short iloc, const int size_nl, const std::vector<int>& pvec)
{
    assert(size_nl > 0);
    assert((int)nlindex_.size() > iloc);
    assert((int)size_nl_.size() > iloc);
    assert(work_nlindex_.size()
           == static_cast<unsigned int>(omp_get_max_threads()));

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    const int numpt        = mygrid.size();

    nlindex_[iloc].resize(size_nl);
    size_nl_[iloc] = size_nl;
    std::vector<int>& nli(nlindex_[iloc]);
    const int lnumpt = numpt / subdivx_;

    for (int i = 0; i < size_nl; i++)
    {
        // assert(i < lnumpt);
        if ((pvec[i] < iloc * lnumpt) || (pvec[i] >= (iloc + 1) * lnumpt))
        {
            (*MPIdata::sout) << " iloc=" << iloc << ", i=" << i
                             << ", pvec=" << pvec[i] << std::endl;
            mgmol_abort();
        }

        assert(pvec[i] >= iloc * lnumpt);
        assert(pvec[i] < (iloc + 1) * lnumpt);

        nli[i] = pvec[i];
    }

    for (short it = 0; it < omp_get_max_threads(); it++)
        if ((int)work_nlindex_[it].size() < size_nl)
            work_nlindex_[it].resize(size_nl);
}

void KBprojectorSparse::registerPsi(const short iloc, const ORBDTYPE* const psi)
{
    assert(iloc < subdivx_);
    assert(iloc < static_cast<int>(nlindex_.size()));

    const short thread = omp_get_thread_num();
    const int sizenl   = size_nl_[iloc];
    const std::vector<int>& rnlindex(nlindex_[iloc]);
    assert(work_nlindex_.size()
           == static_cast<unsigned int>(omp_get_max_threads()));
    assert(static_cast<unsigned int>(thread) < work_nlindex_.size());
    std::vector<KBPROJDTYPE>& work(work_nlindex_[thread]);
    for (int i = 0; i < sizenl; i++)
    {
        const int j = rnlindex[i];
        // convert to KBPROJDTYPE if necessary to enable optimized dot products
        work[i] = (KBPROJDTYPE)psi[j];
    }
}

bool KBprojectorSparse::overlapPE() const
{
    assert(!size_nl_.empty());

    int n = 0;
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        n += size_nl_[iloc];
    }
    return (n > 0);
}

void KBprojectorSparse::allocateProjectors(const short iloc, const int icount)
{
    const int nprojs = nProjectors();

    // projector may not overlap with all local patches, so add empty
    // vector in slots associated with smaller iloc if they have not been
    // filled
    while (static_cast<int>(projectors_storage_.size()) < iloc)
        projectors_storage_.push_back(std::vector<KBPROJDTYPE>(0));
    projectors_storage_.push_back(std::vector<KBPROJDTYPE>(nprojs * icount));
    assert(static_cast<int>(projectors_storage_.size()) == (iloc + 1));

    KBPROJDTYPE* pstorage(&projectors_storage_[iloc][0]);

    // Loop over radial projectors
    for (short l = 0; l <= maxl_; l++)
    {
        // Skip the local potential
        if (l != llocal_)
        {
            for (short p = 0; p < multiplicity_[l]; ++p)
            {
                for (short m = 0; m < 2 * l + 1; m++)
                {
                    ptr_projector_[iloc][l][p][m] = pstorage;
                    pstorage += icount;
                }
            }
        }
    }
}

void KBprojectorSparse::setProjectors(const short iloc, const int icount)
{
    assert(is_in_domain_ != nullptr);
    assert(is_in_domain_[iloc] != nullptr);

    allocateProjectors(iloc, icount);

    // Loop over radial projectors
    for (short l = 0; l <= maxl_; l++)
    {
        // Skip the local potential
        if (l != llocal_)
        {
            switch (l)
            {
                case 0:
                    setSProjector(iloc, icount);
                    break;

                case 1:
                    setPProjector(iloc, icount);
                    break;

                case 2:
                    setDProjector(iloc, icount);
                    break;

                case 3:
                    setFProjector(iloc, icount);
                    break;

                default:
                    std::cerr << "KBprojectorSparse::setProjectors(): "
                              << "Angular momentum state not implemented!!!"
                              << std::endl;
                    mgmol_abort();
            }
        }
    }
}

/*
void KBprojectorSparse::setSupersamplingSProjector(const short iloc, const int
icount) { assert((int)ptr_projector_.size() > iloc);
    assert(ptr_projector_[iloc].size() > 0);
    assert(ptr_projector_[iloc][0].size() > 0);
        std::array<double,3> botMeshCorner={0,0,0};
        std::array<double,3> topMeshCorner={0,0,0};
        std::array<double,3> subDomainBotMeshCorner={start0,start1,start2};
        std::array<double,3> subDomainTopMeshCorner={start0 + dim0*h0,
                                                     start1 + dim1*h1,
                                                     start2 + dim2*h2};
        std::array<int,3> SSLRad={0,0,0};

    for (short p = 0; p < multiplicity_[0]; ++p)
    {
        assert(ptr_projector_[iloc][0][p].size() > 0);

        int jcount = 0;
        const RadialInter& nlproj(species_.getRadialKBP(0, p));
        auto lambda_lpot = [&nlproj] (double radius) {return
nlproj.cubint(radius); }

        KBPROJDTYPE* rtptr = ptr_projector_[iloc][0][p][0];
        // Construct subdomain containing molecule
        std::array<double,3> atomicCenter={center_[0], center_[1], center_[2]};
        SSLRad[0] = std::ceil(lrad/h0)+1; // +1 just to be safe and make sure
subdomain gets everything SSLRad[1] = std::ceil(lrad/h1)+1; SSLRad[2] =
std::ceil(lrad/h2)+1; botMeshCorner[0] =
std::max(std::round((atomicCenter[0]-start0)/h0)*h0 + start0 - SSLRad[0]*h0,
                        subDomainBotMeshCorner[0]);
        botMeshCorner[1] = std::max(std::round((atomicCenter[1]-start1)/h1)*h1 +
start1 - SSLRad[1]*h1, subDomainBotMeshCorner[1]); botMeshCorner[2] =
std::max(std::round((atomicCenter[2]-start2)/h2)*h2 + start2 - SSLRad[2]*h2,
                        subDomainBotMeshCorner[2]);
        topMeshCorner[0] = std::min(botMeshCorner[0] + 2*h0*SSLRad[0],
subDomainTopMeshCorner[0]); topMeshCorner[1] = std::min(botMeshCorner[1] +
2*h1*SSLRad[1], subDomainTopMeshCorner[1]); topMeshCorner[2] =
std::min(botMeshCorner[2] + 2*h2*SSLRad[2], subDomainTopMeshCorner[2]); const
bool harmonics=true;

        superSampling<0> current(atomicCenter, botMeshCorner, topMeshCorner,
harmonics, lambda_lpot); int xlimits =
std::round((topMeshCorner[0]-botMeshCorner[0])/h0); int ylimits =
std::round((topMeshCorner[1]-botMeshCorner[1])/h1); int zlimits =
std::round((topMeshCorner[2]-botMeshCorner[2])/h2); int xoffset =
std::round((botMeshCorner[0]-start0)/h0); int yoffset =
std::round((botMeshCorner[1]-start1)/h1); int zoffset =
std::round((botMeshCorner[2]-start2)/h2); int offset=0;

        for(int ix = xoffset; ix <= xoffset + xlimits; ix++) {
            int istart = ix*dim1*dim2;
            for(int iy = yoffset; iy <= yoffset + ylimits; iy++) {
                int jstart = istart + iy*dim2;
                for(int iz = zoffset; iz <= zoffset + zlimits; iz++) {
                    v_nuc_[jstart + iz]    += current.values_[0][offset];
                    offset++;
                }
            }
        }
*/

// Generates an s-projector
void KBprojectorSparse::setSProjector(const short iloc, const int icount)
{
    assert((int)ptr_projector_.size() > iloc);
    assert(ptr_projector_[iloc].size() > 0);
    assert(ptr_projector_[iloc][0].size() > 0);

    const bool* const is_in_domain = is_in_domain_[iloc];

    const double factor = sqrt(1. / (4. * M_PI));

    for (short p = 0; p < multiplicity_[0]; ++p)
    {
        assert(ptr_projector_[iloc][0][p].size() > 0);

        int jcount = 0;
        const RadialInter& nlproj(species_.getRadialKBP(0, p));

        KBPROJDTYPE* rtptr = ptr_projector_[iloc][0][p][0];

        int idx = 0;

        double x = kb_proj_start_[0] - center_[0];

        for (int ix = 0; ix < range_kbproj_; ix++)
        {
            double y = kb_proj_start_[1] - center_[1];

            for (int iy = 0; iy < range_kbproj_; iy++)
            {
                double z = kb_proj_start_[2] - center_[2];

                for (int iz = 0; iz < range_kbproj_; iz++)
                {
                    if (is_in_domain[idx])
                    {
                        double r = sqrt(x * x + y * y + z * z);

                        assert(jcount < icount);
                        rtptr[jcount]
                            = (KBPROJDTYPE)(nlproj.cubint(r) * factor);

#if CHECK_NORM
                        assert(p < norm2_[0].size());
                        double dval = (double)rtptr[jcount];
                        norm2_[0][p] += dval * dval;
#endif
                        jcount++;
                    }

                    idx++;
                    z += h_[2];
                }

                y += h_[1];
            }

            x += h_[0];
        }

        if (jcount != icount)
        {
            std::cerr << "KBprojectorSparse::setSProjector(): " << std::endl;
            mgmol_abort("Problem ith non-local projectors generation");
        }
    }
}

// Generates p-projectors
void KBprojectorSparse::setPProjector(const short iloc, const int icount)
{
    assert((int)ptr_projector_.size() > iloc);
    assert((int)ptr_projector_[iloc].size() > 1);
    assert((int)ptr_projector_[iloc][1].size() == multiplicity_[1]);

    const bool* const is_in_domain = is_in_domain_[iloc];

    const double factor = sqrt(3. / (4. * M_PI));

    for (short p = 0; p < multiplicity_[1]; p++)
    {
        assert(ptr_projector_[iloc][1][p].size() > 0);

        int jcount         = 0;
        KBPROJDTYPE* projx = ptr_projector_[iloc][1][p][0];
        KBPROJDTYPE* projy = ptr_projector_[iloc][1][p][1];
        KBPROJDTYPE* projz = ptr_projector_[iloc][1][p][2];

        int idx = 0;
        const RadialInter& nlproj(species_.getRadialKBP(1, p));

        double x = kb_proj_start_[0] - center_[0];

        for (int ix = 0; ix < range_kbproj_; ix++)
        {
            double y = kb_proj_start_[1] - center_[1];

            for (int iy = 0; iy < range_kbproj_; iy++)
            {
                double z = kb_proj_start_[2] - center_[2];

                for (int iz = 0; iz < range_kbproj_; iz++)
                {
                    if (is_in_domain[idx])
                    {
                        const double r = sqrt(x * x + y * y + z * z);

                        // Get the projector value in r
                        const double t1 = nlproj.cubint(r);

                        if (r > rthreshold)
                        {
                            double coeff = factor * t1 / r;

                            projx[jcount] = (KBPROJDTYPE)(x * coeff);
                            projy[jcount] = (KBPROJDTYPE)(y * coeff);
                            projz[jcount] = (KBPROJDTYPE)(z * coeff);
                        }
                        else
                        {
                            assert(fabs(t1) < 1.e-3);

                            projx[jcount] = 0.;
                            projy[jcount] = 0.;
                            projz[jcount] = 0.;
                        }
#if CHECK_NORM
                        double dval1 = (double)projx[jcount];
                        double dval2 = (double)projy[jcount];
                        double dval3 = (double)projz[jcount];
                        norm2_[1][3 * p + 0] += dval1 * dval1;
                        norm2_[1][3 * p + 1] += dval2 * dval2;
                        norm2_[1][3 * p + 2] += dval3 * dval3;
#endif
                        jcount++;
                    }
                    idx++;
                    z += h_[2];
                }
                y += h_[1];
            }
            x += h_[0];
        }

        if (jcount != icount)
        {
            mgmol_abort(
                "setPProjector: Problem with non-local projector generation");
        }

    } // loop over multiplicity
}

// Generates d-projectors
void KBprojectorSparse::setDProjector(const short iloc, const int icount)
{
    assert((int)ptr_projector_.size() > iloc);
    assert((int)ptr_projector_[iloc].size() > 2);
    assert(multiplicity_[2] > 0);
    assert((int)ptr_projector_[iloc][2].size() == multiplicity_[2]);

    const bool* const is_in_domain = is_in_domain_[iloc];

    const double sqrt3     = sqrt(3.0);
    const double inv_sqrt3 = 1. / sqrt3;

    const double factor = sqrt(5. / (4. * M_PI));

    for (short p = 0; p < multiplicity_[2]; p++)
    {
        int jcount      = 0;
        KBPROJDTYPE* r1 = ptr_projector_[iloc][2][p][0];
        KBPROJDTYPE* r2 = ptr_projector_[iloc][2][p][1];
        KBPROJDTYPE* r3 = ptr_projector_[iloc][2][p][2];
        KBPROJDTYPE* r4 = ptr_projector_[iloc][2][p][3];
        KBPROJDTYPE* r5 = ptr_projector_[iloc][2][p][4];

        int idx = 0;

        const RadialInter& nlproj(species_.getRadialKBP(2, p));

        double x = kb_proj_start_[0] - center_[0];

        for (int ix = 0; ix < range_kbproj_; ix++)
        {
            double y = kb_proj_start_[1] - center_[1];

            for (int iy = 0; iy < range_kbproj_; iy++)
            {
                double z = kb_proj_start_[2] - center_[2];

                for (int iz = 0; iz < range_kbproj_; iz++)
                {
                    if (is_in_domain[idx])
                    {
                        double rr = x * x + y * y + z * z;

                        if (rr > rthreshold)
                        {
                            double r = sqrt(rr);

                            // Get the radial part of the projector
                            double t1 = nlproj.cubint(r);

                            t1 *= (sqrt3 / rr) * factor;

                            r1[jcount] = (KBPROJDTYPE)(t1 * x * y);
                            r2[jcount] = (KBPROJDTYPE)(t1 * y * z);
                            r3[jcount] = (KBPROJDTYPE)(t1 * z * x);
                            r4[jcount]
                                = (KBPROJDTYPE)(0.5 * t1 * (x * x - y * y));
                            r5[jcount] = (KBPROJDTYPE)(0.5 * t1
                                                       * (sqrt3 * z * z
                                                           - rr * inv_sqrt3));
                        }
                        else
                        {
                            r1[jcount] = 0.;
                            r2[jcount] = 0.;
                            r3[jcount] = 0.;
                            r4[jcount] = 0.;
                            r5[jcount] = 0.;
                        }
#if CHECK_NORM
                        assert(5 * p + 4 < norm2_[2].size());
                        double dval1 = (double)r1[jcount];
                        double dval2 = (double)r2[jcount];
                        double dval3 = (double)r3[jcount];
                        double dval4 = (double)r4[jcount];
                        double dval5 = (double)r5[jcount];

                        norm2_[2][5 * p + 0] += dval1 * dval1;
                        norm2_[2][5 * p + 1] += dval2 * dval2;
                        norm2_[2][5 * p + 2] += dval3 * dval3;
                        norm2_[2][5 * p + 3] += dval4 * dval4;
                        norm2_[2][5 * p + 4] += dval5 * dval5;
#endif
                        jcount++;
                    }

                    idx++;
                    z += h_[2];
                }
                y += h_[1];
            }
            x += h_[0];
        }

        if (jcount != icount)
        {
            mgmol_abort("setDProjector: Problem with non-local generation");
        }

    } // loop over multiplicity
}

// Generates f-projectors
void KBprojectorSparse::setFProjector(const short iloc, const int icount)
{
    assert((int)ptr_projector_.size() > iloc);
    assert((int)ptr_projector_[iloc].size() > 3);
    assert(multiplicity_[3] > 0);
    assert((int)ptr_projector_[iloc][3].size() == multiplicity_[3]);

    const bool* const is_in_domain = is_in_domain_[iloc];

    const double sqrt2 = sqrt(2.0);
    const double sqrt3 = sqrt(3.0);
    const double sqrt5 = sqrt(5.0);
    const double sqrt7 = sqrt(7.0);

    const double factor = sqrt(1. / (2. * M_PI));

    for (short p = 0; p < multiplicity_[3]; p++)
    {
        int jcount      = 0;
        KBPROJDTYPE* r1 = ptr_projector_[iloc][3][p][0];
        KBPROJDTYPE* r2 = ptr_projector_[iloc][3][p][1];
        KBPROJDTYPE* r3 = ptr_projector_[iloc][3][p][2];
        KBPROJDTYPE* r4 = ptr_projector_[iloc][3][p][3];
        KBPROJDTYPE* r5 = ptr_projector_[iloc][3][p][4];
        KBPROJDTYPE* r6 = ptr_projector_[iloc][3][p][5];
        KBPROJDTYPE* r7 = ptr_projector_[iloc][3][p][6];

        int idx = 0;

        const RadialInter& nlproj(species_.getRadialKBP(3, p));

        double x = kb_proj_start_[0] - center_[0];

        for (int ix = 0; ix < range_kbproj_; ix++)
        {
            double y = kb_proj_start_[1] - center_[1];

            for (int iy = 0; iy < range_kbproj_; iy++)
            {
                double z = kb_proj_start_[2] - center_[2];

                for (int iz = 0; iz < range_kbproj_; iz++)
                {
                    if (is_in_domain[idx])
                    {
                        double rr = x * x + y * y + z * z;

                        if (rr > rthreshold)
                        {
                            const double r = sqrt(rr);

                            // Get the radial part of the projector
                            double t1 = nlproj.cubint(r);

                            t1 *= (0.25 * sqrt7 / (r * rr)) * factor;

                            const double x2 = x * x;
                            const double y2 = y * y;
                            const double z2 = z * z;

                            r1[jcount] = (KBPROJDTYPE)(sqrt5 * t1
                                                       * (3. * x2 - y2) * y);
                            r2[jcount]
                                = (KBPROJDTYPE)(2. * sqrt2 * sqrt3 * sqrt5 * t1
                                                * x * y * z);
                            r3[jcount] = (KBPROJDTYPE)(sqrt3 * t1 * y
                                                       * (4. * z2 - x2 - y2));
                            r4[jcount]
                                = (KBPROJDTYPE)(sqrt2 * t1 * z
                                                * (2. * z2 - 3. * x2 - 3 * y2));
                            r5[jcount] = (KBPROJDTYPE)(sqrt3 * t1 * x
                                                       * (4. * z2 - x2 - y2));
                            r6[jcount] = (KBPROJDTYPE)(sqrt2 * sqrt3 * sqrt5
                                                       * t1 * (x2 - y2) * z);
                            r7[jcount] = (KBPROJDTYPE)(sqrt5 * t1
                                                       * (x2 - 3. * y2) * x);
                        }
                        else
                        {
                            r1[jcount] = 0.;
                            r2[jcount] = 0.;
                            r3[jcount] = 0.;
                            r4[jcount] = 0.;
                            r5[jcount] = 0.;
                            r6[jcount] = 0.;
                            r7[jcount] = 0.;
                        }
#if CHECK_NORM
                        assert(7 * p + 6 < norm2_[3].size());
                        double dval1 = (double)r1[jcount];
                        double dval2 = (double)r2[jcount];
                        double dval3 = (double)r3[jcount];
                        double dval4 = (double)r4[jcount];
                        double dval5 = (double)r5[jcount];
                        double dval6 = (double)r6[jcount];
                        double dval7 = (double)r7[jcount];
                        norm2_[3][7 * p + 0] += dval1 * dval1;
                        norm2_[3][7 * p + 1] += dval2 * dval2;
                        norm2_[3][7 * p + 2] += dval3 * dval3;
                        norm2_[3][7 * p + 3] += dval4 * dval4;
                        norm2_[3][7 * p + 4] += dval5 * dval5;
                        norm2_[3][7 * p + 5] += dval6 * dval6;
                        norm2_[3][7 * p + 6] += dval7 * dval7;
#endif
                        jcount++;
                    }

                    idx++;
                    z += h_[2];
                }
                y += h_[1];
            }
            x += h_[0];
        }

        if (jcount != icount)
        {
            mgmol_abort("setDProjector: Problem with non-local generation");
        }

    } // loop over multiplicity
}

void KBprojectorSparse::setKBProjStart()
{
    assert(range_kbproj_ >= 0);
    assert(range_kbproj_ < 256);
    assert(h_[0] > 0.);
    assert(h_[1] > 0.);
    assert(h_[2] > 0.);

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    for (short dir = 0; dir < 3; dir++)
    {
        const double cell_origin = mygrid.origin(dir);

        // n1 = nb of nodes between the boundary and center
        double n1 = (center_[dir] - cell_origin) / h_[dir];
        assert(fabs(n1) < 10000.);

        // get the integral part of n1 in ic
        double i1;
        double f1 = modf(n1, &i1);
        int ic    = (int)i1;
        if (f1 > 0.5) ic++;
        if (f1 < -0.5) ic--;

        kb_proj_start_index_[dir] = ic - (range_kbproj_ >> 1);
        kb_proj_start_[dir] = cell_origin + h_[dir] * kb_proj_start_index_[dir];
        //(*MPIdata::sout)<<"nlproj_start_[i]="<<nlproj_start_[i]<<endl;
        //(*MPIdata::sout)<<"nlstart_[i]     ="<<nlstart_[i]<<endl;

        assert(kb_proj_start_index_[dir] > -10000);
        assert(kb_proj_start_index_[dir] < 10000);
    }
}

// Generate range of indices over which the projectors
//   will be mapped onto the global grid.
//
void KBprojectorSparse::setProjIndices(const short dir)
{
    assert(dir >= 0 && dir < 3);
    assert(range_kbproj_ >= 0);
    assert(range_kbproj_ < 256);
    assert(!std::isnan(kb_proj_start_index_[dir]));

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    const short ngrid      = mygrid.gdim(dir);

    assert(ngrid > 0);
    assert(ngrid < 10000);

    proj_indices_[dir].resize(range_kbproj_);

    short kbix = kb_proj_start_index_[dir];
    for (short ix = 0; ix < range_kbproj_; ix++)
    {
        proj_indices_[dir][ix] = kbix % ngrid;

        while (proj_indices_[dir][ix] < 0)
        {
            proj_indices_[dir][ix] += ngrid; // periodic BC
        }
        assert(proj_indices_[dir][ix] >= 0);
        assert(proj_indices_[dir][ix] < ngrid);

        kbix++;
    }
}

bool KBprojectorSparse::overlapWithBox(
    const short index_low[3], const short index_high[3])
{
    bool map0 = false;
    bool map1 = false;
    bool map2 = false;
    for (short idx = 0; idx < range_kbproj_; idx++)
    {
        if ((proj_indices_[0][idx] >= index_low[0])
            && (proj_indices_[0][idx] <= index_high[0]))
        {
            map0 = true;
            break;
        }
    }
    if (map0)
    {
        for (short idx = 0; idx < range_kbproj_; idx++)
        {
            if ((proj_indices_[1][idx] >= index_low[1])
                && (proj_indices_[1][idx] <= index_high[1]))
            {
                map1 = true;
                break;
            }
        }
        if (map1)
            for (short idx = 0; idx < range_kbproj_; idx++)
            {
                if ((proj_indices_[2][idx] >= index_low[2])
                    && (proj_indices_[2][idx] <= index_high[2]))
                {
                    map2 = true;
                    break;
                }
            }
    }
    return map2;
}

// Generate index array "pvec" and
// an array "is_in_domain" with value true if the node is in the region
int KBprojectorSparse::get_index_array(std::vector<int>& pvec, const short iloc,
    const short index_low[3], const short index_high[3])
{
    assert(range_kbproj_ >= 0);
    assert(range_kbproj_ < 256);

    bool* const is_in_domain = is_in_domain_[iloc];

    const int dimkb2 = range_kbproj_ * range_kbproj_;
    const int dimkb3 = dimkb2 * range_kbproj_;

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    const short dimx = mygrid.dim(0);
    const short dimy = mygrid.dim(1);
    const short dimz = mygrid.dim(2);
    const int incx   = dimy * dimz;

    for (int i = 0; i < dimkb3; i++)
        is_in_domain[i] = false; // default value

    int icount = 0;
    int idx    = 0;
    for (short ix = 0; ix < range_kbproj_; ix++)
    {
        if ((proj_indices_[0][ix] >= index_low[0])
            && (proj_indices_[0][ix] <= index_high[0]))
        {
            const short lix = proj_indices_[0][ix] % dimx;

            for (short iy = 0; iy < range_kbproj_; iy++)
            {
                if ((proj_indices_[1][iy] >= index_low[1])
                    && (proj_indices_[1][iy] <= index_high[1]))
                {

                    const short liy = proj_indices_[1][iy] % dimy;

                    for (short iz = 0; iz < range_kbproj_; iz++)
                    {
                        if ((proj_indices_[2][iz] >= index_low[2])
                            && (proj_indices_[2][iz] <= index_high[2]))
                        {

                            const short liz = proj_indices_[2][iz] % dimz;

                            pvec[icount] = incx * lix + dimz * liy + liz;

                            is_in_domain[idx] = true;

                            icount++;
                        }
                        idx++;
                    }
                }
                else
                {
                    idx += range_kbproj_;
                }
            }
        }
        else
        {
            idx += dimkb2;
        }
    }

    assert(idx == dimkb3);

    return icount;
}

template <typename T>
void KBprojectorSparse::axpySKetT(
    const short iloc, const double alpha, T* const dst) const
{
    assert(multiplicity_[0] == 1);

    const KBPROJDTYPE* const proj(ptr_projector_[iloc][0][0][0]);
    const std::vector<int>& pidx = nlindex_[iloc];
    const int size_nl            = size_nl_[iloc];
    for (int idx = 0; idx < size_nl; idx++)
    {
        dst[pidx[idx]] += (T)(proj[idx] * alpha);
    }
}

template <typename T>
void KBprojectorSparse::axpyKetT(
    const short iloc, const std::vector<double>& alpha, T* const dst) const
{
    const short thread = omp_get_thread_num();
    std::vector<KBPROJDTYPE>& work_proj(work_proj_[thread]);
    assert(work_proj.size() > 0);

    const int size_nl                = size_nl_[iloc];
    KBPROJDTYPE* const loc_work_proj = &work_proj[0];
    memset(loc_work_proj, 0, size_nl * sizeof(KBPROJDTYPE));

    std::vector<const KBPROJDTYPE*> projectors;
    getProjectors(iloc, projectors);

    for (unsigned short i = 0; i < alpha.size(); i++)
    {
        LinearAlgebraUtils<MemorySpace::Host>::MPaxpy(
            size_nl, alpha[i], projectors[i], loc_work_proj);
    }

    const int* const pidx = &nlindex_[iloc][0];
    for (int idx = 0; idx < size_nl; idx++)
    {
        dst[pidx[idx]] += (T)loc_work_proj[idx];
    }
}

bool KBprojectorSparse::setIndexesAndProjectors()
{
    clear();

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    const int nl  = species_.dim_nl();
    const int nl3 = nl * nl * nl;

    std::vector<int> pvec(nl3);
    is_in_domain_ = new bool*[subdivx_];

    setKBProjStart();

    for (short k = 0; k < 3; k++)
        setProjIndices(k);

    // Determine if this ion projector maps onto this subdomain
    short index_low[3], index_high[3];
    index_low[1]  = mygrid.istart(1);
    index_low[2]  = mygrid.istart(2);
    index_high[1] = mymesh->ihigh1();
    index_high[2] = mymesh->ihigh2();

#if CHECK_NORM
    norm2_.resize(4);
    for (short p = 0; p < multiplicity_[0]; ++p)
        norm2_[0].push_back(0.);
    if (multiplicity_.size() > 1)
        for (short p = 0; p < 3 * multiplicity_[1]; ++p)
            norm2_[1].push_back(0.);
    if (multiplicity_.size() > 2)
        for (short p = 0; p < 5 * multiplicity_[2]; ++p)
            norm2_[2].push_back(0.);
    if (multiplicity_.size() > 3)
        for (short p = 0; p < 7 * multiplicity_[3]; ++p)
            norm2_[3].push_back(0.);
#endif

    bool map_nl = false;
    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        index_low[0]  = mymesh->ilow0(iloc);
        index_high[0] = mymesh->ihigh0(iloc);
        bool map2     = overlapWithBox(index_low, index_high);

        if (map2)
        {
            map_nl = true;

            is_in_domain_[iloc] = new bool[nl3];

            // get "pvec" and "is_in_domain"
            int icount = get_index_array(pvec, iloc, index_low, index_high);
            assert(icount <= nl3);
            //            assert(icount <= mymesh->npointsPatch());
            assert(icount > 0);

            setNLindex(iloc, icount, pvec);

            setProjectors(iloc, icount);
        }
        else
        {
            size_nl_[iloc]      = 0;
            is_in_domain_[iloc] = nullptr;
        } // end if map

    } // end for iloc

#if CHECK_NORM
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&norm2_[0][0], multiplicity_[0], MPI_SUM);
    if (onpe0)
        for (short p = 0; p < multiplicity_[0]; ++p)
            cout << "Norm2 S-projector, p=" << p << " ="
                 << norm2_[0][p] * mygrid.vel() << std::endl;

    if (maxl_ > 0)
    {
        mmpi.allreduce(&norm2_[1][0], 3 * multiplicity_[1], MPI_SUM);
        if (onpe0)
            for (short p = 0; p < multiplicity_[1]; ++p)
            {
                cout << "Norm2 Px-projector, p=" << p << " ="
                     << norm2_[1][3 * p + 0] * mygrid.vel() << std::endl;
                cout << "Norm2 Py-projector, p=" << p << " ="
                     << norm2_[1][3 * p + 1] * mygrid.vel() << std::endl;
                cout << "Norm2 Pz-projector, p=" << p << " ="
                     << norm2_[1][3 * p + 2] * mygrid.vel() << std::endl;
            }

        if (maxl_ > 1)
        {
            mmpi.allreduce(&norm2_[2][0], 5 * multiplicity_[2], MPI_SUM);
            if (onpe0)
                for (short p = 0; p < multiplicity_[2]; ++p)
                {
                    cout << "Norm2 D-projector, p=" << p << " ="
                         << norm2_[2][5 * p + 0] * mygrid.vel() << std::endl;
                    cout << "Norm2 D-projector, p=" << p << " ="
                         << norm2_[2][5 * p + 1] * mygrid.vel() << std::endl;
                    cout << "Norm2 D-projector, p=" << p << " ="
                         << norm2_[2][5 * p + 2] * mygrid.vel() << std::endl;
                    cout << "Norm2 D-projector, p=" << p << " ="
                         << norm2_[2][5 * p + 3] * mygrid.vel() << std::endl;
                    cout << "Norm2 D-projector, p=" << p << " ="
                         << norm2_[2][5 * p + 4] * mygrid.vel() << std::endl;
                }
            if (maxl_ > 2)
            {
                mmpi.allreduce(&norm2_[3][0], 7 * multiplicity_[3], MPI_SUM);
                if (onpe0)
                    for (short p = 0; p < multiplicity_[1]; ++p)
                    {
                        cout << "Norm2 F-projector, p=" << p << " ="
                             << norm2_[3][7 * p + 0] * mygrid.vel()
                             << std::endl;
                        cout << "Norm2 F-projector, p=" << p << " ="
                             << norm2_[3][7 * p + 1] * mygrid.vel()
                             << std::endl;
                        cout << "Norm2 F-projector, p=" << p << " ="
                             << norm2_[3][7 * p + 2] * mygrid.vel()
                             << std::endl;
                        cout << "Norm2 F-projector, p=" << p << " ="
                             << norm2_[3][7 * p + 3] * mygrid.vel()
                             << std::endl;
                        cout << "Norm2 F-projector, p=" << p << " ="
                             << norm2_[3][7 * p + 4] * mygrid.vel()
                             << std::endl;
                        cout << "Norm2 F-projector, p=" << p << " ="
                             << norm2_[3][7 * p + 5] * mygrid.vel()
                             << std::endl;
                        cout << "Norm2 F-projector, p=" << p << " ="
                             << norm2_[3][7 * p + 6] * mygrid.vel()
                             << std::endl;
                    }
            }
        }
    }
#endif
    return map_nl;
}

void KBprojectorSparse::setPtrProjectors()
{
    ptr_projector_.resize(subdivx_);

    for (short iloc = 0; iloc < subdivx_; iloc++)
    {
        ptr_projector_[iloc].resize(maxl_ + 1);
        for (short l = 0; l <= maxl_; l++)
        {
            if (l != llocal_)
            {
                assert(multiplicity_[l] > 0);
                ptr_projector_[iloc][l].resize(multiplicity_[l]);
                for (short p = 0; p < multiplicity_[l]; p++)
                    ptr_projector_[iloc][l][p].resize(2 * l + 1);
            }
        }
    }
}

void KBprojectorSparse::getProjectors(
    const short iloc, std::vector<const KBPROJDTYPE*>& projectors) const
{
    projectors.clear();

    for (short l = 0; l <= maxl_; l++)
    {
        if (llocal_ != l)
        {
            for (short p = 0; p < multiplicity_[l]; ++p)
                for (short m = 0; m < (2 * l + 1); ++m)
                {
                    projectors.push_back(getProjector(iloc, l, p, m));
                }
        }
    }
}

// get sign factor for <KB,Psi> matrix for all the projectors
// of Ion
void KBprojectorSparse::getKBsigns(std::vector<short>& kbsigns) const
{
    kbsigns.clear();

    for (short l = 0; l <= maxl_; l++)
    {
        if (llocal_ != l)
        {
            for (short p = 0; p < multiplicity_[l]; p++)
                for (short m = 0; m < 2 * l + 1; m++)
                    kbsigns.push_back(species_.getKBsign(l, p));
        }
    }
}

void KBprojectorSparse::getKBcoeffs(std::vector<double>& coeffs) const
{
    coeffs.clear();

    for (short l = 0; l <= maxl_; l++)
    {
        if (llocal_ != l)
        {
            for (short p = 0; p < multiplicity_[l]; p++)
                for (short m = 0; m < 2 * l + 1; m++)
                    coeffs.push_back(species_.getKBcoeff(l, p));
        }
    }
}

double KBprojectorSparse::dotPsi(const short iloc, const short index) const
{
    short i = 0;
    for (short l = 0; l <= maxl_; l++)
    {
        if (llocal_ != l)
        {
            for (short p = 0; p < multiplicity_[l]; p++)
                for (short m = 0; m < 2 * l + 1; m++)
                {
                    if (i == index) return dotPsi(iloc, l, p, m);
                    i++;
                }
        }
    }

    return 0.;
}

double KBprojectorSparse::maxRadius() const
{
    double radius = 0.;

    if (maxl_ > 0)
        for (short dir = 0; dir < 3; dir++)
        {
            double left  = center_[dir] - kb_proj_start_[dir];
            double right = kb_proj_start_[dir] + (range_kbproj_ - 1) * h_[dir]
                           - center_[dir];

            radius = left > radius ? left : radius;
            radius = right > radius ? right : radius;
        }

    return radius;
}

template void KBprojectorSparse::axpySKetT<double>(
    const short iloc, const double alpha, double* const dst) const;
template void KBprojectorSparse::axpySKetT<float>(
    const short iloc, const double alpha, float* const dst) const;

template void KBprojectorSparse::axpyKetT<double>(const short iloc,
    const std::vector<double>& alpha, double* const dst) const;
template void KBprojectorSparse::axpyKetT<float>(
    const short iloc, const std::vector<double>& alpha, float* const dst) const;
