// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "MasksSet.h"
#include "GridMaskMax.h"
#include "GridMaskMult.h"
#include "LocalizationRegions.h"
#include "Mesh.h"

#define USE_MASKMAX 1

using namespace std;

const double r1               = 0.99999;
const double inv_one_minus_r1 = 1. / (1. - r1);

// const double rc_exp=0.75;
const double rc_exp2 = 0.75;
// const double rc_cos=0.5;

// const double fac_exp =1.*(1./(1.-rc_exp));
const double fac_exp2 = 2. * (1. / (1. - rc_exp2)) * (1. / (1. - rc_exp2));
// const double fac_cos =1./(1.-rc_cos);

// const double exp_rc = exp(-fac_exp*(1.-rc_exp));
const double exp2_rc = exp(-fac_exp2 * (1. - rc_exp2) * (1. - rc_exp2));
// const double exp_coeff = 1./(1.-exp_rc);
const double exp2_coeff = 1. / (1. - exp2_rc);

// step function made C0 to avoid numerical differences
// between runs
static lmasktype one(const double r)
{
    assert(r <= 1.);
    assert(r >= 0.);

    if (r < r1)
        return 1.;
    else
        return (lmasktype)((1. - r) * inv_one_minus_r1);
}

// static float one(const float r)
//{
//    assert( r<=1. );
//    assert( r>=0. );
//
//    if( r<r1 )return 1.;
//    else      return (1.-r)*inv_one_minus_r1;
//}

// static double cosinus(const double r)
//{
//    assert( r<=1. );
//    assert( r>=0. );
//
//    double val=1.;
//    if( r>rc_cos )val = cos(fac_cos*(r-rc_cos)*M_PI_2);
//    assert( val>=0. );
//    assert( val<=1. );
//
//    return val;
//}

// static float cosinus(const float r)
//{
//    assert( r<=1. );
//    assert( r>=0. );
//
//    float val=1.;
//    if( r>rc_cos )val = cos(fac_cos*(r-rc_cos)*M_PI_2);
//    assert( val>=0. );
//    assert( val<=1. );
//
//    return val;
//}

// static float expDecay(const float r)
//{
//    assert( r<=1. );
//    assert( r>=0. );
//
//    float val=1.;
//    if( r>rc_exp )val = (exp(-fac_exp*(r-rc_exp))-exp_rc)*exp_coeff;
//    assert( val>=0. );
//    assert( val<=1. );
//
//    return val;
//}

// static double expDecay(const double r)
//{
//    assert( r<=1. );
//    assert( r>=0. );
//
//    float val=1.;
//    if( r>rc_exp )val = (exp(-fac_exp*(r-rc_exp))-exp_rc)*exp_coeff;
//    assert( val>=0. );
//    assert( val<=1. );
//
//    return val;
//}

static lmasktype gaussianDecay(const double r)
{
    assert(r <= 1.);
    assert(r >= 0.);

    float val = 1.;
    if (r > rc_exp2)
        val = (exp(-fac_exp2 * (r - rc_exp2) * (r - rc_exp2)) - exp2_rc)
              * exp2_coeff;
    assert(val >= 0.);
    assert(val <= 1.);

    return (lmasktype)val;
}

void MasksSet::clear()
{
    for (map<int, GridMask*>::const_iterator it = pgrid_masks_.begin();
         it != pgrid_masks_.end(); it++)
    {
        assert(it->second != NULL);
        delete it->second;
    }

    pgrid_masks_.clear();
}

void MasksSet::setup(const std::shared_ptr<LocalizationRegions> lrs,
    const double override_radius)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "MasksSet::setup()" << endl;
    if (lrs->globalNumLRs() == 0) return;

    const vector<int>& gids(lrs->getOverlapGids());
    allocate(gids);

    for (unsigned short ln = 0; ln < mg_levels_; ln++)
    {
        initialize(lrs, ln, override_radius);
    }

    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "MasksSet: Total: " << pgrid_masks_.size()
                         << " masks" << endl;
}

void MasksSet::allocate(const vector<int>& gids)
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    const int subdivx      = mymesh->subdivx();

    // Build masks for functions
    for (vector<int>::const_iterator it = gids.begin(); it != gids.end(); it++)
    {
        GridMask* grid_mask = nullptr;
        if (!type_corr_mask_)
        {
#if USE_MASKMAX
            grid_mask = new GridMaskMax(mg_levels_, subdivx, mygrid);
#else
            grid_mask = new GridMaskMult(mg_levels_, subdivx, mygrid);
#endif
        }
        else
        {
            grid_mask = new GridMaskMult(mg_levels_, subdivx, mygrid);
        }
        pgrid_masks_.insert(pair<int, GridMask*>(*it, grid_mask));
    }
}

MasksSet& MasksSet::operator=(const MasksSet& masks)
{
    if (this == &masks) return *this;

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    const int subdivx      = mymesh->subdivx();

    clear();

    map<int, GridMask*> mmap = masks.pgrid_masks_;
    for (map<int, GridMask*>::const_iterator it = mmap.begin();
         it != mmap.end(); it++)
    {
        // Build masks for functions
        GridMask* grid_mask = new GridMaskMax(mg_levels_, subdivx, mygrid);
        pgrid_masks_.insert(pair<int, GridMask*>(it->first, grid_mask));
        *grid_mask = *(it->second);
    }

    return *this;
}

//
void MasksSet::update(const std::shared_ptr<LocalizationRegions> lrs)
{
    assert(lrs->volume() > 0.);

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Reset localization masks" << endl;

    clear();
    setup(lrs);
}

// Initialize pgrid_masks_ centered on vector of positions
int MasksSet::initialize(const std::shared_ptr<LocalizationRegions> lrs,
    const unsigned short ln, const double override_radius)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << " MasksSet::initialize(), ln=" << ln
                         << ", n=" << pgrid_masks_.size() << endl;

#if USE_MASKMAX
    lmasktype (*maskfunc)(const double)
        = type_corr_mask_ ? &one : &gaussianDecay;
#else
    lmasktype (*maskfunc)(const double) = type_corr_mask_ ? &cosinus : &one;
#endif

    map<int, GridMask*>::iterator it;
    if (override_radius < 0.1)
    {
        for (it = pgrid_masks_.begin(); it != pgrid_masks_.end(); it++)
        {
            const int gid = it->first;
            Vector3D center;
            const double rc = lrs->getCenterAndRadius(gid, center);
            // mask is created here
            (it->second)->init(center, rc, ln, maskfunc);
        }
    }
    else
    {
        for (it = pgrid_masks_.begin(); it != pgrid_masks_.end(); it++)
        {
            int gid = it->first;
            (it->second)
                ->init(lrs->getCenter(gid), override_radius, ln, maskfunc);
        }
    }

#ifdef DEBUG
    if (onpe0)
    {
        (*MPIdata::sout) << " initialize done: ln " << ln << endl;
    }
#endif
    return 0;
}
