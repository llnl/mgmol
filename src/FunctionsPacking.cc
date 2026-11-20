// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "FunctionsPacking.h"
#include "Index.h"
#include "LocalizationRegions.h"
#include "MPIdata.h"
#include "SymmetricMatrix.h"
#include "coloring.h"

#include "MGmol_abort.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <list>
#include <numeric>
using namespace std;

FunctionsPacking::FunctionsPacking(std::shared_ptr<LocalizationRegions> lrs,
    const bool global, const MPI_Comm comm)
    : comm_(comm)
{
    setup(lrs, global);
}

FunctionsPacking::FunctionsPacking(const FunctionsPacking& p)
{
    gid2color_   = p.gid2color_;
    global_size_ = p.global_size_;
    num_colors_  = p.num_colors_;
}

// Setup based on
// Recursive Largest First (RLF) algorithm
void FunctionsPacking::setup(
    std::shared_ptr<LocalizationRegions> lrs, const bool global)
{
    Control& ct = *(Control::instance());
    list<list<int>> colored_gids;

    std::vector<int> gids;
    if (global)
        lrs->getGidsGlobal(gids);
    else
        gids = lrs->getOverlapGids();

    // SymmetricMatrix<char> orbi_overlap(numst_,comm_);
    SymmetricMatrix<char>* orbi_overlap;
    if (ct.RLFColoring())
        orbi_overlap = new SymmetricMatrix<char>(gids, comm_);
    else
        orbi_overlap = new SymmetricMatrix<char>(gids.size(), comm_);
    if (global)
    {
        if (onpe0 && ct.verbose > 1)
            (*MPIdata::sout) << " PACK STATES: Global coloring..." << endl;
        initOrbiOverlapGlobal(lrs, 0, *orbi_overlap);
    }
    else
    {
        if (onpe0 && ct.verbose > 1)
            (*MPIdata::sout) << " PACK STATES: Local coloring..." << endl;
        initOrbiOverlapLocal(lrs, 0, *orbi_overlap);
    }

    global_size_ = orbi_overlap->dimension();

    getColors(*orbi_overlap, colored_gids);

    delete orbi_overlap;
}

// compute map "gid2color_" that maps gids to slots
void FunctionsPacking::getColors(
    const SymmetricMatrix<char>& overlaps, list<list<int>>& colored_gids)
{
    assert(overlaps.dimension() < 100000);
    Control& ct = *(Control::instance());

    if (onpe0 && ct.verbose > 0)
    {
        (*MPIdata::sout) << "setup FunctionsPacking for size=" << global_size_
                         << endl;
    }

    if (!ct.RLFColoring())
    {
        greedyColor(overlaps, colored_gids, (ct.verbose > 0 && onpe0),
            (*MPIdata::sout));
    }
    else
    {
        colorRLF(overlaps, colored_gids, (ct.verbose > 0 && onpe0),
            (*MPIdata::sout));
    }

    num_colors_ = (short)colored_gids.size();
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "FunctionsPacking::num_colors_= " << num_colors_
                         << endl;

    assert(num_colors_ < 10000);

    vector<short> nb_orb(num_colors_);

    // this is where we actually attribute gids to colors to slots
    short color = 0;

    // loop over colored_gids
    list<list<int>>::const_iterator ic = colored_gids.begin();
    while (ic != colored_gids.end())
    {
        nb_orb[color] = (short)ic->size();
        //(*MPIdata::sout)<<"nb_orb["<<color<<"]="<<nb_orb[color]<<endl;

        // loop over functions of same color
        list<int>::const_iterator ii = ic->begin();
        while (ii != ic->end())
        {
            int gid = *ii;
            gid2color_.insert(pair<int, short>(gid, color));
            // if( onpe0 )
            //    (*MPIdata::sout)<<"Orbital "<<(*ii)<<" of color
            //    "<<gid2color_[(*ii)]<<endl;

            ii++;
        }
        color++;
        ic++;
    }

    // final check
    ic = colored_gids.begin();
    while (ic != colored_gids.end())
    {
        list<int>::const_iterator ii = ic->begin();
        while (ii != ic->end())
        {
            assert(gid2color_.find(*ii)->second < num_colors_);
            ii++;
        }
        ic++;
    }

    int ntot_orb = accumulate(nb_orb.begin(), nb_orb.end(), 0);

    if (ntot_orb != global_size_)
    {
        if (onpe0)
            std::cerr << ntot_orb << " orbitals instead of " << global_size_
                      << endl;
        mgmol_abort();
    }
}

short FunctionsPacking::checkOverlapLRs(
    std::shared_ptr<LocalizationRegions> lrs, const int gid1,
    const int gid2) const
{
    return lrs->overlap(gid1, gid2) ? 1 : 0;
}

//  Initialize the array orbi_overlap_[j][i] telling if
//  the orbitals i and j overlap somewhere (on any PE or region)
// at level "level"
void FunctionsPacking::initOrbiOverlapLocal(
    std::shared_ptr<LocalizationRegions> lrs, const short level,
    SymmetricMatrix<char>& orbi_overlap)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 2)
        (*MPIdata::sout) << "LocGridOrbitals::initOrbiOverlapLocal() for level "
                         << level << endl;

    const vector<int>& local_gids(lrs->getOverlapGids());

    assert(static_cast<unsigned int>(orbi_overlap.dimension())
           == local_gids.size());

    // Initialization loop for orbi_overlap over all the states
    for (vector<int>::const_iterator it1 = local_gids.begin();
         it1 != local_gids.end(); it1++)
    {
        const int gid1 = *it1;
        orbi_overlap.setVal(gid1, gid1, 1);
        for (vector<int>::const_iterator it2 = local_gids.begin(); it2 != it1;
             it2++)
        {
            const int gid2 = *it2;
            short oo1      = checkOverlapLRs(lrs, gid1, gid2);
            orbi_overlap.setVal(gid1, gid2, oo1);
        }
    }
}

void FunctionsPacking::initOrbiOverlapGlobal(
    std::shared_ptr<LocalizationRegions> lrs, const short level,
    SymmetricMatrix<char>& orbi_overlap)
{
    Control& ct   = *(Control::instance());
    const int dim = orbi_overlap.dimension();
    if (onpe0 && ct.verbose > 2)
    {
        (*MPIdata::sout)
            << "LocGridOrbitals::initOrbiOverlapGlobal() for level " << level
            << endl;
        (*MPIdata::sout) << "LocGridOrbitals::initOrbiOverlapGlobal() for size "
                         << dim << endl;
    }

    // Initialization loop for orbi_overlap over all the states
    for (int gid1 = 0; gid1 < dim; gid1++)
    {
        orbi_overlap.setVal(gid1, gid1, 1);
        for (int gid2 = 0; gid2 < gid1; gid2++)
        {
            short oo1 = checkOverlapLRs(lrs, gid1, gid2);
            // short oo1=checkOverlap(gid1, gid2,level);
            // if( oo1!=oo2 )
            //{
            //    cout<<"mype="<<mype<<", gid1="<<gid1<<", gid2="<<gid2<<",
            //    oo1="<<oo1<<", oo2="<<oo2<<endl;
            //}
            // orbi_overlap_->setVal(gid1,gid2,checkOverlap(gid1, gid2,level));
            orbi_overlap.setVal(gid1, gid2, oo1);
        }
    }
    orbi_overlap.mpiAllOr();
}
