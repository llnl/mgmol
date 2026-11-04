// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include <iomanip>
#include <iostream>
using namespace std;

#include "DensityMatrixSparse.h"
#include "MGmol_MPI.h"
#include <string.h>

Timer DensityMatrixSparse::gather_DM_tm_("DensityMatrixSparse::gather_DM");
DensityMatrixSparse::DensityMatrixSparse(
    std::shared_ptr<LocalizationRegions> lrs, const int ndim,
    const std::vector<int>& locvars)
    : dim_(ndim), locvars_(locvars)
{
    assert(ndim >= 0);

    MGmol_MPI& mmpi     = *(MGmol_MPI::instance());
    orbital_occupation_ = mmpi.nspin() > 1 ? 1. : 2.;

    update_index_ = -1;

    if (dim_ > 0)
    {
        dm_ = new VariableSizeMatrix<sparserow>("DM", 4096);
    }
    lrs->getLocalSubdomainIndices(locfcns_);
}

DensityMatrixSparse::~DensityMatrixSparse()
{
    if (dim_ > 0)
    {
        assert(dm_ != nullptr);
        delete dm_;
    }
}

void DensityMatrixSparse::setUniform(const double nel)
{
    const double occ = (double)((double)nel / (double)dim_);
    assert(occ < 1.01);
    update_index_++;
    (*dm_).reset();
    const double uval = (double)occ * orbital_occupation_;
    for (std::vector<int>::const_iterator st = locvars_.begin();
         st != locvars_.end(); ++st)
        (*dm_).insertMatrixElement(*st, *st, uval, INSERT, true);
}

void DensityMatrixSparse::setAufbau(const double nel)
{
    dm_->reset();
    double sum                          = 0;
    std::vector<int>::const_iterator st = locvars_.begin();
    while (sum <= nel - 1)
    {
        dm_->insertMatrixElement(*st, *st, 1, INSERT, true);
        st++;
        sum += 1;
    }
    if (st != locvars_.end())
    {
        dm_->insertMatrixElement(*st, *st, nel - sum, INSERT, true);
        st++;
    }
    while (st != locvars_.end())
    {
        dm_->insertMatrixElement(*st, *st, 0, INSERT, true);
        st++;
    }
}

void DensityMatrixSparse::setto2InvS(const VariableSizeMatrix<sparserow>& invS)
{
    *dm_ = invS;
    dm_->scale(orbital_occupation_);

    update_index_++;
}
// build density matrix, given computed locally centered data
void DensityMatrixSparse::assembleMatrixFromCenteredData(
    const std::vector<double>& data, const std::vector<int>& localRowIds,
    const int* globalColIds, DataDistribution& dtor_DM,
    const int orbitals_index)
{
    assert((int)data.size() >= (int)localRowIds.size());
    assert((int)data.size() % (int)localRowIds.size() == 0);
    const int n = (int)data.size() / (int)localRowIds.size();
    // initialize sparse ordering of rows
    dm_->setupSparseRows(locvars_);
    // assemble centered rows
    int pos = 0;
    for (std::vector<int>::const_iterator it = localRowIds.begin();
         it != localRowIds.end(); ++it)
    {
        dm_->initializeLocalRow(n, *it, globalColIds, &data[pos]);
        pos += n;
    }

    // Distribute local rows/cols of DM and gather data from neighbors
    gather_DM_tm_.start();
    /* Distribute local inverse and gather data */
    dtor_DM.updateLocalRows((*dm_));
    gather_DM_tm_.stop();

    update_index_ = orbitals_index;
}
// compute trace of dot product dm_ . vsmat
double DensityMatrixSparse::getTraceDotProductWithMat(
    VariableSizeMatrix<sparserow>* vsmat)
{
    assert(dm_ != nullptr);
    //   assert(dm_->n() != 0);
    /* compute trace */
    double trace = 0.0;
    for (std::vector<int>::iterator itr = locfcns_.begin();
         itr != locfcns_.end(); ++itr)
    {
        trace += (*dm_).AmultSymBdiag(vsmat, *itr);
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    mmpi.allreduce(&trace, 1, MPI_SUM);

    return trace;
}
/* print a few rows of the density matrix */
void DensityMatrixSparse::printDM(ostream& os, int nrows) const
{
    assert(dm_ != nullptr);
    //  assert( dm_->n() > 0);

    (*dm_).print(os, locfcns_, nrows);

    return;
}
// get submatrix of Density matrix
void DensityMatrixSparse::getLocalMatrix(
    LocalMatrices<MATDTYPE, memory_space_type>& localX,
    const vector<vector<int>>& global_indexes)
{
    assert(dm_ != nullptr);

    const short subdiv           = (short)global_indexes.size();
    const short chromatic_number = (short)global_indexes[0].size();

    if (chromatic_number == 0) return;
    if (dm_->n() == 0) return;

    for (short iloc = 0; iloc < subdiv; iloc++)
    {
        std::vector<MATDTYPE> tmp(chromatic_number * chromatic_number);
        memset(tmp.data(), 0,
            chromatic_number * chromatic_number * sizeof(MATDTYPE));

        const vector<int>& gids(global_indexes[iloc]);
#pragma omp parallel for
        for (int icolor = 0; icolor < chromatic_number; icolor++)
        {
            const int st1 = gids[icolor];
            if (st1 == -1) continue;
            for (int jcolor = 0; jcolor < chromatic_number; jcolor++)
            {
                const int st2 = gids[jcolor];
                if (st2 == -1) continue;
                tmp[icolor + chromatic_number * jcolor]
                    = (MATDTYPE)(*dm_).get_value(st1, st2);
            }
        }
        localX.setValues(tmp.data(), chromatic_number, iloc);
    }
}
