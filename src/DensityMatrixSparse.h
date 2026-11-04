// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_DENSITYMATRIXSPARSE_H
#define MGMOL_DENSITYMATRIXSPARSE_H

#include "ClusterOrbitals.h"
#include "DataDistribution.h"
#include "LocalizationRegions.h"
#include "MPIdata.h"
#include "Orbitals.h"
#include "VariableSizeMatrix.h"

#include <vector>

using memory_space_type = typename Orbitals::memory_space_type;

#define DM_NPRINT_ROWS_AND_COLS 5

class DensityMatrixSparse
{
    static Timer gather_DM_tm_;
    int dim_;
    std::vector<int> locvars_;
    std::vector<int> locfcns_;

    VariableSizeMatrix<sparserow>* dm_;

    int update_index_;

    double orbital_occupation_;

public:
    DensityMatrixSparse(std::shared_ptr<LocalizationRegions> lrs,
        const int ndim, const std::vector<int>& locvars);
    DensityMatrixSparse(const DensityMatrixSparse&);

    ~DensityMatrixSparse();

    void setUniform(const double nel);
    void setAufbau(const double nel);

    void setto2InvS(const VariableSizeMatrix<sparserow>& invS);
    int getIndex() const { return update_index_; }
    void setMatrix(const VariableSizeMatrix<sparserow>& mat)
    {
        *dm_ = mat;
        update_index_++;
    }
    void assembleMatrixFromCenteredData(const std::vector<double>& data,
        const std::vector<int>& locRowIds, const int* globalColIds,
        DataDistribution& dtor_DM, const int orbitals_index);
    double getEntry(const int row, const int col) const
    {
        return dm_->get_value(row, col);
    }

    void getEntries(const int row, const std::vector<int>& cols,
        std::vector<double>& values) const
    {
        dm_->getSortedRowValues(row, cols, values);

        assert(values.size() == cols.size());
    }
    double getTraceDotProductWithMat(VariableSizeMatrix<sparserow>* vsmat);
    VariableSizeMatrix<sparserow>* mat() { return dm_; }
    void printDM(std::ostream& os, int nrows = NUM_PRINT_ROWS) const;
    void getLocalMatrix(LocalMatrices<MATDTYPE, memory_space_type>& localX,
        const std::vector<std::vector<int>>& global_indexes);
    void printTimers(std::ostream& os) { gather_DM_tm_.print(os); }
};

#endif
