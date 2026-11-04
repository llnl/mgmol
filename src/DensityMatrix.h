// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_DENSITYMATRIX_H
#define MGMOL_DENSITYMATRIX_H

#include "HDFrestart.h"
#include "MGmol_MPI.h"
#include "global.h"

#include <cassert>
#include <cstring>
#include <ostream>
#include <vector>

#define DM_NPRINT_ROWS_AND_COLS 5

template <class MatrixType>
class DensityMatrix
{
    const int dim_;
    std::vector<double> occupation_;

    MatrixType* dm_;
    MatrixType* kernel4dot_;
    MatrixType* work_;

    /*!
     * Keep track of changes, incremented every time dm_ is updated
     */
    int update_index_;

    bool occ_uptodate_;
    bool uniform_occ_;
    bool stripped_;

    /*!
     * Max. occupation of an orbital: 1 with spin, 2 otherwise
     */
    double orbital_occupation_;

    DensityMatrix();
    DensityMatrix& operator=(const DensityMatrix&);
    DensityMatrix(const DensityMatrix&);

    void build();

public:
    DensityMatrix(const int ndim);

    ~DensityMatrix();

    /*!
     * initialize with same occupation in all states
     */
    void setUniform(const double nel);

    /*!
     * initialize by occupying lowest states
     */
    void setAufbau(const double nel);

    int getIndex() const { return update_index_; }

    bool occupationsUptodate() const { return occ_uptodate_; }
    bool fromUniformOccupations() const { return uniform_occ_; }

    double dot(const MatrixType& mat)
    {
        assert(!stripped_);
        return dm_->traceProduct(mat);
    }

    void print(std::ostream& os) const
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());

        assert(!stripped_);
        if (mmpi.instancePE0()) os << " DensityMatrix" << std::endl;
        dm_->print(os, 0, 0, DM_NPRINT_ROWS_AND_COLS, DM_NPRINT_ROWS_AND_COLS);
    }

    const MatrixType& getMatrix() const
    {
        assert(!stripped_);
        assert(dm_ != 0);
        return *dm_;
    }

    const MatrixType& kernel4dot() const { return *kernel4dot_; }

    void setMatrix(const MatrixType& mat)
    {
        *dm_ = mat;
        update_index_++;

        setDummyOcc();

        occ_uptodate_ = false;
        uniform_occ_  = false;
        stripped_     = false;
    }

    // set occupations to meaningless values to catch uninitialized use
    void setDummyOcc()
    {
        for (auto& occ : occupation_)
            occ = -1.;
    }

    void initMatrix(const double* const val)
    {
        dm_->init(val, dim_);
        setDummyOcc();

        occ_uptodate_ = false;
        uniform_occ_  = false;
        stripped_     = false;

        update_index_ = 0;
    }

    void getOccupations(std::vector<double>& occ) const
    {
        assert(!occupation_.empty());
        assert(occ_uptodate_);
        assert((int)occ.size() == dim_);

        memcpy(&occ[0], &occupation_[0], dim_ * sizeof(double));
    }

    void setOccupations(const std::vector<double>& occ);

    void setto2InvS(const MatrixType& invS);

    void stripS(const MatrixType& ls);
    void dressUpS(const MatrixType& ls);

    // dm_ -> u*dm_*u^T
    void transform(const MatrixType& u);

    double computeEntropy() const;
    void computeOccupations(const MatrixType& ls);
    void build(const std::vector<double>& occ);
    void build(const MatrixType& z);
    void build(const MatrixType& z, const std::vector<double>& occ);

    void rotate(const MatrixType& rotation_matrix, const bool flag_eigen);
    void printOccupations(std::ostream& os) const;
    void diagonalize(const MatrixType& ls, std::vector<double>& occ);
    void diagonalize(
        const char eigv, std::vector<double>& occ, MatrixType& vect);
    double getExpectation(const MatrixType& A);
    void mix(const double mix, const MatrixType& matA);

    /*!
     * dm <- dm + (dm-previous_dm) = 2.*dm - previous_dm
     */
    void linearExtrapolate(const MatrixType& previous_dm);

    int write(HDFrestart& h5f_file, std::string& name);
    int read(HDFrestart& h5f_file, std::string& name);
};

#endif
