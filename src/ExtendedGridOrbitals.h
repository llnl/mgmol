// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_EXTENDEDGRIDORBITALS_H
#define MGMOL_EXTENDEDGRIDORBITALS_H

#include "BlockVector.h"
#ifdef MGMOL_USE_SCALAPACK
#include "DistMatrix.h"
#endif
#include "DotProductManager.h"
#include "GridFunc.h"
#include "HDFrestart.h"
#include "Lap.h"
#include "MPIdata.h"
#include "MasksSet.h"
#include "Mesh.h"
#include "Orbitals.h"
#include "ReplicatedMatrix.h"
#include "SinCosOps.h"
#include "SquareLocalMatrices.h"
#include "XYZOps.h"

#include "hdf5.h"
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class ProjectedMatricesInterface;
class LocalizationRegions;
class ClusterOrbitals;
#ifndef MGMOL_USE_SCALAPACK
typedef double DISTMATDTYPE;
#endif

template <typename ScalarType>
class ExtendedGridOrbitals : public Orbitals
{
private:
    const std::string name_;

    ////////////////////////////////////////////////////////
    // common data shared by all instances of class
    ////////////////////////////////////////////////////////
    static Timer matB_tm_;
    static Timer invBmat_tm_;
    static Timer overlap_tm_;
    static Timer dot_product_tm_;
    static Timer addDot_tm_;
    static Timer prod_matrix_tm_;
    static Timer assign_tm_;
    static Timer normalize_tm_;
    static Timer axpy_tm_;
    static Timer gemm_nn_tm_;
    static Timer gemm_tn_tm_;

    static int lda_; // leading dimension for storage
    static int numpt_;

    static DotProductManager<ExtendedGridOrbitals<ScalarType>>*
        dotProductManager_;

    static int data_wghosts_index_;

    static int numst_;

    ////////////////////////////////////////////////////////
    // common data shared by copies of object (and copied whan a copy is made)
    ////////////////////////////////////////////////////////

    // pointers to objects owned outside class
    ProjectedMatricesInterface* proj_matrices_;

    ////////////////////////////////////////////////////////
    // instance specific data
    ////////////////////////////////////////////////////////
    BlockVector<ScalarType, memory_space_type> block_vector_;

    ////////////////////////////////////////////////////////
    //
    // private functions
    //
    void projectOut(ScalarType* const, const int);

    void multiply_by_ReplicatedMatrix(const ReplicatedMatrix& matrix,
        const double alpha, ExtendedGridOrbitals<ORBDTYPE>& product);
#ifdef MGMOL_USE_SCALAPACK
    void multiply_by_DistMatrix(
        const dist_matrix::DistMatrix<DISTMATDTYPE>& matrix, const double alpha,
        ExtendedGridOrbitals<ORBDTYPE>& product);
#endif
    void multiply_by_matrix(
        const DISTMATDTYPE* const, ScalarType*, const int) const;
    void scal(const int i, const double alpha) { block_vector_.scal(i, alpha); }
    virtual void assign(const int i, const ScalarType* const v, const int n = 1)
    {
        block_vector_.assign(i, v, n);
    }
    ExtendedGridOrbitals& operator=(const ExtendedGridOrbitals& orbitals);
    ExtendedGridOrbitals();

    void computeMatB(const ExtendedGridOrbitals&, const pb::Lap<ScalarType>&);

    void computeLocalProduct(const ScalarType* const, const int,
        LocalMatrices<MATDTYPE, MemorySpace::Host>&,
        const bool transpose = false);
#ifdef HAVE_MAGMA
    void computeLocalProduct(const ScalarType* const, const int,
        LocalMatrices<MATDTYPE, MemorySpace::Device>&,
        const bool transpose = false);
#endif

    void computeGlobalIndexes();
    void computeInvNorms2(std::vector<std::vector<double>>& inv_norms2) const;
    void computeDiagonalGram(VariableSizeMatrix<sparserow>& diagS) const;

    /*!
     * Specialized functions
     */
#ifdef MGMOL_USE_SCALAPACK
    void addDotWithNcol2DistMatrix(
        ExtendedGridOrbitals&, dist_matrix::DistMatrix<DISTMATDTYPE>&) const;
#endif
    void addDotWithNcol2ReplicatedMatrix(
        ExtendedGridOrbitals&, ReplicatedMatrix&) const;

    void initFourier();
    void initRand();

    ScalarType* psi(const int i) const { return block_vector_.vect(i); }

    void app_mask(const int, ScalarType*, const short) const {};
#ifdef HAVE_MAGMA
    void multiplyByMatrix(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Device>& matrix,
        ScalarType* product, const int ldp) const;
#endif
    void multiplyByMatrix(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix,
        ScalarType* product, const int ldp) const;

    void setup();

protected:
    const pb::Grid& grid_;

    // indexes corresponding to valid function in each subdomain
    static std::vector<std::vector<int>> overlapping_gids_;

public:
    friend class SinCosOps<ExtendedGridOrbitals>;
    friend class XYZOps<ExtendedGridOrbitals>;

    double norm() const;

    const std::vector<int>& getAllOverlappingGids() const
    {
        return overlapping_gids_[0];
    }
    const std::vector<int>& getLocalGids() const
    {
        return overlapping_gids_[0];
    }

    ExtendedGridOrbitals(std::string name, const pb::Grid& my_grid,
        const short subdivx, const int numst, const short bc[3],
        ProjectedMatricesInterface*, std::shared_ptr<LocalizationRegions>,
        MasksSet* masks, MasksSet* corrmasks, ClusterOrbitals* local_cluster,
        const bool setup_flag = true);

    ExtendedGridOrbitals(const std::string& name, const ExtendedGridOrbitals& A,
        const bool copy_data = true);
    ExtendedGridOrbitals(const std::string& name, const ExtendedGridOrbitals& A,
        ProjectedMatricesInterface* proj_matrices, const bool copy_data = true);

    virtual ~ExtendedGridOrbitals();

    static void printTimers(std::ostream& os);

    void resetDotProductMatrices();

    void reset(MasksSet* masks, MasksSet* corrmasks,
        std::shared_ptr<LocalizationRegions> lrs);

    virtual void assign(const ExtendedGridOrbitals& orbitals);
    void copyDataFrom(const ExtendedGridOrbitals& src);

    ProjectedMatricesInterface* getProjMatrices()
    {
        assert(proj_matrices_ != nullptr);
        return proj_matrices_;
    }

    const ProjectedMatricesInterface* projMatrices() const
    {
        return proj_matrices_;
    }

    int numst(void) const { return numst_; }
    int getLda() const { return lda_; }
    int getLocNumpt() const { return numpt_; }
    int getNumpt() const { return numpt_; }

    bool isCompatibleWith(const ExtendedGridOrbitals&) const { return true; }

    void resetDataWithGhostsIndex() { data_wghosts_index_ = 0; }

    void setDataWithGhosts(const bool force = false)
    {
        if (data_wghosts_index_ == getIterativeIndex() && !force) return;

        block_vector_.setDataWithGhosts();

        // if( onpe0 )
        //    (*MPIdata::sout)<<"setDataWithGhosts with iterative index
        //    "<<getIterativeIndex()<<endl;
        data_wghosts_index_ = getIterativeIndex();
    }

    template <typename T>
    void setDataWithGhosts(
        pb::GridFuncVector<T, memory_space_type>* data_wghosts)
    {
        assert(data_wghosts != 0);

        block_vector_.setDataWithGhosts(data_wghosts);
    }
    pb::GridFunc<ScalarType>& getFuncWithGhosts(const int i)
    {
        //(*MPIdata::sout)<<" data_wghosts_index_="<<data_wghosts_index_
        //    <<" getIterativeIndex()   ="<<getIterativeIndex()<<endl;
        if (data_wghosts_index_ != getIterativeIndex())
        {
            setDataWithGhosts();
        }
        assert(data_wghosts_index_ == getIterativeIndex());
        //(*MPIdata::sout)<<"getFuncWithGhosts with index
        //"<<getIterativeIndex()<<endl;
        return block_vector_.getVectorWithGhosts(i);
    }

    pb::GridFuncVector<ScalarType, memory_space_type>* getPtDataWGhosts()
    {
        return block_vector_.getPtDataWGhosts();
    }

    void trade_boundaries()
    {
        static int last_index_traded = -1;

        if (data_wghosts_index_ != last_index_traded)
        {
#ifdef PRINT_OPERATIONS
            if (onpe0)
                (*MPIdata::sout)
                    << "ExtendedGridOrbitals::trade_boundaries()" << std::endl;
#endif
            block_vector_.trade_boundaries();
            last_index_traded = data_wghosts_index_;
        }
    }

    void set_storage(ScalarType* new_storage)
    {
        assert(new_storage != 0);
        block_vector_.setStorage(new_storage);
    }
    ScalarType* getPsi(const int i, const int iloc = 0) const
    {
        (void)iloc;
        return block_vector_.vect(i);
    }
    template <typename T>
    void setPsi(const pb::GridFunc<T>& gf_work, const int ist)
    {
        block_vector_.assignComponent(gf_work, ist);
    }
    template <typename T>
    void setPsi(const pb::GridFuncVector<T, memory_space_type>& gf_work)
    {
        block_vector_.assign(gf_work);
    }
    void setToDataWithGhosts() { block_vector_.setToDataWithGhosts(); }
    int chromatic_number(void) const
    {
        assert(numst_ < 10000);
        return numst_;
    }
    void applyDiagonalOp(
        const std::vector<POTDTYPE>& v, ExtendedGridOrbitals& hphi) const
    {
        block_vector_.applyDiagonalOp(v, hphi.block_vector_);
    }

    short subdivx(void) const { return 1; }
    void printChromaticNumber(std::ostream& os) const
    {
        if (onpe0) os << " Max. chromatic_number: " << numst_ << std::endl;
    }
    void printNumst(std::ostream& os) const
    {
        if (onpe0) os << " Number of states   = " << numst_ << std::endl;
    }
    void computeBAndInvB(const pb::Lap<ScalarType>& LapOper);

    void computeGram(const int verbosity = 0);
    void computeGramAndInvS(const int verbosity = 0);
#ifdef MGMOL_USE_SCALAPACK
    void computeGram(dist_matrix::DistMatrix<DISTMATDTYPE>& gram_mat);
    void computeGram(const ExtendedGridOrbitals& orbitals,
        dist_matrix::DistMatrix<DISTMATDTYPE>& gram_mat);
#endif

    ScalarType maxAbsValue() const { return block_vector_.maxAbsValue(); }

    /*!
     * use predefined (default) dot product type
     */
    double dotProduct(const ExtendedGridOrbitals& orbitals);
    /*!
     * use different dot product type
     */
    double dotProduct(const ExtendedGridOrbitals&, const short dot_type);

    static void setDotProduct(const short dot_type);
    void computeDiagonalElementsDotProduct(const ExtendedGridOrbitals& orbitals,
        std::vector<DISTMATDTYPE>& ss) const;

    void computeLocalProduct(const ExtendedGridOrbitals&,
        LocalMatrices<MATDTYPE, MemorySpace::Host>&,
        const bool transpose = false);
    void getLocalOverlap(SquareLocalMatrices<MATDTYPE, MemorySpace::Host>&);
    void getLocalOverlap(const ExtendedGridOrbitals& orbitals,
        SquareLocalMatrices<MATDTYPE, MemorySpace::Host>&);

    template <class MatrixType>
    void addDotWithNcol2Matrix(ExtendedGridOrbitals&, MatrixType&) const;

    void scal(const double alpha)
    {
        block_vector_.scal(alpha);
        incrementIterativeIndex();
    }
    void projectOut(ExtendedGridOrbitals&);

    void normalize();
    void orthonormalize2states(const int st1, const int st2);
    void orthonormalizeLoewdin(const bool overlap_uptodate = false,
        SquareLocalMatrices<MATDTYPE, MemorySpace::Host>* matrixTransform
        = nullptr,
        const bool update_matrices = true);

    ExtendedGridOrbitals& operator-=(const ExtendedGridOrbitals& orbitals)
    {
        block_vector_ -= orbitals.block_vector_;
        return *this;
    }

    void initGauss(const double, const std::shared_ptr<LocalizationRegions>);

    template <typename CoeffType>
    void axpy(const CoeffType alpha, const ExtendedGridOrbitals&);

    void app_mask(const int, pb::GridFunc<ScalarType>&, const short) const {};

    void applyMask(const bool = false) {};
    void applyCorrMask(const bool = false) {};

#ifdef HAVE_MAGMA
    void multiplyByMatrix(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Device>& matrix);
#endif
    void multiplyByMatrix(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix);

    void multiplyByMatrix(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& matrix,
        ExtendedGridOrbitals& product) const;
    void multiply_by_matrix(
        const DISTMATDTYPE* const matrix, ExtendedGridOrbitals& product) const;
    template <class MatrixType>
    void multiply_by_matrix(const MatrixType&, const double alpha,
        ExtendedGridOrbitals<ORBDTYPE>& product);
    void multiplyByMatrix2states(const int st1, const int st2,
        const double* mat, ExtendedGridOrbitals& product);

    int write(HDFrestart&, const std::string& name = "Function");
    int read_hdf5(HDFrestart& h5f_file);
    int read_func_hdf5(HDFrestart&, const std::string& name = "Function");

    void initWF(const std::shared_ptr<LocalizationRegions> lrs);
    void checkCond(const double tol, const bool flag_stop);
    double normState(const int st) const;
    const std::vector<std::vector<int>>& getOverlappingGids() const
    {
        assert(overlapping_gids_.size() > 0);
        return overlapping_gids_;
    }
    int getGlobalIndex(const short iloc, const short color) const
    {
        assert(overlapping_gids_.size() > 0);
        assert(iloc < static_cast<int>(overlapping_gids_.size()));
        assert(color < static_cast<int>(overlapping_gids_[iloc].size()));
        return overlapping_gids_[iloc][color];
    }
    int getColor(const int gid) const { return gid; }
    double getMaxR() const
    {
        Mesh* mymesh           = Mesh::instance();
        const pb::Grid& mygrid = mymesh->grid();
        return mygrid.maxDomainSize();
    }

#ifdef MGMOL_HAS_LIBROM
    void set(std::string file_path, int rdim);
#endif
};

template <typename ScalarType>
int ExtendedGridOrbitals<ScalarType>::lda_ = 0;
template <typename ScalarType>
int ExtendedGridOrbitals<ScalarType>::numpt_ = 0;
template <typename ScalarType>
int ExtendedGridOrbitals<ScalarType>::data_wghosts_index_ = -1;
template <typename ScalarType>
int ExtendedGridOrbitals<ScalarType>::numst_ = -1;
template <typename ScalarType>
std::vector<std::vector<int>>
    ExtendedGridOrbitals<ScalarType>::overlapping_gids_;

#endif
