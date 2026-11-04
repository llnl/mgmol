// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_PROJECTED_MATRICES_SPARSE_H
#define MGMOL_PROJECTED_MATRICES_SPARSE_H

#include "ProjectedMatricesInterface.h"

#include <string.h>

#include "ClusterOrbitals.h"
#include "DensityMatrixSparse.h"
#include "LinearSolver.h"
#include "LocalizationRegions.h"
#include "ShortSightedInverse.h"
#include "SquareLocalMatrices.h"
#include "Timer.h"
#include "VariableSizeMatrix.h"

#include "hdf5.h"
#include <iostream>

// static const int MAX_MAT_SIZE = 10000;
// const double tol_matrix_elements=1.e-14;
// const double tol_matrix_elements=0.;

class HDFrestart;

// struct to hold a double and the associated rank
struct typeDoubleInt
{
    double val;
    int rank;
};

class ProjectedMatricesSparse : public ProjectedMatricesInterface
{
    bool isDataSetup_;

    double min_val_;

    int index(const int phi_index, const int pot_index) const
    {
        return (phi_index % 100) + 100 * pot_index;
    }

    static Timer compute_inverse_tm_;
    static Timer compute_invB_tm_;
    static Timer init_gram_matrix_tm_;
    static Timer update_theta_tm_;
    static Timer update_submatT_tm_;
    static Timer update_submatX_tm_;
    static Timer eigsum_tm_;
    static Timer consolidate_H_tm_;
    static Timer eig_interval_tm_;

    ShortSightedInverse* invS_;
    VariableSizeMatrix<sparserow>* submatT_;
    VariableSizeMatrix<sparserow>* matHB_;

    // local data structure to hold partial elements of H
    VariableSizeMatrix<sparserow>* sH_;

    DensityMatrixSparse* dm_;

    SquareLocalMatrices<MATDTYPE, memory_space_type>* localX_;
    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>* localT_;

    /* Data distribution objects */
    DataDistribution* distributor_invS_;

    void clearData();

    int dim_;

    /* local data size */
    int lsize_;

    std::shared_ptr<LocalizationRegions> lrs_;
    ClusterOrbitals* local_cluster_;

    // indexes corresponding to valid function in each subdomain
    std::vector<std::vector<int>> global_indexes_;

    std::vector<int> locvars_;

    void updateLocalMat(const VariableSizeMatrix<sparserow>& submatM,
        SquareLocalMatrices<MATDTYPE, MemorySpace::Host>* localM);
    double getExpectationMat(VariableSizeMatrix<sparserow>* mat);
    VariableSizeMatrix<sparserowtab>* getGramMat() { return invS_->gramMat(); }
    void computeGenEigenInterval(std::vector<double>& interval,
        const int maxits, const double padding = 0.01);

    double eigenvalue0_;

public:
    ProjectedMatricesSparse(const int ndim, const double width,
        std::shared_ptr<LocalizationRegions> lrs,
        ClusterOrbitals* local_cluster = nullptr);
    ~ProjectedMatricesSparse() override;

    void setup(const std::vector<std::vector<int>>& global_indexes) override;

    void updateSubMatT() override;
    void updateTheta() override;
    double getExpectationH() override;
    void consolidateH() override;
    void consolidateOrbitalsOverlapMat(VariableSizeMatrix<sparserow>& mat);
    double dotProductWithInvS(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss) override;
    double dotProductSimple(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss) override;
    double dotProductWithDM(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss) override;
    void applyInvS(
        SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& mat) override;
    double getEigSum() override;
    double getLinDependent2states(
        int& st1, int& st2, const bool print_flag = false) const override;
    void printMatrices(std::ostream& os) const override;
    void printTimers(std::ostream& os) override;
    double getNel() const override;
    double getLowestEigenvalue() const override
    {
        assert(eigenvalue0_ == eigenvalue0_);
        return eigenvalue0_;
    }
    void printGramMatrix2states(
        const int st1, const int st2, std::ostream& os) const;

    void initializeGramMatrix(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss,
        const int orbitals_index) override
    {
        assert(invS_ != NULL);
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        mmpi.barrier();
        init_gram_matrix_tm_.start();
        (*invS_).initGramMatrix(ss, global_indexes_, orbitals_index);

        init_gram_matrix_tm_.stop();
    }

    SquareLocalMatrices<MATDTYPE, memory_space_type>& getLocalX() const override
    {
        return *localX_;
    }

    SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& getLocalT() const override
    {
        return *localT_;
    }

    void updateSubMatX() override
    {
        assert(dm_ != NULL);
        update_submatX_tm_.start();
        (*dm_).getLocalMatrix((*localX_), global_indexes_);
        update_submatX_tm_.stop();
    }

    void updateHB()
    {
        /* scale H */
        //    (*matHB_).scale(vel_);
    }
    void setDMuniform(const double nel) override { dm_->setUniform(nel); }
    void setDMaufbau(const double nel) override { dm_->setAufbau(nel); }

    int getDMMatrixIndex() const override
    {
        assert(dm_ != nullptr);
        return dm_->getIndex();
    }
    int getGramMatrixIndex() const
    {
        assert(invS_ != nullptr);
        return invS_->getGramMatrixOrbitalsIndex();
    }
    double computeEntropy() override { return 0.; }

    // This call assumes sH has not been consolidated and its row ordering on
    // the local subdomain (locvars_) has been defined by calling
    // setupSparseRows(locvars_). NOTE: We restrict only to contributions
    // computed/ belonging in the local subdomain

    // Add data from square local matrix (only contributions from functions
    // overlapping subdomain)
    void setLocalMatrixElementsHl(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& slH) override
    {
        // std::cout<<"Set Hl elements with matrix of size
        // "<<slH.m()<<std::endl;
        Control& ct = *(Control::instance());
        sH_->insertMatrixElements(
            slH, global_indexes_, ct.numst, tol_matrix_elements);
    }

    void setLocalMatrixElementsHnl(
        const SquareSubMatrix<MATDTYPE>& slH) override
    {
        // std::cout<<"Set Hnl elements with matrix of size "
        //         <<slH.getGids().size()<<std::endl;
        sH_->insertMatrixElements(slH, tol_matrix_elements);
    }

    void clearSparseH() override
    {
        (*sH_).reset();
        // initialize sH_ with locvars_.size() empty rows
        (*sH_).setupSparseRows(locvars_);
    }

    /* print the Theta matrix -- for diagnostics */
    void printTheta(std::ostream& os) const
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.instancePE0())
            os << " Matrix Theta ... n = " << (*submatT_).n() << std::endl;

        std::vector<int> locfcns;
        (*lrs_).getLocalSubdomainIndices(locfcns);
        (*submatT_).print(os, locfcns);
    }

    /* print the HB matrix -- for diagnostics */
    void printHB(std::ostream& os) const
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.instancePE0())
            os << " Matrix HB ... n = " << (*matHB_).n() << std::endl;

        std::vector<int> locfcns;
        (*lrs_).getLocalSubdomainIndices(locfcns);
        (*matHB_).print(os, locfcns);
    }

    /* print the inverse Gram matrix -- for diagnostics */
    void printDM(std::ostream& os) const override
    {
        assert(dm_ != nullptr);
        dm_->printDM(os);
    }

    void setDMto2InvS() override
    {
        assert(invS_ != nullptr);
        dm_->setto2InvS(invS_->getInvS());
    }

    void computeInvS() override
    {
        assert(invS_ != NULL);
        assert(distributor_invS_ != NULL);

        compute_inverse_tm_.start();
        (*invS_).computeInvS(*distributor_invS_);

        //   double rcond = checkCond(1000.);
        //   if(onpe0)cout<<"RCOND = "<<rcond<<endl;
        compute_inverse_tm_.stop();
    }
    void updateThetaAndHB() override
    {
        updateTheta();
        updateHB();
    }

    void printGramMM(std::ofstream& tfile) override
    {
        assert(invS_ != NULL);
        (*invS_).printGramMM(tfile);
    }

    /* print (part of) the (global) Gram matrix -- for diagnostics */
    void printS(std::ostream& os) const override
    {
        assert(invS_ != NULL);
        (*invS_).printGramMat(os);
    }

    /* print the H matrix -- for diagnostics */
    void printH(std::ostream& os) const
    {
        /* matrix sH has local contributions, so we need to
         * gather neighboring data first.
         */
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.instancePE0()) os << " Matrix H " << std::endl;

        std::vector<int> locfcns;
        (*lrs_).getLocalSubdomainIndices(locfcns);
        (*sH_).print(os, locfcns);
    }

    void resetDotProductMatrices() override { assert((*invS_).gramMat() != 0); }

    void computeInvB() override
    {
        assert(invS_ != NULL);
        compute_invB_tm_.start();
        (*invS_).computeInvS(*distributor_invS_);
        compute_invB_tm_.stop();
    }

    double computeCond() override
    {
        assert(invS_ != NULL);

        return (*invS_).computeGramMatCond();
    }

    double checkCond(const double tol, const bool flag = true) override
    {
        double rcond = computeCond();

        if (rcond > tol)
        {
            // ofstream tfile("s.mm", ios::out);;
            // gm_->printMM(tfile);
            // tfile.close();
            MGmol_MPI& mgmolmpi = *(MGmol_MPI::instance());
            mgmolmpi.barrier();
            if (mgmolmpi.instancePE0())
                (*MPIdata::sout)
                    << " CONDITION NUMBER OF THE OVERLAP MATRIX EXCEEDS TOL: "
                    << rcond << "!!!" << std::endl;
            if (flag) mgmolmpi.abort();
        }
        return rcond;
    }

    double getTraceDiagProductWithInvS(std::vector<double>& ddiag) override
    {
        assert(invS_ != NULL);
        return (*invS_).getTraceDiagProductWithInvS(ddiag);
    }

    bool precondIsNew() const { return invS_->precondIsNew(); }

    bool isGramMatAugmented() { return invS_->isGramAugmented(); }

    double computeTraceInvSmultMat(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& mat,
        const bool consolidate);
    double computeTraceDMmultMat(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& mat,
        const bool consolidate);
    double computeTraceInvSmultMat(
        VariableSizeMatrix<sparserow>& vsmat, const bool consolidate);
    void stripDM() override { return; };
    void dressupDM() override { return; };

    const VariableSizeMatrix<sparserow>& getH() { return *matHB_; }
    void assignH(const VariableSizeMatrix<sparserow>& matH) { *matHB_ = matH; }
    void printEigenvalues(std::ostream& /*os*/) const { }
    void printOccupations(std::ostream& /*os*/) const override { }
    void setHB2H() override { }

    DensityMatrixSparse& getDM() { return *dm_; }
};

#endif
