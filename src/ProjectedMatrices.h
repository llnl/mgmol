// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_PROJECTED_MATRICES_H
#define MGMOL_PROJECTED_MATRICES_H

#include "ChebyshevApproximation.h"
#include "DensityMatrix.h"
#include "GramMatrix.h"
#include "ProjectedMatricesInterface.h"
#include "SquareLocalMatrices.h"
#include "SquareSubMatrix.h"
#include "Timer.h"

#include "hdf5.h"
#include "tools.h"
#include <iostream>

template <class MatrixType>
class ProjectedMatrices : public ProjectedMatricesInterface
{
    static short n_instances_;

    static GramMatrix<MatrixType>* gram_4dotProducts_;

    // spin: 0 for ignoring spin, 1 for calculation with spin
    const bool with_spin_;

    std::vector<double> eigenvalues_;

    /*!
     * matrices to save old values and enable mixing or reset
     */
    std::unique_ptr<MatrixType> mat_X_old_;
    std::unique_ptr<MatrixType> mat_L_old_;

    static Timer sygv_tm_;
    static Timer compute_inverse_tm_;
    static Timer compute_invB_tm_;
    static Timer init_gram_matrix_tm_;
    static Timer update_theta_tm_;
    static Timer update_submatT_tm_;
    static Timer update_submatX_tm_;
    static Timer eigsum_tm_;
    static Timer consolidate_H_tm_;
    static Timer compute_entropy_tm_;

    ProjectedMatrices& operator=(const ProjectedMatrices& src);
    ProjectedMatrices(const ProjectedMatrices& pm);

    /*!
     * Matrices to be used to multiply orbitals on the right
     * (dimension equal to number of local colors)
     */
    std::unique_ptr<SquareLocalMatrices<double, memory_space_type>>
        localX_; // density matrix
    std::unique_ptr<SquareLocalMatrices<double, MemorySpace::Host>>
        localT_; // theta=inv(S)*H_phi

    // internal data structures to hold local contributions
    // to matrix H
    std::unique_ptr<SquareLocalMatrices<double, MemorySpace::Host>> localHl_;
    std::unique_ptr<SquareSubMatrix<double>> localHnl_;

    void printEigenvaluesHa(std::ostream& os) const;
    void printEigenvaluesEV(std::ostream& os) const;

    void computeChemicalPotentialAndOccupations(
        const std::vector<double>& energies, const double width,
        const int max_numst);

    void computeChemicalPotentialAndOccupations()
    {
        computeChemicalPotentialAndOccupations(width_, dim_);
    }

    // fill orbitals with prescribed values read in config file
    void fillOrbitals();

    double computeChemicalPotentialAndDMwithChebyshev(
        const int order, const double emin, const double emax);

protected:
    // indexes corresponding to valid function in each subdomain
    std::vector<std::vector<int>> global_indexes_;

    unsigned dim_;

    std::unique_ptr<MatrixType> theta_;
    std::unique_ptr<MatrixType> matHB_;
    std::unique_ptr<MatrixType> matH_;

    /*!
     * Density Matrix
     */
    std::unique_ptr<DensityMatrix<MatrixType>> dm_;

    /*!
     * Matrix of eigenvectors used to build DM
     * (if used)
     */
    std::unique_ptr<MatrixType> ev_;

    /*!
     * Gram matrix of orbitals overlaps
     */
    std::unique_ptr<GramMatrix<MatrixType>> gm_;

    /*!
     * Dipole Matrix
     */
    std::unique_ptr<MatrixType> matDx_;
    std::unique_ptr<MatrixType> matDy_;
    std::unique_ptr<MatrixType> matDz_;

    // work matrix for tmp usage
    std::unique_ptr<MatrixType> work_;

    std::vector<double> cheb_interval_;
    void printTheta(std::ostream& os) const
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.instancePE0()) os << " Matrix Theta" << std::endl;
        theta_->print(os, 0, 0, NPRINT_ROWS_AND_COLS, NPRINT_ROWS_AND_COLS);
    }

    void printHB(std::ostream& os) const
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.instancePE0()) os << " Matrix HB" << std::endl;
        matHB_->print(os, 0, 0, NPRINT_ROWS_AND_COLS, NPRINT_ROWS_AND_COLS);
    }

    void printH(std::ostream& os) const
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.instancePE0()) os << " Matrix H" << std::endl;
        matH_->print(os, 0, 0, NPRINT_ROWS_AND_COLS, NPRINT_ROWS_AND_COLS);
    }

    // convert a SquareLocalMatrices object into a MatrixType
    void convert(const SquareLocalMatrices<double, MemorySpace::Host>& src,
        MatrixType& dst);

    void setupGlobalIndexes(const std::vector<std::vector<int>>&);

    std::string getMatrixType();

public:
    ProjectedMatrices(const int, const bool with_spin, const double occ_width);
    ~ProjectedMatrices() override;

    void setup(const std::vector<std::vector<int>>& global_indexes) override;

    void setLocalMatrixElementsHnl(const SquareSubMatrix<double>& slH) override
    {
        localHnl_.reset(new SquareSubMatrix<double>(slH));
    }
    void setLocalMatrixElementsHl(
        const SquareLocalMatrices<double, MemorySpace::Host>& slH) override
    {
        localHl_->copy(slH);
    }
    void clearSparseH() override { }

    void consolidateH() override;

    double getTraceDiagProductWithInvS(std::vector<double>& ddiag) override;

    SquareLocalMatrices<double, memory_space_type>& getLocalX() const override
    {
        return *localX_;
    }

    SquareLocalMatrices<double, MemorySpace::Host>& getLocalT() const override
    {
        return *localT_;
    }

    void updateSubMatX() override { updateSubMatX(dm_->getMatrix()); }

    void updateSubMatX(const MatrixType& dm);

    void updateSubMatT() override;

    void computeLoewdinTransform(
        SquareLocalMatrices<double, MemorySpace::Host>& localPi,
        const int orb_index, const bool transform_matrices);

    void printTimers(std::ostream& os) override;

    void initializeGramMatrix(
        const SquareLocalMatrices<double, MemorySpace::Host>& ss,
        const int orbitals_index) override
    {
        assert(gm_);

        init_gram_matrix_tm_.start();

        convert(ss, *work_);

        gm_->setMatrix(*work_, orbitals_index);
        init_gram_matrix_tm_.stop();
    }
    void printGramMM(std::ofstream& tfile) override { gm_->printMM(tfile); }
    void printHamiltonianMM(std::ofstream& tfile) { matH_->printMM(tfile); }
    int getDMMatrixIndex() const override
    {
        assert(dm_);
        return dm_->getIndex();
    }

    /*
     * Initialize DM by setting diagonal
     */
    void setDMuniform(const double nel) override { dm_->setUniform(nel); }
    void setDMaufbau(const double nel) override { dm_->setAufbau(nel); }

    int dim() const { return dim_; }

    void computeInvS() override;

    void computeInvB() override
    {
        compute_invB_tm_.start();
        computeInvS();
        compute_invB_tm_.stop();
    }

    const MatrixType& dm() const;

    bool occupationsUptodate() const { return dm_->occupationsUptodate(); }

    const MatrixType& getLS() const { return gm_->getCholeskyL(); }

    void updateLS() { gm_->updateLS(); }

    void updateTheta() override
    {
        // theta = invB * Hij
        update_theta_tm_.start();
        theta_->symm('l', 'l', 1., gm_->getInverse(), *matH_, 0.);

        update_theta_tm_.stop();
    }

    void updateThetaAndHB() override
    {
        updateTheta();

        updateHB();
    }

    void assignH(const MatrixType& matH) { *matH_ = matH; }

    const MatrixType& getH() { return *matH_; }
    const MatrixType& getEigenvectors()
    {
        assert(ev_ != nullptr);
        return *ev_;
    }

    void setHB2H() override
    {
        // if( onpe0 )
        //    (*MPIdata::sout)<<"ProjectedMatrices::setHB2H()..."<<endl;
        *matHB_ = *matH_;
    }

    virtual void updateHB() { setHB2H(); }

    void setGram2Id(const int orbitals_index) { gm_->set2Id(orbitals_index); }

    void setGramMatrix(const MatrixType& mat, const int orbitals_index)
    {
        assert(gm_ != 0);
        gm_->setMatrix(mat, orbitals_index);
        gm_->computeInverse();
    }

    const MatrixType& getGramMatrix()
    {
        assert(gm_ != nullptr);
        return gm_->getMatrix();
    }

    void getOccupations(std::vector<double>& occ) const override;
    void setOccupations(const std::vector<double>& occ);

    void stripDM() override;
    void dressupDM() override;

    void applyInvS(
        SquareLocalMatrices<double, MemorySpace::Host>& mat) override;
    void applyInvS(MatrixType& mat)
    {
        assert(gm_ != nullptr);
        gm_->applyInv(mat);
    }

    void setDMto2InvS() override;
    void buildDM(const MatrixType& z);
    void buildDM(const MatrixType& z, const std::vector<double>&);
    void buildDM(const std::vector<double>&);

    double getEigSum() override;
    double getExpectation(const MatrixType& A);
    double getExpectationH() override;

    void solveGenEigenProblem(MatrixType& zz, char job = 'v');
    void computeOccupationsFromDM();

    virtual void rotateAll(
        const MatrixType& rotation_matrix, const bool flag_eigen);

    double getNel() const override;

    double getLowestEigenvalue() const override { return eigenvalues_[0]; }

    void printS(std::ostream& os) const override { gm_->print(os); }

    void printMatrices(std::ostream& os) const override
    {
        printS(os);
        printH(os);
        printTheta(os);
        printHB(os);
    }

    void printDM(std::ostream& os) const override;
    void printOccupations(std::ostream& os) const override;

    double computeCond() override { return gm_->computeCond(); }

    double computeEntropy(const double kbt);
    double computeEntropy() override;
    double computeEntropyWithCheb(const double kbt);
    double checkCond(const double tol, const bool flag = true) override;
    int writeDM(HDFrestart& h5f_file) override;
    int writeSavedDM(HDFrestart& h5f_file) override;
    int readDM(HDFrestart& h5f_file) override;
    int readWFDM(HDFrestart& h5f_file) override;
    void printEigenvalues(std::ostream& os) const;
    void updateDM() override;
    void updateDMwithEigenstates();
    void updateDMwithSP2();
    void updateDMwithEigenstatesAndRotate(MatrixType& zz);
    void updateDMwithChebApproximation() override;
    void computeChemicalPotentialAndOccupations(
        const double width, const int max_numst)
    {
        computeChemicalPotentialAndOccupations(eigenvalues_, width, max_numst);
    }

    void saveDM() override
    {
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.instancePE0())
            std::cout << "ProjectedMatrices::saveDM()" << std::endl;
        if (!mat_X_old_)
        {
            mat_X_old_.reset(new MatrixType(dm_->getMatrix()));
            mat_L_old_.reset(new MatrixType(gm_->getCholeskyL()));
        }
        else
        {
            *mat_X_old_ = dm_->getMatrix();
            *mat_L_old_ = gm_->getCholeskyL();
        }
    }

    void resetDM() override
    {
        dm_->setMatrix(*mat_X_old_);
        dm_->stripS(*mat_L_old_);
    }

    void updateDMwithRelax(const double mix) override
    {
        // cout<<"ProjectedMatrices::updateDMwithRelax()..."<<endl;
        assert(mat_X_old_);

        dm_->mix(mix, *mat_X_old_);
    }

    SquareLocalMatrices<double, MemorySpace::Host> getReplicatedDM();

    double getLinDependent2states(
        int& st1, int& st2, const bool /*flag*/) const override
    {
        return gm_->getLinDependent2states(st1, st2);
    }

    virtual void initializeMatB(
        const SquareLocalMatrices<double, MemorySpace::Host>& ss) override
    {
        (void)ss;
        std::cerr
            << "ERROR: ProjectedMatrices::initializeMatB() should never be "
               "called!!!"
            << std::endl;
    }

    void resetDotProductMatrices() override;
    double dotProductWithInvS(
        const SquareLocalMatrices<double, MemorySpace::Host>& ss) override;
    double dotProductWithDM(
        const SquareLocalMatrices<double, MemorySpace::Host>& ss) override;
    double dotProductSimple(
        const SquareLocalMatrices<double, MemorySpace::Host>& ss) override;
    double computeTraceInvSmultMat(
        const SquareLocalMatrices<double, MemorySpace::Host>& mat);
    double computeTraceInvSmultMat(const MatrixType& mat)
    {
        MatrixType pmatrix(mat);
        gm_->applyInv(pmatrix);

        return pmatrix.trace();
    }
    double computeTraceInvSmultMatMultTheta(const MatrixType& mat);
    double computeTraceMatMultTheta(const MatrixType& mat)
    {
        assert(theta_ != 0);
        MatrixType pmatrix("pmatrix", dim_, dim_);
        pmatrix.gemm('n', 'n', 1.0, mat, *theta_, 0.);
        return pmatrix.trace();
    }
    void computeMatMultTheta(const MatrixType& mat, MatrixType& pmat)
    {
        assert(theta_ != 0);
        pmat.gemm('n', 'n', 1.0, mat, *theta_, 0.);
    }
    MatrixType& getMatHB() { return *matHB_; }
    void setDM(const MatrixType& mat) { dm_->setMatrix(mat); }
    void setEigenvalues(const std::vector<double>& eigenvalues)
    {
        memcpy(eigenvalues_.data(), eigenvalues.data(),
            eigenvalues.size() * sizeof(double));
    }
    const std::vector<double>& getEigenvalues() const { return eigenvalues_; }
    void computeGenEigenInterval(std::vector<double>& interval,
        const int maxits, const double padding = 0.01);
    DensityMatrix<MatrixType>& getDM() { return *dm_; }
};

template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::init_gram_matrix_tm_(
    "ProjectedMatrices::initialize_Gram_Matrix");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::sygv_tm_("ProjectedMatrices::sygv");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::compute_inverse_tm_(
    "ProjectedMatrices::computeInverse");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::compute_invB_tm_(
    "ProjectedMatrices::computeInvB");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::update_theta_tm_(
    "ProjectedMatrices::updateTheta");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::update_submatT_tm_(
    "ProjectedMatrices::updateSubmatT");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::update_submatX_tm_(
    "ProjectedMatrices::updateSubmatX");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::eigsum_tm_("ProjectedMatrices::eigsum");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::consolidate_H_tm_(
    "ProjectedMatrices::consolidate_sH");
template <class MatrixType>
Timer ProjectedMatrices<MatrixType>::compute_entropy_tm_(
    "ProjectedMatrices::compute_entropy");
#endif
