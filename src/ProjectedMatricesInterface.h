// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_PROJECTED_MATRICES_INTERFACE_H
#define MGMOL_PROJECTED_MATRICES_INTERFACE_H

#include <cassert>
#include <iostream>
#include <vector>

/* Currently only needed functions in ProjectedMatrices.h class. Must be
 * cleaned-up later */
#include "ChebyshevApproximationFunction.h"
#include "Control.h"
#include "MGmol_MPI.h"
#include "Orbitals.h"
#include "SquareLocalMatrices.h"
#include "SquareSubMatrix.h"
#include "entropy.h"
#include "fermi.h"
#include "hdf5.h"
#include "tools.h"

class HDFrestart;

using memory_space_type = typename Orbitals::memory_space_type;

/* tolerance for matrix elements */
const double tol_matrix_elements = 1.e-14;

class ProjectedMatricesInterface : public ChebyshevApproximationFunction
{
private:
    int iterative_index_h_;

    int index(const int phi_index, const int pot_index) const
    {
        return (phi_index % 100) + 100 * pot_index;
    }

protected:
    bool with_spin_;
    double width_;

    double nel_;

    short subdiv_;
    short chromatic_number_;

    double mu_;

    // pointer to member function
    std::vector<double> (ProjectedMatricesInterface::*funcptr_)(
        const std::vector<double>& nodes);

public:
    ProjectedMatricesInterface(const bool with_spin, const double width)
        : ChebyshevApproximationFunction(),
          iterative_index_h_(-1),
          with_spin_(with_spin),
          width_(width),
          subdiv_(-1),
          chromatic_number_(-1)
    {
        Control& ct     = *(Control::instance());
        nel_            = ct.getNelSpin();
        MGmol_MPI& mmpi = *(MGmol_MPI::instance());
        if (mmpi.PE0() && ct.verbose > 1)
            std::cout << "ProjectedMatricesInterface: nel_=" << nel_
                      << std::endl;
    };

    // define Fermi distribution function to be approximated by Chebyshev
    // approximation f(x) = 1 / (1 + Exp[(E-mu)/kbT])
    std::vector<double> chebfunDM(const std::vector<double>& nodes)
    {
        assert(static_cast<int>(nodes.size()) > 0);

        const int n = static_cast<int>(nodes.size());
        std::vector<double> fvals(n, 0.);
        // compute fermi distribution function
        fermi_distribution(mu_, n, width_, nodes, fvals);

        return fvals;
    }

    // define entropy function to be approximated by Chebyshev approximation
    // s(f) = -f ln(f) - (1-f) ln(1-f). Assumes primary variables are the
    // occupations
    std::vector<double> chebfunEntropyFromOcc(const std::vector<double>& nodes)
    {
        assert(static_cast<int>(nodes.size()) > 0);

        const int n = static_cast<int>(nodes.size());
        std::vector<double> fvals(n, 0.);
        // compute entropy function
        MGmol_MPI& mmpi           = *(MGmol_MPI::instance());
        double orbital_occupation = mmpi.nspin() > 1 ? 1. : 2.;

        // evaluate entropy function assuming primary variables given are the
        // occupations
        entropy_eval(nodes, fvals, orbital_occupation);

        return fvals;
    }

    // define entropy function to be approximated by Chebyshev approximation
    // s(f) = -f ln(f) - (1-f) ln(1-f). Assumes primary variables are the
    // energies
    std::vector<double> chebfunEntropyFromEnergies(
        const std::vector<double>& nodes)
    {
        assert(static_cast<int>(nodes.size()) > 0);

        const int n = static_cast<int>(nodes.size());
        std::vector<double> fvals(n, 0.);
        // compute entropy function
        MGmol_MPI& mmpi           = *(MGmol_MPI::instance());
        double orbital_occupation = mmpi.nspin() > 1 ? 1. : 2.;

        // evaluate the entropy function assuming the primary variables given
        // are the energies or eigenvalues of H. function-of-a-function approach
        // for computing entropy.
        entropy_evalFromEnergies(
            mu_, n, width_, nodes, fvals, orbital_occupation);

        return fvals;
    }

    std::vector<double> eval(const std::vector<double>& nodes) override
    {
        return (this->*(funcptr_))(nodes);
    }

    virtual ~ProjectedMatricesInterface() {};

    virtual void setup(const std::vector<std::vector<int>>& global_indexes) = 0;

    // initial setup function
    void setupBase(const int subdiv, const int chromatic_number)
    {
        subdiv_           = subdiv;
        chromatic_number_ = chromatic_number;
    }

    void setHiterativeIndex(const int phi_index, const int pot_index)
    {
        assert(phi_index < 100000);
        iterative_index_h_ = index(phi_index, pot_index);
    }
    bool isHupToDate(const int phi_index, const int pot_index)
    {
        assert(phi_index < 100000);
        return (iterative_index_h_ == index(phi_index, pot_index));
    }
    virtual void printTimers(std::ostream& os) = 0;
    // initialize Gram matrix with SquareLocalMatrices (input)
    virtual void initializeGramMatrix(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss,
        const int orbitals_index)
        = 0;

    virtual void setLocalMatrixElementsHl(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& slH)
        = 0;
    virtual void setLocalMatrixElementsHnl(const SquareSubMatrix<MATDTYPE>& slH)
        = 0;

    virtual void clearSparseH() = 0;
    virtual void consolidateH() = 0;

    // return density matrix (inverse Gram if no unoccupied states)
    virtual SquareLocalMatrices<MATDTYPE, memory_space_type>& getLocalX() const
        = 0;

    // return S**(-1)*H (or B**(-1)*H with Mehrstellen)
    virtual SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& getLocalT() const
        = 0;

    // returns local Gram Matrix
    //    virtual SquareLocalmatrices& getLocalS()const=0;

    // return Tr(X*H) or Tr(S**(-1)*H) if occupied states only
    virtual double getExpectationH() = 0;

    virtual void updateSubMatX() = 0;

    virtual void updateSubMatT() = 0;

    // return sum_i ( ddiag[i]*S**(-1)[i][i] )
    virtual double getTraceDiagProductWithInvS(std::vector<double>& ddiag) = 0;

    virtual double computeEntropy() = 0;
    virtual double getEigSum()      = 0;

    virtual void updateTheta()                     = 0;
    virtual void computeInvB()                     = 0;
    virtual void printGramMM(std::ofstream& tfile) = 0;

    virtual void setDMuniform(const double nel) = 0;
    virtual void setDMaufbau(const double nel)  = 0;

    virtual double dotProductWithInvS(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
        = 0;
    virtual double dotProductWithDM(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
        = 0;
    virtual double dotProductSimple(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
        = 0;
    virtual void resetDotProductMatrices() = 0;

    virtual void stripDM()                             = 0;
    virtual void dressupDM()                           = 0;
    virtual void printS(std::ostream& os) const        = 0;
    virtual void computeInvS()                         = 0;
    virtual void printMatrices(std::ostream& os) const = 0;
    virtual void applyInvS(
        SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& mat)
        = 0;
    virtual double getLinDependent2states(int& st1, int& st2, const bool) const
        = 0;
    virtual double checkCond(const double tol, const bool flag = true) = 0;
    virtual double getNel() const                                      = 0;
    virtual void updateThetaAndHB()                                    = 0;
    virtual void printDM(std::ostream& os) const                       = 0;

    //////////////////////////////////////////////////////////////////////
    /* Default implementation - returns error message */
    virtual void printOccupations(std::ostream& os) const
    {
        (void)os;

        exitWithErrorMessage("printOccupations");
    }
    virtual int getDMMatrixIndex() const
    {
        exitWithErrorMessage("getDMMatrixIndex");

        return 0;
    }
    virtual double computeCond() = 0;
    virtual void getOccupations(std::vector<double>& occ) const
    {
        (void)occ;

        exitWithErrorMessage("getOccupations");
    }
    virtual void setHB2H() { exitWithErrorMessage("setHB2H"); }
    virtual void setAuxilliaryEnergiesFromOccupations()
    {
        exitWithErrorMessage("setAuxilliaryEnergiesFromOccupations");
    }
    virtual void saveDM() { exitWithErrorMessage("saveDM"); }
    virtual void resetDM() { exitWithErrorMessage("resetDM"); }
    virtual void updateDMwithRelax(const double mix)
    {
        (void)mix;

        exitWithErrorMessage("updateDMwithRelax");
    }
    virtual int readDM(HDFrestart& h5f_file)
    {
        (void)h5f_file;

        exitWithErrorMessage("readDM");

        return 0;
    }
    virtual int readWFDM(HDFrestart& h5f_file)
    {
        (void)h5f_file;

        exitWithErrorMessage("readWFDM");

        return 0;
    }
    virtual int writeDM(HDFrestart& h5f_file)
    {
        (void)h5f_file;

        exitWithErrorMessage("writeDM");

        return 0;
    }
    virtual int writeSavedDM(HDFrestart& h5f_file)
    {
        (void)h5f_file;

        exitWithErrorMessage("writeSavedDM");

        return 0;
    }
    virtual void updateDMwithChebApproximation()
    {
        exitWithErrorMessage("updateDMwithChebApproximation");
    }
    virtual void updateDM() { exitWithErrorMessage("updateDM"); }

    virtual void setDMto2InvS() { exitWithErrorMessage("setDMto2InvS"); }
    virtual void initializeMatB(
        const SquareLocalMatrices<MATDTYPE, MemorySpace::Host>& ss)
    {
        (void)ss;

        exitWithErrorMessage("initializeMatB");
    }
    virtual double getLowestEigenvalue() const = 0;
};

#endif
