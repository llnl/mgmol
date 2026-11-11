// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_included_Poisson
#define MGMOL_included_Poisson

#include "PoissonInterface.h"

#include "GridFunc.h"
#include "MPIdata.h"
#include "Rho.h"
#include "Timer.h"

#include <vector>

// template <typename T>
class Poisson : public PoissonInterface
{
protected:
    const pb::Grid& grid_;

    pb::GridFunc<POTDTYPE>* vh_;
    pb::GridFunc<POTDTYPE>* vepsilon_;

    double Int_vhrho_;
    double Int_vhrhoc_;
    double Int_vhrho_old_;

    short bc_[3];

public:
    // Constructor
    Poisson(const pb::Grid& grid, const short bc[3]) : grid_(grid)
    {
        bc_[0] = bc[0];
        bc_[1] = bc[1];
        bc_[2] = bc[2];

        vh_       = new pb::GridFunc<POTDTYPE>(grid_, bc[0], bc[1], bc[2]);
        vepsilon_ = nullptr;
    };

    // Destructor
    virtual ~Poisson() override { delete vh_; };

    virtual void setup(const short nu1, const short nu2, const short max_sweeps,
        const double tol, const short max_nlevels,
        const bool gather_coarse_level = true)
        = 0;

    void getVh(POTDTYPE* vh) { vh_->init_vect(vh, 'd'); }
    void getVepsilon(POTDTYPE* veps) { vepsilon_->init_vect(veps, 'd'); }
    const pb::GridFunc<POTDTYPE>& vh() const
    {
        assert(vh_ != NULL);
        return *vh_;
    }

    double IntVhRho(void) const { return Int_vhrho_; }
    double IntVhRhoc(void) const { return Int_vhrhoc_; }
    double IntVhRho_old(void) const { return Int_vhrho_old_; }

    virtual void set_rhod(pb::GridFunc<RHODTYPE>* /*rhod*/) {};
    void set_vh(const pb::GridFunc<POTDTYPE>& vh) { (*vh_) = vh; };
    void set_vh(const std::vector<POTDTYPE>& vh)
    {
        vh_->assign(vh.data(), 'd');
    };
    void resetVh() { vh_->resetData(); }
    void set_vepsilon(const POTDTYPE* const vepsilon)
    {
        vepsilon_->assign(vepsilon, 'd');
    };

    virtual void solve(
        const pb::GridFunc<RHODTYPE>& rho, const pb::GridFunc<RHODTYPE>& rhoc)
        = 0;
    void computeVhRho(
        const pb::GridFunc<RHODTYPE>& rho, const pb::GridFunc<RHODTYPE>& rhoc)
    {
        const double vel = grid_.vel();
        Int_vhrho_       = vel * vh_->gdot(rho);
        Int_vhrhoc_      = vel * vh_->gdot(rhoc);
    }

    virtual void applyOperator(
        pb::GridFunc<POTDTYPE>& vh, pb::GridFunc<POTDTYPE>& lhs)
    {
        std::cerr << "ERROR: Abstract method Poisson::applyOperator()"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
};

#endif
