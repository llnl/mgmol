// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

// $Id: PBh4M.h,v 1.17 2010/01/28 22:56:47 jeanluc Exp $
#ifndef PB_MH4_H
#define PB_MH4_H

#include "MGmol_abort.h"
#include "PB.h"

namespace pb
{
template <class T>
class PBh4M : public PB<T>
{

protected:
    GridFunc<T> pp_;
    GridFunc<T> sqrt_a_;
    GridFunc<T> inv_sqrt_a_;

    double dot_sqrt_a_;

public:
    // constructor
    PBh4M(const Grid& mygrid, DielFunc<T>& myepsilon);
    PBh4M(const Grid& mygrid, const double e0, const double rho0,
        const double drho0)
        : PB<T>(mygrid, e0, rho0, drho0),
          pp_(PB<T>::grid_, 1, 1, 1),
          sqrt_a_(PB<T>::grid_, 1, 1, 1),
          inv_sqrt_a_(PB<T>::grid_, 1, 1, 1)
    {
        PB<T>::initialized_ = false;
    };

    PBh4M& operator=(const PBh4M& /*oper*/)
    {
        mgmol_abort("operator= of PBh4M");
        return *this;
    }
    // copy constructor
    PBh4M(const PBh4M& oper);

    PBh4M(const Grid& mygrid, DielFunc<T>& myepsilon, GridFunc<T>& pp)
        : PB<T>(mygrid, myepsilon),
          pp_(pp, PB<T>::grid_),
          sqrt_a_(mygrid, 1, 1, 1),
          inv_sqrt_a_(mygrid, 1, 1, 1)
    {
        sqrt_a_.copyFrom(PB<T>::epsilon_);
        inv_sqrt_a_.copyFrom(PB<T>::epsilon_);
        for (short i = 0; i < 3; i++)
        {
            assert(pp_.grid().dim(i) > 0);
            assert(pp_.grid().dim(i) < 10000);
        }
        // std::cout<<"Construct PBh4M operator"<<std::endl;
    };

    // construct a coarse grid operator
    PBh4M coarseOp(const Grid&);

    PBh4M replicatedOp(const Grid&);

    void setLowerOrderGrid() override
    {
        FDoper<T>::setFDLowerOrderGrid(minNumberGhosts());
    }

    PBh4M& getLowerOrderOp() { return *this; }

    // A->B
    void apply(GridFunc<T>& A, GridFunc<T>& B) override
    {
        FDoper<T>::del2_4th_Mehr(A, B);
        PB<T>::work1_.prod(pp_, A);
        rhs(PB<T>::work1_, PB<T>::work2_);
        B += PB<T>::work2_;
    }

    void init(GridFunc<T>&);

    void transform(GridFunc<T>& A) const override { A *= inv_sqrt_a_; }

    void inv_transform(GridFunc<T>& A) const override { A /= inv_sqrt_a_; }

    static short minNumberGhosts() { return 2; }

    void rhs(GridFunc<T>& A, GridFunc<T>& B) const override
    {
        assert(A.grid().sizeg() == PB<T>::grid_.sizeg());
        assert(B.grid().sizeg() == PB<T>::grid_.sizeg());

        FDoper<T>::rhs_4th_Mehr1(A, B);
        B.set_bc(A.bc(0), A.bc(1), A.bc(2));
    }
    void rhs(GridFunc<T>& A, T* const B) const override
    {
        FDoper<T>::rhs_4th_Mehr1(A, B);
    }

    void jacobi(GridFunc<T>&, const GridFunc<T>&, GridFunc<T>&) override;

    void get_vepsilon(GridFunc<T>&, GridFunc<T>&, GridFunc<T>&) override;

    ~PBh4M() override
    {
        // std::cout<<"destroy PBh4M"<<endl;
    }
};

} // namespace pb

#endif
