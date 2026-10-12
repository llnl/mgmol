// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef PB_LAPH2_H
#define PB_LAPH2_H

#include "Lap.h"

namespace pb
{
template <class T>
class Laph2 : public Lap<T>
{
    double diagEl_;
    double invDiagEl_;

    Laph2* lower_order_op_;

public:
    using memory_space_type = FDoperInterface::memory_space_type;

    Laph2(const Grid& mygrid) : Lap<T>(mygrid)
    {
        // cout<<" Create Laph2 operator\n";

        diagEl_
            = 2. * (Lap<T>::inv_h2(0) + Lap<T>::inv_h2(1) + Lap<T>::inv_h2(2));
        invDiagEl_    = 1. / diagEl_;
        Lap<T>::name_ = "2nd order";

        lower_order_op_ = nullptr;
    }

    ~Laph2() override
    {
        if (lower_order_op_ != nullptr)
        {
            delete lower_order_op_;
            lower_order_op_ = nullptr;
        }
    }

    // construct a coarse grid operator
    Laph2 coarseOp(const Grid& mygrid)
    {
        Grid coarse_G(mygrid.coarse_grid(), 1);

        Laph2 A(coarse_G);

        return A;
    }

    Laph2 replicatedOp(const Grid& replicated_grid) const
    {
        Laph2 replicated_A(replicated_grid);

        return replicated_A;
    }

    void setLowerOrderGrid() override
    {
        this->setFDLowerOrderGrid(minNumberGhosts());
    }

    Laph2& getLowerOrderOp()
    {
        if (lower_order_op_ == nullptr)
        {
            this->setFDLowerOrderGrid(Laph2::minNumberGhosts());
            lower_order_op_ = new Laph2(Lap<T>::getLowerOrderGrid());
        }
        return *lower_order_op_;
    }

    static short minNumberGhosts() { return 1; }
    static short order() { return 2; }

    // A->B
    void apply(GridFunc<T>& A, GridFunc<T>& B) override
    {
        // if(grid_.mype_env().mytask()==0)cout<<Lap<T>::name_<<endl;
        this->del2_2nd(A, B);
        B.set_bc(A.bc(0), A.bc(1), A.bc(2));
    }

    void jacobi(GridFunc<T>&, const GridFunc<T>&, GridFunc<T>&) override;

    // optimal factor according to "Multigrid" by Trottenberg, Osterlee,
    // Schueler p. 73
    double jacobiFactor() const override { return 6. * invDiagEl_ / 7.; }

    double diagEl(void) const override { return diagEl_; };
    double invDiagEl(void) const override { return invDiagEl_; };
};

} // namespace pb

#endif
