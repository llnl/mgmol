// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef PB_H
#define PB_H

#include "DielFunc.h"
#include "FDoper.h"
#include "MGmol_abort.h"

// operator -epsilon div*grad -(grad epsilon)* grad

namespace pb
{
template <class T>
class PB : public FDoper<T>
{
protected:
    bool initialized_;
    DielFunc<T> epsilon_;
    GridFunc<T> work1_;
    GridFunc<T> work2_;

public:
    // constructor
    PB(const Grid& mygrid, const double e0)
        : FDoper<T>(mygrid),
          epsilon_(FDoper<T>::grid_, e0),
          work1_(FDoper<T>::grid_, 0, 0, 0),
          work2_(FDoper<T>::grid_, 0, 0, 0)
    {
        initialized_ = false;
    };

    // constructor
    PB(const Grid& mygrid, const double e0, const double rho0,
        const double drho0)
        : FDoper<T>(mygrid),
          epsilon_(FDoper<T>::grid_, e0, rho0, drho0),
          work1_(FDoper<T>::grid_, 0, 0, 0),
          work2_(FDoper<T>::grid_, 0, 0, 0)
    {
        initialized_ = false;
    };

    // constructor
    PB(const Grid& mygrid, DielFunc<T>& myepsilon)
        : FDoper<T>(mygrid),
          epsilon_(myepsilon, FDoper<T>::grid_),
          work1_(FDoper<T>::grid_, 0, 0, 0),
          work2_(FDoper<T>::grid_, 0, 0, 0)
    {
        if (!epsilon_.updated_boundaries()) epsilon_.trade_boundaries();
        // std::cout<<"Build operator PB"<<std::endl;
    };

    // copy constructor
    PB(const PB& oper)
        : FDoper<T>(oper.grid_),
          epsilon_(FDoper<T>::grid_),
          work1_(FDoper<T>::grid_, 0, 0, 0),
          work2_(FDoper<T>::grid_, 0, 0, 0)
    {
        // std::cout<<"Copy constructor for PB"<<std::endl;
        assert(FDoper<T>::grid_.sizeg() > 1);
        epsilon_     = oper.epsilon_;
        work1_       = oper.work1_;
        work2_       = oper.work2_;
        initialized_ = oper.initialized();
    }

    PB& operator=(const PB& /*oper*/)
    {
        mgmol_abort("operator= of PB");
        return *this;
    }

    // A->B
    void apply(GridFunc<T>& A, GridFunc<T>& B) override = 0;

    void init_epsilon(
        GridFunc<T>& gf_rhod, const double rho0, const double drho0)
    {
        epsilon_.Gepsilon_rho(gf_rhod, rho0, drho0);
    }

    GridFunc<T> operator*(GridFunc<T>& A)
    {
        GridFunc<T> work(A.grid(), A.bc(0), A.bc(1), A.bc(2));

        apply(A, work);

        return work;
    }

    double energyES(GridFunc<T>&, GridFunc<T>&);

    bool initialized() const { return initialized_; }

    virtual void jacobi(GridFunc<T>& A, const GridFunc<T>& B, GridFunc<T>& W)
        = 0;

    virtual void get_vepsilon(GridFunc<T>&, GridFunc<T>&, GridFunc<T>&) = 0;
    virtual void setLowerOrderGrid()                                    = 0;

    ~PB() override
    {
        // std::cout<<"destroy PB"<<std::endl;
    }
};

} // namespace pb

#endif
