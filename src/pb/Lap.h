// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef PB_LAP_H
#define PB_LAP_H

#include "FDoper.h"

#include <string>

namespace pb
{
enum struct LapType
{
    Mehrstellen1,
    Mehrstellen2,
    FD2nd,
    FD4th,
    FD6th,
    FD8th,
    UNDEFINED
};

template <class T>
class Lap : public FDoper<T>
{
public:
    using memory_space_type = typename FDoperInterface::memory_space_type;

protected:
    std::string name_;

public:
    Lap(const Grid& mygrid) : FDoper<T>(mygrid) { }

    Lap& operator=(const Lap& v) = default;

    // A->B
    void apply(GridFunc<T>& A, GridFunc<T>& B) override = 0;
    virtual void applyWithPot(GridFunc<T>&, const double* const, T*)
    {
        std::cerr << "ERROR: Lap::applyWithPot() not implemented" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::string name() const { return name_; }

    virtual void jacobi(GridFunc<T>&, const GridFunc<T>&, GridFunc<T>&) = 0;

    void jacobi(GridFunc<T>&, const GridFunc<T>&, GridFunc<T>&, const double);

    double energyES(GridFunc<T>&, GridFunc<T>&);
    virtual double diagEl(void) const       = 0;
    virtual double invDiagEl(void) const    = 0;
    virtual void setLowerOrderGrid(void)    = 0;
    virtual double jacobiFactor(void) const = 0;

    ~Lap() override { }
};

} // namespace pb

#endif
