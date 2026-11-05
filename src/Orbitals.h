// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_ORBITALS_H
#define MGMOL_ORBITALS_H

#include "memory_space.h"

#include <hdf5.h>

enum struct OrthoType
{
    Eigenfunctions,
    Nonorthogonal,
    Orthonormal,
    UNDEFINED
};

class Orbitals
{
    int iterative_index_;

public:
#ifdef HAVE_MAGMA
    using memory_space_type = MemorySpace::Device;
#else
    using memory_space_type = MemorySpace::Host;
#endif

    Orbitals() { iterative_index_ = -10; }

    virtual ~Orbitals() {};

    Orbitals(const Orbitals& A, const bool copy_data)
    {
        if (copy_data)
        {
            iterative_index_ = A.iterative_index_;
        }
        else
        {
            iterative_index_ = -10;
        }
    }

    void resetIterativeIndex() { iterative_index_ = 0; }

    void setIterativeIndex(const Orbitals& orbitals)
    {
        iterative_index_ = orbitals.iterative_index_;
    }

    void setIterativeIndex(const short iterative_index)
    {
#ifdef DEBUG
        MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
        mmpi.barrier();
#endif
        iterative_index_ = iterative_index;
    }

    void incrementIterativeIndex(const short inc = 1)
    {
        iterative_index_ += inc;
    };

    bool compareIterativeIndex(const Orbitals& orbitals) const
    {
        assert(orbitals.iterative_index_ >= 0);
        return (iterative_index_ == orbitals.iterative_index_);
    }

    short getIterativeIndex() const { return iterative_index_; }

    hid_t outHdfDataType(const short out_restart_info) const
    {
        hid_t dtype_id
            = out_restart_info > 3 ? H5T_NATIVE_DOUBLE : H5T_NATIVE_FLOAT;
        return dtype_id;
    }

private:
    Orbitals& operator=(const Orbitals& orbitals);
};
#endif
