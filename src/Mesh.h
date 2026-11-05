// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_MESH_H
#define MGMOL_MESH_H

#include "GridFactory.h"
#include "PEenv.h"
#include <cassert>
#include <ostream>

#include <mpi.h>

// Main Mesh structure
class Mesh
{
    static Mesh* pinstance_;

    static MPI_Comm comm_;

    static unsigned ngpts_[3];
    static double origin_[3];
    static double lattice_[3];
    static pb::LapType lap_type_;

    // nb. subdivision of grid in direction 0
    static int subdivx_;
    static int numpt_;
    static int loc_numpt_;

    pb::Grid* myGrid_;
    pb::PEenv* myPEenv_;

    Mesh()
    {
        myPEenv_ = new pb::PEenv(comm_, ngpts_[0], ngpts_[1], ngpts_[2], 1);

        myGrid_ = GridFactory::createGrid(
            ngpts_, origin_, lattice_, lap_type_, false, *myPEenv_);
        numpt_     = myGrid_->size();
        loc_numpt_ = numpt_;
    };

    ~Mesh()
    {
        delete myGrid_;
        delete myPEenv_;
    };
    Mesh(const Mesh&) {};

public:
    static Mesh* instance()
    {
        if (pinstance_ == nullptr)
        {
            pinstance_ = new Mesh();
        }
        return pinstance_;
    }

    static void deleteInstance()
    {
        delete pinstance_;
        pinstance_ = nullptr;
    }

    static void setup(MPI_Comm comm, const unsigned ngpts[3],
        const double origin[3], const double lattice[3],
        const pb::LapType lap_type)
    {
        assert(pinstance_ == NULL);
        assert(ngpts[0] > 0);
        assert(ngpts[1] > 0);
        assert(ngpts[2] > 0);
        assert(lattice[0] > 1.e-3);
        assert(lattice[1] > 1.e-3);
        assert(lattice[2] > 1.e-3);
        comm_ = comm;
        for (int i = 0; i < 3; i++)
        {
            ngpts_[i]   = ngpts[i];
            origin_[i]  = origin[i];
            lattice_[i] = lattice[i];
        }
        lap_type_ = lap_type;
    }

    void print(std::ostream&) const;

    void subdivGridx(const int nlevels = 1);

    const pb::Grid& grid() const { return *myGrid_; }
    const pb::PEenv& peenv() const { return *myPEenv_; }
    int subdivx() const { return subdivx_; }
    int numpt() const
    {
        assert(numpt_ > 0);
        assert(numpt_ < 1000000000);
        return numpt_;
    }
    int locNumpt() const
    {
        assert(loc_numpt_ > 0);
        assert(numpt_ < 1000000000);
        return loc_numpt_;
    }
    short ilow0(const short iloc) const
    {
        return myGrid_->istart(0) + iloc * myGrid_->dim(0) / subdivx_;
    }

    short ihigh0(const short iloc) const
    {
        return ilow0(iloc) + myGrid_->dim(0) / subdivx_ - 1;
    }
    short ihigh1() const { return myGrid_->istart(1) + myGrid_->dim(1) - 1; }
    short ihigh2() const { return myGrid_->istart(2) + myGrid_->dim(2) - 1; }

    int npointsPatch()
    {
        return myGrid_->dim(0) * myGrid_->dim(1) * myGrid_->dim(2) / subdivx_;
    }
};

#endif
