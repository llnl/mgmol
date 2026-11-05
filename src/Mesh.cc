// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Mesh.h"
#include <cmath>
#include <iomanip>

MPI_Comm Mesh::comm_   = MPI_COMM_NULL;
Mesh* Mesh::pinstance_ = nullptr;

unsigned Mesh::ngpts_[3]    = { 0, 0, 0 };
double Mesh::origin_[3]     = { 0., 0., 0. };
double Mesh::lattice_[3]    = { 0., 0., 0. };
pb::LapType Mesh::lap_type_ = pb::LapType::UNDEFINED;
int Mesh::subdivx_          = 1;
int Mesh::numpt_            = -1;
int Mesh::loc_numpt_        = -1;

void Mesh::print(std::ostream& os) const
{
    os << " Grid:" << std::endl;
    os << std::fixed << std::setprecision(5) << "\t hx  = " << myGrid_->hgrid(0)
       << " bohr"
       << ",  hy  = " << myGrid_->hgrid(1) << " bohr"
       << ",  hz  = " << myGrid_->hgrid(2) << " bohr" << std::endl;
    os << "\t nx  = " << myGrid_->gdim(0) << ", ny  = " << myGrid_->gdim(1)
       << ", nz  = " << myGrid_->gdim(2) << std::endl
       << std::endl;
    os << " subdivx = " << subdivx_ << std::endl;

    os << std::endl
       << " Grid anisotropy " << myGrid_->anisotropy() << std::endl;

    os << std::endl
       << " Processors topology: " << myPEenv_->n_mpi_task(0) << "*"
       << myPEenv_->n_mpi_task(1) << "*" << myPEenv_->n_mpi_task(2)
       << std::endl;

    double t1 = M_PI / (myGrid_->hmax());
    t1        = 0.5 * t1 * t1;
    os << " Equivalent energy cutoff ( 0.5*(PI/hmax)^2 ): " << t1 << " Rydbergs"
       << std::endl;
    os << " Orthorhombic cell:" << std::endl;
    os << " Origin = " << myGrid_->origin(0) << "\t" << myGrid_->origin(1)
       << "\t" << myGrid_->origin(2) << std::endl;
    os << " Dimension = " << myGrid_->ll(0) << "\t" << myGrid_->ll(1) << "\t"
       << myGrid_->ll(2) << std::endl;
    os << std::endl;
}

void Mesh::subdivGridx(const int nlevels)
{
    assert(subdivx_ == 1);
    assert(nlevels >= 1);
    assert(numpt_ > 1);

    // minimum # grid points/subdomain
    // note: we don't want too many as it increases the cost of all to all
    // communications when gathering data for local (subdomain) matrices
    const int smin = std::max(2 * (1 << nlevels), 16);

    const int dimx = myGrid_->dim(0);
    for (int i = smin; i <= dimx; i += 4)
        if (dimx % i == 0)
        {
            subdivx_ = dimx / i;
            break;
        }

    loc_numpt_ = numpt_ / subdivx_;
}
