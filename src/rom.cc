// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "mgmol_config.h"
#ifdef MGMOL_HAS_LIBROM

#include "ExtendedGridOrbitals.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"

#include "librom.h"

#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>

// Save the wavefunction snapshots
template <class OrbitalsType>
int MGmol<OrbitalsType>::save_orbital_snapshot(
    std::string file_path, OrbitalsType& orbitals)
{
    std::string snapshot_filename = file_path;
    struct stat s;
    if (stat(file_path.c_str(), &s) == 0)
    {
        if (s.st_mode & S_IFDIR)
        {
            snapshot_filename = file_path + "/orbital";
        }
        else if (s.st_mode & S_IFREG)
        {
            snapshot_filename = file_path + "_orbital";
        }
        else
        {
            std::cout << file_path
                      << " exists but is not a directory or a file."
                      << std::endl;
            return 1;
        }
    }

    const int dim          = orbitals.getLocNumpt();
    const int totalSamples = orbitals.chromatic_number();

    CAROM::Options svd_options(dim, totalSamples, 1);
    CAROM::BasisGenerator basis_generator(
        svd_options, false, snapshot_filename);

    for (int i = 0; i < totalSamples; ++i)
        basis_generator.takeSample(orbitals.getPsi(i));

    basis_generator.writeSnapshot();

    return 0;
}

template <class OrbitalsType>
void MGmol<OrbitalsType>::project_orbital(
    std::string file_path, int rdim, OrbitalsType& orbitals)
{
    const int dim          = orbitals.getLocNumpt();
    const int totalSamples = orbitals.chromatic_number();

    CAROM::Options svd_options(dim, totalSamples, 1);
    CAROM::BasisGenerator basis_generator(svd_options, false, "foo");

    for (int i = 0; i < totalSamples; ++i)
        basis_generator.takeSample(orbitals.getPsi(i));
    const CAROM::Matrix* orbital_snapshots
        = basis_generator.getSnapshotMatrix();

    CAROM::BasisReader reader(file_path);
    CAROM::Matrix* orbital_basis = reader.getSpatialBasis(rdim);

    CAROM::Matrix* proj_orbital_coeff
        = orbital_basis->transposeMult(orbital_snapshots);
    CAROM::Matrix* proj_orbital_snapshots
        = orbital_basis->mult(proj_orbital_coeff);

    Control& ct = *(Control::instance());
    Mesh* mesh  = Mesh::instance();
    pb::GridFunc<ORBDTYPE> gf_psi(
        mesh->grid(), ct.bcWF[0], ct.bcWF[1], ct.bcWF[2]);
    CAROM::Vector snapshot, proj_snapshot;
    for (int i = 0; i < totalSamples; ++i)
    {
        orbital_snapshots->getColumn(i, snapshot);
        proj_orbital_snapshots->getColumn(i, proj_snapshot);
        gf_psi.assign(proj_snapshot.getData());
        orbitals.setPsi(gf_psi, i);
        snapshot -= proj_snapshot;
        std::cout << "Error for orbital " << i << " = " << snapshot.norm()
                  << std::endl;
    }
}

// template <typename ScalarType>
// void ExtendedGridOrbitals<ScalarType>::set(std::string file_path, int rdim)
//{
//     const int dim = getLocNumpt();

//    CAROM::BasisReader reader(file_path);
//    CAROM::Matrix* orbital_basis = reader.getSpatialBasis(rdim);

//    Control& ct = *(Control::instance());
//    Mesh* mymesh           = Mesh::instance();
//    pb::GridFunc<ORBDTYPE> gf_psi(mymesh->grid(), ct.bcWF[0], ct.bcWF[1],
//    ct.bcWF[2]); CAROM::Vector psi; for (int i = 0; i < rdim; ++i)
//    {
//        orbital_basis->getColumn(i, psi);
//        gf_psi.assign(psi.getData());
//        setPsi(gf_psi, i);
//    }
//}

template class MGmol<LocGridOrbitals<ORBDTYPE>>;
template class MGmol<ExtendedGridOrbitals<ORBDTYPE>>;

#endif // MGMOL_HAS_LIBROM
