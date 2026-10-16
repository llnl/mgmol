// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "ConstraintSet.h"
#include "ExtendedGridOrbitals.h"
#include "Hamiltonian.h"
#include "LocGridOrbitals.h"
#include "MGmol.h"
#include "Potentials.h"
#include "ReplicatedWorkSpace.h"
#ifdef MGMOL_USE_SCALAPACK
#include "SparseDistMatrix.h"
#endif

#include "mgmol_run.h"

template <class OrbitalsType>
int MGmol<OrbitalsType>::setupFromInput(const std::string filename)
{
    Control& ct = *(Control::instance());
    if (ct.verbose > 0)
        printWithTimeStamp(
            "MGmol<OrbitalsType>::setupFromInput()...", std::cout);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    /*
     * Setup global mesh for calculations
     */
    unsigned ngpts[3]    = { ct.ngpts_[0], ct.ngpts_[1], ct.ngpts_[2] };
    double origin[3]     = { ct.ox_, ct.oy_, ct.oz_ };
    const double cell[3] = { ct.lx_, ct.ly_, ct.lz_ };
    Mesh::setup(mmpi.commSpin(), ngpts, origin, cell, ct.lap_type);

    hamiltonian_.reset(new Hamiltonian<OrbitalsType>());
    Potentials& pot = hamiltonian_->potential();

    ct.registerPotentials(pot);
    ct.setSpecies(pot);

    if (ct.diel) pot.turnOnDiel();

    ct.adjust();

    Mesh* mymesh = Mesh::instance();
    if (ct.isLocMode()) mymesh->subdivGridx(ct.getMGlevels());

    const pb::PEenv& myPEenv = mymesh->peenv();

    if (ct.restart_info > 0)
        h5f_file_.reset(
            new HDFrestart(ct.restart_file, myPEenv, ct.restart_file_type));

    int status = readCoordinates(filename, false);
    if (status == -1) return -1;

    const short myspin = mmpi.myspin();
    const int nel      = ions_->getNValenceElectrons();
    // for the case of extended wavefunctions, we can determine the number
    // of empty states from the number of wavefunctions in restart file
    if (ct.restart_info > 2 && !ct.short_sighted)
    {
        std::string name = "Function";
        int count        = h5f_file_->countFunctionObjects(name);
        std::cout << "found " << count << " functions in restart file..."
                  << std::endl;
        int nempty = ct.withSpin() ? count - nel : int(count - (0.5 * nel));
        ct.setNempty(nempty);
    }
    ct.setNumst(myspin, nel);

    ct.setTolEnergy();
    ct.setSpreadRadius();

    ct.checkNLrange();

    // now that we know the number of states, we can set a few other static
    // data
    if (!ct.short_sighted)
    {
        ReplicatedWorkSpace<double>::instance().setup(ct.numst);

#ifdef MGMOL_USE_SCALAPACK
        if (!ct.rmatrices)
        {
            MatricesBlacsContext::instance().setup(mmpi.commSpin(), ct.numst);

            dist_matrix::DistMatrix<DISTMATDTYPE>::setBlockSize(64);

            dist_matrix::DistMatrix<DISTMATDTYPE>::setDefaultBlacsContext(
                MatricesBlacsContext::instance().bcxt());

            dist_matrix::SparseDistMatrix<
                DISTMATDTYPE>::setNumTasksPerPartitioning(128);

            int npes = mmpi.size();
            setSparseDistMatriConsolidationNumber(npes);
        }
#endif
    }

    if (ct.rmatrices) ReplicatedMatrix::setMPIcomm(mmpi.commSpin());

    OrbitalsType::setDotProduct(ct.dot_product_type);

    mgmol_check();

    return 0;
}

template <class OrbitalsType>
int MGmol<OrbitalsType>::setupLRs(const std::string filename)
{
    Control& ct = *(Control::instance());

    if (!(ct.isLocMode() || ct.init_loc == 1)) return 0;

    // create localization regions
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();
    Vector3D vcell(mygrid.ll(0), mygrid.ll(1), mygrid.ll(2));
    lrs_.reset(new LocalizationRegions(vcell, ct.tol_orb_centers_move));

    if (ct.restart_info < 3 || !ct.isLocMode()) setupLRsFromInput(filename);

    return 0;
}

template <class OrbitalsType>
int MGmol<OrbitalsType>::setupLRsFromInput(const std::string filename)
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    std::ifstream* tfile = nullptr;
    if (mmpi.instancePE0() && !filename.empty())
    {
        os_ << "Read LRs from file " << filename << std::endl;
        tfile = new std::ifstream(filename.data(), std::ios::in);
        if (!tfile->is_open())
        {
            std::cerr << " Unable to open file " << filename << std::endl;
            mmpi.abort();
        }
        else
        {
            os_ << "Open " << filename << std::endl;
        }
    }

    readLRsFromInput(tfile);

    if (!(tfile == nullptr))
    {
        os_ << "Close " << filename << std::endl;
        tfile->close();
        delete tfile;
    }

    return 0;
}

template <class OrbitalsType>
int MGmol<OrbitalsType>::setupConstraintsFromInput(const std::string filename)
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    std::ifstream* tfile = nullptr;
    if (mmpi.instancePE0() && !filename.empty())
    {
        os_ << "Read constraints from file " << filename << std::endl;
        tfile = new std::ifstream(filename.data(), std::ios::in);
        if (!tfile->is_open())
        {
            std::cerr << " Unable to open file " << filename << std::endl;
            mmpi.abort();
        }
        else
        {
            os_ << "Open " << filename << std::endl;
        }
    }

    constraints_->readConstraints(tfile);

    if (!(tfile == nullptr))
    {
        os_ << "Close " << filename.data() << std::endl;
        tfile->close();
        delete tfile;
    }

    return 0;
}

template class MGmol<LocGridOrbitals<ORBDTYPE>>;
template class MGmol<ExtendedGridOrbitals<ORBDTYPE>>;
