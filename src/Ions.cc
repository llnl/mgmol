// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Ions.h"
#include "Control.h"
#include "MGmol_MPI.h"
#include "MGmol_blas1.h"
#include "MPIdata.h"
#include "Mesh.h"
#include "Species.h"
#include "Timer.h"
#include "hdf_tools.h"
#include "mgmol_mpi_tools.h"
#include "tools.h"

#include <cmath>
#include <iostream>
#include <iterator>
#include <mpi.h>

Timer ions_setupInteractingIons_tm("ions_setupInteractingIons");
Timer ions_setup_tm("ions::setup");

const double ang2bohr = 1.8897269;
// const double rmax = 8.0;

std::map<std::string, short> Ions::map_species_ = { { "H", 1 }, { "D", 1 },
    { "Li", 3 }, { "Be", 4 }, { "B", 5 }, { "C", 6 }, { "N", 7 }, { "O", 8 },
    { "F", 9 }, { "Na", 11 }, { "Mg", 12 }, { "Al", 13 }, { "Si", 14 },
    { "P", 15 }, { "S", 16 }, { "Cl", 17 }, { "K", 19 }, { "Ca", 20 },
    { "Cr", 24 }, { "Mn", 25 }, { "Fe", 26 }, { "Co", 27 }, { "Ni", 28 },
    { "Cu", 29 }, { "Zn", 30 }, { "Ga", 31 }, { "Ge", 32 }, { "Br", 35 },
    { "La", 57 }, { "Au", 79 } };

int Ions::num_ions_          = -1;
short Ions::max_num_proj_    = -1;
double Ions::max_Vl_radius_  = -1.;
double Ions::max_Vnl_radius_ = -1.;

template <typename T>
void writeData2d(HDFrestart& h5f_file, std::string datasetname,
    std::vector<T>& data, const size_t n, T element)
{
    hid_t file_id = h5f_file.file_id();
#ifdef MGMOL_USE_HDF5P
    if (h5f_file.useHdf5p())
    {
        // fill up data array to dimension common to all tasks
        short s = data.size();
        short ms;
        mgmol_tools::allreduce(&s, &ms, 1, MPI_MAX, h5f_file.comm_active());
        for (short i = s; i < ms; i++)
            data.push_back(element);
        size_t dims[2] = { data.size() / n, n };

        mgmol_tools::parallelWrite2d(
            file_id, datasetname, data, dims, h5f_file.comm_active());
    }
    else
#endif
    {
        size_t dims[2] = { data.size() / n, n };
        mgmol_tools::write2d(file_id, datasetname, data, dims);
    }
}

Ions::Ions(const double lat[3], const std::vector<Species>& sp)
    : species_(sp), setup_(false), has_locked_atoms_(false)
{
    setupSubdomains(lat);
}

void Ions::setupSubdomains(const double lat[3])
{
    MGmol_MPI& mmpi          = *(MGmol_MPI::instance());
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    for (short i = 0; i < 3; i++)
    {
        assert(lat[i] > 0.);
        if (lat[i] > 10000.)
        {
            (*MPIdata::serr) << "Ions constructor: lattice[" << i
                             << "]=" << lat[i] << "!!!" << std::endl;
            mmpi.abort();
        }
        lattice_[i] = lat[i];
    }

    for (short i = 0; i < 3; i++)
        div_lattice_[i] = lattice_[i] / (double)(myPEenv.n_mpi_task(i));

    double offset[3] = { (double)myPEenv.my_mpi(0) * div_lattice_[0],
        (double)myPEenv.my_mpi(1) * div_lattice_[1],
        (double)myPEenv.my_mpi(2) * div_lattice_[2] };

    for (short i = 0; i < 3; i++)
        if (myPEenv.n_mpi_task(i) == (myPEenv.my_mpi(i) + 1))
            div_lattice_[i] = lattice_[i] - offset[i];

    // save cartesian communicator info
    cart_comm_     = myPEenv.cart_comm();
    const int disp = -1;
    for (int dir = 0; dir < 3; dir++)
        MPI_Cart_shift(cart_comm_, dir, disp, &source_[dir], &dest_[dir]);
}

Ions::Ions(const std::vector<double>& p, const std::vector<short>& anum,
    const double lat[3], const std::vector<Species>& sp)
    : species_(sp)
{
    setupSubdomains(lat);

    double rmax = getMaxListRadius();
    setupListIonsBoundaries(rmax);

    num_ions_ = setAtoms(p, anum);

    setup();
}

Ions::Ions(const Ions& ions, const double shift[3]) : species_(ions.species_)
{
    for (const auto& ion : ions.list_ions_)
    {
        Ion* newion = new Ion(*ion);
        newion->shiftPosition(shift);
        newion->setup();
        list_ions_.push_back(newion);
    }
    for (short i = 0; i < 3; ++i)
        lattice_[i] = ions.lattice_[i];

    std::vector<Ion*>::iterator iion       = list_ions_.begin();
    std::vector<Ion*>::const_iterator cion = ions.list_ions_.begin();
    while (iion != list_ions_.end())
    {
        (*iion)->set_here((*cion)->here());
        if ((*cion)->here())
        {
            local_ions_.push_back(*iion);
        }
        iion++;
        cion++;
    }
    local_names_      = ions.local_names_;
    atmove_           = ions.atmove_;
    pmass_            = ions.pmass_;
    taum_             = ions.taum_;
    tau0_             = ions.tau0_;
    taup_             = ions.taup_;
    fion_             = ions.fion_;
    velocity_         = ions.velocity_;
    rand_states_      = ions.rand_states_;
    has_locked_atoms_ = ions.has_locked_atoms_;

    gids_ = ions.gids_;

    setupListOverlappingIons();

    setupInteractingIons();

    lstep_[0] = ions.lstep_[0];
    lstep_[1] = ions.lstep_[1];
    lstep_[2] = ions.lstep_[2];
    rstep_[0] = ions.rstep_[0];
    rstep_[1] = ions.rstep_[1];
    rstep_[2] = ions.rstep_[2];
}

void Ions::computeMaxNumProjs()
{
#ifdef DEBUG
    if (onpe0)
        (*MPIdata::sout) << " list_ions of size " << list_ions_.size()
                         << " initialized" << std::endl;
#endif

    for (auto& ion : list_ions_)
    {
        max_num_proj_ = std::max(max_num_proj_, ion->nProjectors());
    }

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&max_num_proj_, 1, MPI_MAX);
}

void Ions::setup()
{
    Control& ct = *(Control::instance());

    if (ct.verbose > 0) printWithTimeStamp("Ions::setup()...", std::cout);

    ions_setup_tm.start();

    Ion::resetIndexCount();

    updateListIons();

    // #ifndef NDEBUG
    //     checkUnicityLocalIons();
    // #endif

    setupInteractingIons();

    // initialize ionic stepper data
    initStepperData();

    has_locked_atoms_ = hasLockedAtoms();

    // initialize data for constraints
    setupContraintsData();

    computeMaxNumProjs();

    if (ct.verbose > 0)
        printWithTimeStamp("Ions::setup()... individual ions...", std::cout);

    for (auto& ion : list_ions_)
    {
        ion->setup();
    }

    setMapVL();
    setupListOverlappingIons();

    max_Vl_radius_  = computeMaxVlRadius();
    max_Vnl_radius_ = computeMaxNLprojRadius();

    setup_ = true;

    assert(max_Vnl_radius_ >= 0.);

    ions_setup_tm.stop();
}

Ions::~Ions()
{
    std::vector<Ion*>::iterator ion = list_ions_.begin();
    while (ion != list_ions_.end())
    {
        delete *ion;
        ion++;
    }
}

void Ions::setupListOverlappingIons()
{
    Control& ct = *(Control::instance());
    if (ct.verbose > 1)
        printWithTimeStamp("Ions::setupListOverlappingIons()...", std::cout);

    overlappingNL_ions_.clear();
    overlappingVL_ions_.clear();

    for (auto ion : list_ions_)
    {
        if (ion->map_nl())
        {
            overlappingNL_ions_.push_back(ion);
        }
    }

    for (auto ion : list_ions_)
    {
        if (ion->map_l())
        {
            overlappingVL_ions_.push_back(ion);
        }
    }
}

void Ions::setupInteractingIons()
{
    Control& ct = *(Control::instance());
    if (ct.verbose > 1)
        printWithTimeStamp("Ions::setupInteractingIons()...", std::cout);

    ions_setupInteractingIons_tm.start();
    const double rmax = ct.maxDistanceAtomicInfo();
    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::setupInteractingIons() with radius " << rmax
                         << std::endl;
    }

    interacting_ions_.clear();

    for (auto ion1 : list_ions_)
    {
        // is ion1 interacting with any local ions?
        for (auto ion2 : local_ions_)
        {
            const double r12 = ion1->minimage(*ion2, lattice_, ct.bcPoisson);

            if (r12 < rmax)
            {
                // ion1 is interacting with local ions
                interacting_ions_.push_back(ion1);
                break;
            }
        }
    }

    //(*MPIdata::sout)<<"Number of interacting ions =
    //"<<interacting_ions_.size()<<endl;
    ions_setupInteractingIons_tm.stop();
}

// setup arrays to be used in constraints enforcement
// using references to local_ions and extra "dummy" data
void Ions::setupContraintsData()
{
    Control& ct = *(Control::instance());

    if (ct.verbose > 0)
        printWithTimeStamp("Ions::setupContraintsData()...", std::cout);
    const int nnloc = interacting_ions_.size() - local_ions_.size();
    // std::cout<<"interacting_ions_.size()="<<interacting_ions_.size()<<endl;
    // std::cout<<"local_ions_.size()="<<local_ions_.size()<<endl;
    assert(nnloc >= 0);

    tau0_dummy_.resize(3 * nnloc);
    taup_dummy_.resize(3 * nnloc);
    fion_dummy_.resize(3 * nnloc);
    names_dummy_.clear();
    pmass_dummy_.clear();
    atmove_dummy_.clear();

    interacting_tau0_.clear();
    interacting_taup_.clear();
    interacting_fion_.clear();
    interacting_names_.clear();
    interacting_atmove_.clear();
    interacting_pmass_.clear();

    int ia = 0; // count local ions
    int ib = 0; // count non-local ions
    for (auto& ion : interacting_ions_)
    {
        if (isLocal(ion->name()))
        {
            assert(3 * ia < static_cast<int>(tau0_.size()));
            assert(3 * ia < static_cast<int>(taup_.size()));
            assert(3 * ia < static_cast<int>(fion_.size()));
            assert(ia < static_cast<int>(pmass_.size()));
            assert(ia < static_cast<int>(atmove_.size()));

            // set reference to local data
            interacting_tau0_.push_back(&tau0_[3 * ia]);

            interacting_taup_.push_back(&taup_[3 * ia]);

            interacting_fion_.push_back(&fion_[3 * ia]);

            interacting_names_.push_back(local_names_[ia]);
            interacting_pmass_.push_back(pmass_[ia]);
            interacting_atmove_.push_back(atmove_[ia]);

            ++ia;
        }
        else
        {
            assert(3 * ib < static_cast<int>(tau0_dummy_.size()));
            assert(3 * ib < static_cast<int>(taup_dummy_.size()));
            assert(3 * ib < static_cast<int>(fion_dummy_.size()));

            // fill up dummy data
            ion->getPosition(&tau0_dummy_[3 * ib]);
            ion->getPosition(&taup_dummy_[3 * ib]);
            ion->getForce(&fion_dummy_[3 * ib]);
            names_dummy_.push_back(ion->name());
            pmass_dummy_.push_back(ion->getMass());
            atmove_dummy_.push_back(!ion->locked());

            // set references to dummy data
            interacting_tau0_.push_back(&tau0_dummy_[3 * ib]);

            interacting_taup_.push_back(&taup_dummy_[3 * ib]);

            interacting_fion_.push_back(&fion_dummy_[3 * ib]);

            assert(ib < static_cast<int>(names_dummy_.size()));
            assert(ib < static_cast<int>(pmass_dummy_.size()));
            assert(ib < static_cast<int>(atmove_dummy_.size()));
            interacting_names_.push_back(names_dummy_[ib]);
            interacting_pmass_.push_back(pmass_dummy_[ib]);
            interacting_atmove_.push_back(atmove_dummy_[ib]);

            ++ib;
        }
    }
}

void Ions::iiforce(const short bc[3])
{
    const int nlions = local_ions_.size();
    std::vector<double> forces(3 * nlions, 0.);

    std::vector<Ion*>::const_iterator ion1 = local_ions_.begin();
    int ion1_index                         = 0;

    while (ion1 != local_ions_.end())
    {
        const double z1 = (*ion1)->getZion();
        assert(z1 >= 0.);

        const double rc1 = (*ion1)->getRC();

        std::vector<Ion*>::const_iterator ion2 = interacting_ions_.begin();
        while (ion2 != interacting_ions_.end())
        {
            if (*ion1 != *ion2)
            {
                // Minimum image convention for r
                double dr[3];
                const double r12 = (*ion1)->minimage(**ion2, lattice_, bc, dr);

                const double invr = 1. / r12;

                const double z2 = (*ion2)->getZion();
                assert(z2 >= 0.);

                const double rc2 = (*ion2)->getRC();

                const double t        = rc1 * rc1 + rc2 * rc2;
                const double invt     = 1. / t;
                const double sqrtinvt = sqrt(invt);

                const double s1 = z1 * z2 * invr * invr;
                const double s2 = erfc(r12 * sqrtinvt) * invr;
                const double s3
                    = M_2_SQRTPI * exp(-r12 * r12 * invt) * sqrtinvt;
                const double alpha = s1 * (s2 + s3);

                for (short i = 0; i < 3; i++)
                    forces[3 * ion1_index + i] += dr[i] * alpha;
            }

            ion2++;
        }
        ion1_index++;
        ion1++;
    }

    std::vector<Ion*>::iterator lion = local_ions_.begin();
    int ion_index                    = 0;
    while (lion != local_ions_.end())
    {
        (*lion)->add_force(forces[3 * ion_index + 0], forces[3 * ion_index + 1],
            forces[3 * ion_index + 2]);
        ion_index++;
        lion++;
    }
}

double Ions::energySelf() const
{
    double energy = 0.;

    for (auto ion : local_ions_)
    {
        energy += ion->eself();
    }
    // multiply by 0.5, 2/sqrt(pi) and 1/sqrt(2) to get 1./sqrt(2*pi)
    energy *= 0.5 * M_2_SQRTPI * M_SQRT1_2;

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&energy, 1, MPI_SUM);

    assert(energy == energy);
    return energy; // Hartree
}

// right-looking approach avoids duplicate computation of energy on different
// processors. This works since ion lists are in ascending order of unique ion
// index
double Ions::energyDiff(const short bc[3]) const
{
    double energy = 0.;

    assert(lattice_[0] > 0.);
    assert(lattice_[1] > 0.);
    assert(lattice_[2] > 0.);

    std::vector<Ion*>::const_iterator ion1 = interacting_ions_.begin();
    while (ion1 != interacting_ions_.end())
    {
        if ((*ion1)->here())
        {
            std::vector<Ion*>::const_iterator ion2 = interacting_ions_.begin();
            ;
            while (ion2 != ion1)
            {
                energy += (*ion1)->energyDiff(**ion2, lattice_, bc);

                ion2++;
            }
            // increment ion2 past ion1
            ion2++;
            while (ion2 != interacting_ions_.end())
            {
                energy += (*ion1)->energyDiff(**ion2, lattice_, bc);

                ion2++;
            }
        }

        ion1++;
    }
    assert(energy == energy);

    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    double tenergy           = 0.;
    MPI_Allreduce(&energy, &tenergy, 1, MPI_DOUBLE, MPI_SUM, myPEenv.comm());

    // take half the result to account for double counting loop over
    // interacting_ions
    energy = 0.5 * tenergy;

    return energy; // Hartree
}

// Writes out the positions of all the ions
// PE root does the writing
void Ions::printPositionsGlobal(std::ostream& os, const int root) const
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    std::map<int, std::string> ion_names;
    std::vector<double> positions;
    std::vector<int> islocked;

    gatherNames(ion_names, root, mmpi.commSameSpin());
    gatherPositions(positions, root);
    gatherLockedData(islocked, root);

    if (mmpi.mypeGlobal() == root)
    {

        os << std::endl << " IONIC POSITIONS:" << std::endl;

        os << std::setw(8) << "Atoms" << std::setw(8) << "X" << std::setw(10)
           << "Y" << std::setw(10) << "Z" << std::endl;

        os.setf(std::ios::right, std::ios::adjustfield);
        os.setf(std::ios::fixed, std::ios::floatfield);

        int ion_index                                     = 0;
        std::map<int, std::string>::const_iterator ion_id = ion_names.begin();
        while (ion_id != ion_names.end())
        {
            const int pos = 3 * ion_index;

            os << "$$ ";
            if (islocked[ion_index])
                os << "*";
            else
                os << " ";
            os << std::setw(4) << ion_id->second << std::setw(10)
               << std::setprecision(4) << std::fixed << positions[pos]
               << std::setw(10) << positions[pos + 1] << std::setw(10)
               << positions[pos + 2] << std::endl;

            ion_index++;
            ion_id++;
        }

        os << std::endl;
    }
}

// Writes out the postions of the local ions and their displacements from their
// initial postions.
void Ions::printPositionsLocal(std::ostream& os, const int root) const
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    if (mmpi.mypeGlobal() == root)
    {

        os << std::endl
           << " IONIC POSITIONS AND DISPLACEMENTS ON PE" << root << ":"
           << std::endl;

        os << std::setw(8) << "Atoms" << std::setw(8) << "X" << std::setw(10)
           << "Y" << std::setw(10) << "Z" << std::setw(10) << "dX"
           << std::setw(10) << "dY" << std::setw(10) << "dZ" << std::endl;

        os.setf(std::ios::right, std::ios::adjustfield);
        os.setf(std::ios::fixed, std::ios::floatfield);

        for (auto& ion : local_ions_)
        {
            ion->printPosition(os);
        }

        os << std::endl;
    }
}
// Writes out the postions of the ions and their displacements from their
// initial postions.
void Ions::printPositions(std::ostream& os, const int root) const
{
    Control& ct(*(Control::instance()));
    if (ct.verbose > 2)
    {
        if (num_ions_ < 256)
            printPositionsGlobal(os, root);
        else
            printPositionsLocal(os, root);
    }
}

void Ions::writeAtomicNumbers(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writeAtomicNumbers()..." << std::endl;
    }
    std::vector<int> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherAtomicNumbers(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            assert(ion->atomic_number() > 0);
            assert(ion->atomic_number() < 200);

            data.push_back(ion->atomic_number());
        }
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Atomic_numbers");
        writeData2d(h5f_file, datasetname, data, 1, -1);
    }
}

void Ions::writeAtomNames(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writeAtomNames" << std::endl;
    }

    // gather data to print locally into std::vector "data"
    std::vector<std::string> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherNames(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            data.push_back(ion->name());
        }
    }

    // write data
    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Atomic_names");
        std::string empty;
        writeData2d(h5f_file, datasetname, data, 1, empty);
    }
}

void Ions::lockAtom(const std::string& name)
{
    for (auto& ion : local_ions_)
    {
        std::string name_ion(ion->name());
        if (name.compare(name_ion) == 0)
        {
            ion->lock();
            if (onpe0) (*MPIdata::sout) << "Lock atom " << name << std::endl;
            break;
        }
    }
}

void Ions::readAndLockAtoms(HDFrestart& h5f_file)
{
    std::vector<std::string> data;
    std::string datasetname("/LockedAtomsNames");
    h5f_file.readAtomicData(datasetname, data);

    for (auto& name : data)
    {
        lockAtom(name);
    }
}

void Ions::writeLockedAtomNames(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writeLockedAtomsNames" << std::endl;
    }

    // gather data to print locally into std::vector "data"
    std::vector<std::string> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherLockedNames(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            if (ion->locked()) data.push_back(ion->name());
        }
    }

    if (!data.empty())
    {
        hid_t file_id = h5f_file.file_id();
        if (file_id >= 0)
        {
            std::string datasetname("/LockedAtomsNames");
            std::string empty;
            writeData2d(h5f_file, datasetname, data, 1, empty);
        }
    }
}

void Ions::writeAtomicIDs(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));
    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writeAtomicIDs()..." << std::endl;
    }

    // gather data to print locally into std::vector "data"
    std::vector<int> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherIndexes(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            data.push_back(ion->index());
        }
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Atomic_IDs");
        writeData2d(h5f_file, datasetname, data, 1, -1);
    }
}

void Ions::writeAtomicNLprojIDs(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writeAtomicNLprojIDs()..." << std::endl;
    }

    // gather data to print locally into std::vector "data"
    std::vector<int> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherNLprojIds(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            data.push_back(ion->nlprojid());
        }
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/AtomicNLproj_IDs");
        writeData2d(h5f_file, datasetname, data, 1, -1);
    }
}

void Ions::writePositions(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writePositions" << std::endl;
    }

    std::vector<double> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherPositions(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            data.push_back(ion->position(0));
            data.push_back(ion->position(1));
            data.push_back(ion->position(2));
        }
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Ionic_positions");
        writeData2d(h5f_file, datasetname, data, 3, 1.e32);
    }
}

void Ions::writePreviousPositions(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writePositions" << std::endl;
    }

    std::vector<double> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherPreviousPositions(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            data.push_back(ion->getPreviousPosition(0));
            data.push_back(ion->getPreviousPosition(1));
            data.push_back(ion->getPreviousPosition(2));
        }
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Ionic_previous_positions");
        writeData2d(h5f_file, datasetname, data, 3, 1.e32);
    }
}

void Ions::initFromRestartFile(HDFrestart& h5_file)
{
    assert(list_ions_.empty());
    assert(local_ions_.empty());

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    Control& ct(*(Control::instance()));
    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::setFromRestartFile()..." << std::endl;
    }

    // set up list boundaries
    // get radius of projectors
    double rmax = getMaxListRadius();
    assert(rmax > 0.);
    setupListIonsBoundaries(rmax);

    std::vector<int> at_numbers;
    std::string datasetname("/Atomic_numbers");
    h5_file.readAtomicData(datasetname, at_numbers);
    std::vector<int> at_indexes;
    std::string datasetname_indexes("/Atomic_IDs");
    int nidxs = h5_file.readAtomicData(datasetname_indexes, at_indexes);
    std::vector<int> at_nlprojIds;
    std::string datasetname_nlprojIds("/AtomicNLproj_IDs");
    int npids = h5_file.readAtomicData(datasetname_nlprojIds, at_nlprojIds);
    std::vector<std::string> at_names;
    std::string datasetname_names("/Atomic_names");
    h5_file.readAtomicData(datasetname_names, at_names);
    if (onpe0 && ct.verbose > 2)
    {
        std::cout << "HDF file: at nb=" << at_numbers.size() << std::endl;
        std::cout << "HDF file: names nb=" << at_names.size() << std::endl;
        std::cout << "HDF file: indexes nb=" << at_indexes.size() << std::endl;
        std::cout << "HDF file: at_nlprojIds nb=" << at_nlprojIds.size()
                  << std::endl;
    }

    // if reading "old" format with replicated atoms, fill up empty arrays
    // (atomic indexes were not saved before)
    if (nidxs == -1 && at_numbers.size() > 0)
    {
        for (unsigned int i = 0; i < at_numbers.size(); i++)
            at_indexes.push_back(i);
    }
    if (npids == -1 && at_numbers.size() > 0)
    {
        for (unsigned int i = 0; i < at_numbers.size(); i++)
            at_nlprojIds.push_back(i);
    }

    assert(at_numbers.size() == at_names.size());
    assert(at_numbers.size() == at_indexes.size());
    assert(at_numbers.size() == at_nlprojIds.size());

    num_ions_ = at_names.size();
#ifdef MGMOL_USE_HDF5P
    if (!h5_file.useHdf5p())
#endif
    {
        mmpi.allreduce(&num_ions_, 1, MPI_SUM);
    }

    if (onpe0 && ct.verbose > 0)
    {
        (*MPIdata::sout) << "Ions::setFromRestartFile(), read " << num_ions_
                         << " names... on PE0" << std::endl;
    }

    assert(at_numbers.size() == at_names.size());

    double default_vel[3]    = { 0., 0., 0. };
    double default_coords[3] = { 0., 0., 0. };

    for (unsigned int i = 0; i < at_numbers.size(); i++)
    {
        const int atnum                          = at_numbers[i];
        std::vector<Species>::const_iterator spi = species_.begin();
        while (spi != species_.end())
        {
            if (atnum == spi->getAtomicNumber()) break;
            spi++;
        }
        if (onpe0 && ct.verbose > 3)
            (*MPIdata::sout)
                << "New Ion with name " << at_names[i] << std::endl;
        if (at_indexes[i] >= 0)
        {
            Ion* new_ion = new Ion(*spi, at_names[i], default_coords,
                default_vel, at_indexes[i], at_nlprojIds[i]);

            list_ions_.push_back(new_ion);
            local_ions_.push_back(new_ion);
        }
    }
    readRestartPositions(h5_file);
    readRestartVelocities(h5_file);
    if (ct.LangevinThermostat()) readRestartRandomStates(h5_file);
    readAndLockAtoms(h5_file);

    // remove atoms from local list if not local
    for (std::vector<Ion*>::iterator it = local_ions_.begin();
         it != local_ions_.end();)
    {
        double p[3];
        (*it)->getPosition(p);
        if (!inLocalIons(p[0], p[1], p[2]))
            it = local_ions_.erase(it);
        else
            ++it;
    }

    // rescale all velocities by factor specified in input
    rescaleVelocities(ct.VelocityScalingFactor());

    setup_ = false;

    // update list ions
    updateListIons();

    // #ifndef NDEBUG
    //     checkUnicityLocalIons();
    // #endif
}

void Ions::readRestartPositions(HDFrestart& h5_file)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Read ionic positions from hdf5 file" << std::endl;

    std::vector<double> data;
    std::string datasetname("/Ionic_positions");
    h5_file.readAtomicData(datasetname, data);

    int i = 0;
    for (auto& ion : local_ions_)
    {
        ion->setPosition(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
        i++;
    }
}

void Ions::readRestartPreviousPositions(HDFrestart& h5_file)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Read ionic positions from hdf5 file" << std::endl;

    std::vector<double> data;
    std::string datasetname("/Ionic_previous_positions");
    h5_file.readAtomicData(datasetname, data);
    assert(data.size() == 3 * local_ions_.size());

    int i = 0;
    for (auto& ion : local_ions_)
    {
        ion->setPreviousPosition(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
        i++;
    }
}

void Ions::resetPositionsToPrevious()
{
    for (auto& ion : local_ions_)
    {
        ion->resetPositionsToPrevious();
    }
}

void Ions::writeVelocities(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
        (*MPIdata::sout) << "Ions::writeVelocities" << std::endl;

    std::vector<double> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherVelocities(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            data.push_back(ion->velocity(0));
            data.push_back(ion->velocity(1));
            data.push_back(ion->velocity(2));
        }
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Ionic_velocities");
        writeData2d(h5f_file, datasetname, data, 3, 1.e32);
    }
}

void Ions::writeRandomStates(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
    {
        (*MPIdata::sout) << "Ions::writeRandomStates()..." << std::endl;
    }

    std::vector<unsigned short> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherRandStates(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            data.push_back(ion->randomState(0));
            data.push_back(ion->randomState(1));
            data.push_back(ion->randomState(2));
        }
    }

    if (!data.empty())
        if (data[0] != data[0])
        {
            (*MPIdata::sout)
                << "WARNING: Ions::writeRandomStates: data[0]=" << data[0]
                << std::endl;
        }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Ionic_RandomStates");
        writeData2d(h5f_file, datasetname, data, 3, (unsigned short)0);
    }
}

void Ions::removeMassCenterMotion()
{
    // don't do it if some atoms are locked
    if (has_locked_atoms_) return;

    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Remove mass center motion" << std::endl;

    std::vector<double> mass(local_ions_.size());
    std::vector<double> velocities(3 * local_ions_.size());

    std::vector<Ion*>::iterator ion = local_ions_.begin();
    int i                           = 0;
    double tmp[4]                   = { 0., 0., 0., 0. };
    while (ion != local_ions_.end())
    {
        const int threei       = 3 * i;
        velocities[threei]     = (*ion)->velocity(0);
        velocities[threei + 1] = (*ion)->velocity(1);
        velocities[threei + 2] = (*ion)->velocity(2);
        mass[i]                = (*ion)->getMass();
        tmp[0] += mass[i] * velocities[threei];
        tmp[1] += mass[i] * velocities[threei + 1];
        tmp[2] += mass[i] * velocities[threei + 2];
        tmp[3] += mass[i];

        ion++;
        i++;
    }

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&tmp[0], 4, MPI_SUM);

    const double tmass = 1. / tmp[3];
    for (short j = 0; j < 3; j++)
    {
        tmp[j] *= tmass;
#ifdef DEBUG
        (*MPIdata::sout) << std::setprecision(12);
        if (onpe0) (*MPIdata::sout) << "V[" << j << "]=" << tmp[j] << std::endl;
#endif
    }

    i = 0;
    for (auto& ion : local_ions_)
    {
        const int threei = 3 * i;
        ion->setVelocity(velocities[threei] - tmp[0],
            velocities[threei + 1] - tmp[1], velocities[threei + 2] - tmp[2]);
        i++;
    }

#ifdef DEBUG // check velocity mass center is 0
    mv[0] = 0.;
    mv[1] = 0.;
    mv[2] = 0.;
    for (auto& ion : local_ions_)
    {
        velocities[3 * i]     = ion->velocity(0);
        velocities[3 * i + 1] = ion->velocity(1);
        velocities[3 * i + 2] = ion->velocity(2);
        mv[0] += mass[i] * velocities[3 * i];
        mv[1] += mass[i] * velocities[3 * i + 1];
        mv[2] += mass[i] * velocities[3 * i + 2];

        i++;
    }

    for (short j = 0; j < 3; j++)
    {
        mv[j] *= tmass;
        if (onpe0) (*MPIdata::sout) << "V[" << j << "]=" << mv[j] << std::endl;
    }
#endif
}

void Ions::readRestartVelocities(HDFrestart& h5_file)
{
    Control& ct = *(Control::instance());
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Read atomic velocities from hdf5 file"
                         << std::endl;

    std::vector<double> data;
    std::string datasetname("/Ionic_velocities");
    h5_file.readAtomicData(datasetname, data);

    int i = 0;
    for (auto& ion : local_ions_)
    {
        ion->setVelocity(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
        i++;
    }
}

void Ions::readRestartRandomStates(HDFrestart& h5f_file)
{
    if (onpe0)
        (*MPIdata::sout) << "Read atomic RandomStates from hdf5 file"
                         << std::endl;

    std::vector<unsigned short> data;
    h5f_file.readRestartRandomStates(data);

    int i = 0;
    for (auto& ion : local_ions_)
    {
        ion->setRandomState(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
        i++;
    }
}

void Ions::writeForces(HDFrestart& h5f_file)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 1)
        (*MPIdata::sout) << "Write ionic forces in hdf5 file" << std::endl;

    std::vector<double> data;
    if (h5f_file.gatherDataX())
    {
        Mesh* mymesh             = Mesh::instance();
        const pb::PEenv& myPEenv = mymesh->peenv();
        MPI_Comm comm            = myPEenv.comm_x();

        gatherForces(data, 0, comm);
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            // get position of local ion
            double force[3];
            ion->getForce(&force[0]);
            data.push_back(force[0]);
            data.push_back(force[1]);
            data.push_back(force[2]);
        }
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id >= 0)
    {
        std::string datasetname("/Ionic_forces");
        writeData2d(h5f_file, datasetname, data, 3, 1.e32);
    }
}

void Ions::setLocalForces(
    const std::vector<double>& forces, const std::vector<std::string>& names)
{
    assert(forces.size() == 3 * names.size());

    // loop over global list of forces and atom names
    std::vector<std::string>::const_iterator s = names.begin();
    for (auto it = forces.begin(); it != forces.end(); it += 3)
    {
        // find possible matching ion
        for (auto& ion : local_ions_)
        {
            if (ion->compareName(*s))
            {
                ion->set_force(0, *it);
                ion->set_force(1, *(it + 1));
                ion->set_force(2, *(it + 2));
            }
        }
        s++;
    }
}

void Ions::setLocalForces(
    const std::vector<double>& forces, const std::vector<double>& coords)
{
    assert(forces.size() == coords.size());

    // tolerance can be pretty loose, as long as it does not allow mix up
    // with coordinates of other atoms
    const double tol = 1.e-2;

    // loop over global list of forces and coordinates
    std::vector<double>::const_iterator cit = coords.begin();
    for (auto fit = forces.begin(); fit != forces.end(); fit += 3)
    {
        // find possible matching ion
        for (auto& ion : local_ions_)
        {
            double p[3];
            ion->getPosition(&p[0]);
            double d2 = (p[0] - (*cit)) * (p[0] - (*cit))
                        + (p[1] - (*(cit + 1))) * (p[1] - (*(cit + 1)))
                        + (p[2] - (*(cit + 2))) * (p[2] - (*(cit + 2)));
            double d = std::sqrt(d2);
            if (d < tol)
            {
                ion->set_force(0, *fit);
                ion->set_force(1, *(fit + 1));
                ion->set_force(2, *(fit + 2));
            }
        }
        cit += 3;
    }
}

// Writes out the postions of the ions and the current forces on them by root
void Ions::printForcesGlobal(std::ostream& os, const int root) const
{
    double maxf = 0., avfx = 0., avfy = 0., avfz = 0., maxfx = 0., maxfy = 0.,
           maxfz  = 0.;
    int num_atoms = 0;

    double sum_forces[3] = { 0., 0., 0. };
    int num_movable      = 0;

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    std::map<int, std::string> ion_names;
    std::vector<double> positions;
    std::vector<double> forces;
    std::vector<int> islocked;

    gatherNames(ion_names, root, mmpi.commSpin());
    gatherPositions(positions, root);
    gatherForces(forces, root);
    gatherLockedData(islocked, root);

    if (mmpi.mypeGlobal() == root)
    {
        os << "IONIC POSITIONS AND FORCES:" << std::endl;

        os << std::setiosflags(std::ios::left) << std::setw(8) << "Atoms"
           << resetiosflags(std::ios::left) << std::setw(8) << "X"
           << std::setw(10) << "Y" << std::setw(10) << "Z" << std::setw(10)
           << "FX" << std::setw(10) << "FY" << std::setw(10) << "FZ"
           << std::endl;

        int ion_index                                     = 0;
        std::map<int, std::string>::const_iterator ion_id = ion_names.begin();
        while (ion_id != ion_names.end())
        {

            const int pos = 3 * ion_index;
            os << "## ";
            if (islocked[ion_index])
                os << "*";
            else
                os << " ";
            os << std::setw(4) << ion_id->second
               << std::setiosflags(std::ios::right) << std::setw(10)
               << std::setprecision(4) << std::fixed << positions[pos]
               << std::setw(10) << positions[pos + 1] << std::setw(10)
               << positions[pos + 2] << std::setprecision(7) << std::scientific
               << std::setw(16) << forces[pos] << std::setw(16)
               << forces[pos + 1] << std::setw(16) << forces[pos + 2]
               << std::endl;

            if (!islocked[ion_index])
            {

                avfx += fabs(forces[pos]);
                avfy += fabs(forces[pos + 1]);
                avfz += fabs(forces[pos + 2]);

                double ff = forces[pos] * forces[pos]
                            + forces[pos + 1] * forces[pos + 1]
                            + forces[pos + 2] * forces[pos + 2];

                maxf = std::max(maxf, ff);

                maxfx = std::max(maxfx, fabs(forces[pos]));
                maxfy = std::max(maxfy, fabs(forces[pos + 1]));
                maxfz = std::max(maxfz, fabs(forces[pos + 2]));

                num_movable++;
            }

            for (short ii = 0; ii < 3; ii++)
                sum_forces[ii] += forces[pos + ii];

            num_atoms++;
            ion_index++;
            ion_id++;
        }

        if (num_atoms == 0) return;
        // global statistics
        os << std::endl << "Global Statistics:" << std::endl;
        os << "==========================" << std::endl;

        const double inv_num_atoms = 1. / (double)num_atoms;
        for (short ii = 0; ii < 3; ii++)
            sum_forces[ii] *= inv_num_atoms;

        if (num_movable)
        {
            double inv_num_movable = 1. / (double)num_movable;
            avfx                   = avfx * inv_num_movable;
            avfy                   = avfy * inv_num_movable;
            avfz                   = avfz * inv_num_movable;

            os << std::scientific;
            os << " mean F on movable ions  = (" << avfx << "," << avfy << ","
               << avfz << ")" << std::endl;
            os << " max F on movable ions   = (" << maxfx << "," << maxfy << ","
               << maxfz << ")" << std::endl;
            os << " max |F| on movable ions = " << sqrt(maxf) << std::endl;
        }
        os << " Sum forces on all ions  = (" << sum_forces[0] << ","
           << sum_forces[1] << "," << sum_forces[2] << ")" << std::endl;
        os << std::fixed;
    }
}

// Writes out the postions of the local ions and the current forces on them
void Ions::printForcesLocal(std::ostream& os, const int root) const
{
    int num_atoms   = 0;
    int num_movable = 0;

    double buf[10];
    for (int i = 0; i < 10; i++)
        buf[i] = 0.;

    double* sum_forces = &buf[0];
    double* avg_forces = sum_forces + 3;
    double* max_forces = avg_forces + 3;
    double* maxf       = max_forces + 3;

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.barrier();

    if (mmpi.mypeGlobal() == root)
    {
        os << std::endl
           << std::endl
           << "IONIC POSITIONS AND FORCES ON CENTERED ON PE" << root << ":"
           << std::endl;

        os << std::setiosflags(std::ios::left) << std::setw(8) << "Atoms"
           << std::resetiosflags(std::ios::left) << std::setw(8) << "X"
           << std::setw(10) << "Y" << std::setw(10) << "Z" << std::setw(10)
           << "FX" << std::setw(10) << "FY" << std::setw(10) << "FZ"
           << std::endl;

        for (auto& ion : local_ions_)
        {
            ion->printPositionAndForce(os);

            if (!ion->locked())
            {
                avg_forces[0] += fabs(ion->force(0));
                avg_forces[1] += fabs(ion->force(1));
                avg_forces[2] += fabs(ion->force(2));

                double ff = ion->norm2F();
                maxf[0]   = std::max(maxf[0], ff);

                max_forces[0] = std::max(max_forces[0], fabs(ion->force(0)));
                max_forces[1] = std::max(max_forces[1], fabs(ion->force(1)));
                max_forces[2] = std::max(max_forces[2], fabs(ion->force(2)));

                num_movable++;
            }

            for (short ii = 0; ii < 3; ii++)
                sum_forces[ii] += ion->force(ii);
            num_atoms++;
        }
    }
    else
    {
        for (auto& ion : local_ions_)
        {
            if (!ion->locked())
            {
                avg_forces[0] += fabs(ion->force(0));
                avg_forces[1] += fabs(ion->force(1));
                avg_forces[2] += fabs(ion->force(2));

                double ff = ion->norm2F();
                maxf[0]   = std::max(maxf[0], ff);

                max_forces[0] = std::max(max_forces[0], fabs(ion->force(0)));
                max_forces[1] = std::max(max_forces[1], fabs(ion->force(1)));
                max_forces[2] = std::max(max_forces[2], fabs(ion->force(2)));

                num_movable++;
            }

            for (short ii = 0; ii < 3; ii++)
                sum_forces[ii] += ion->force(ii);
            num_atoms++;
        }
    }
    // global statistics
    mmpi.allreduce(&num_atoms, 1, MPI_SUM);
    if (num_atoms == 0) return;
    if (onpe0)
    {
        os << std::endl << "Global Statistics:" << std::endl;
        os << "==========================" << std::endl;
    }

    mmpi.allreduce(&num_movable, 1, MPI_SUM);
    mmpi.allreduce(&buf[0], 6, MPI_SUM);
    mmpi.allreduce(&buf[6], 4, MPI_MAX);

    const double inv_num_atoms = 1. / (double)num_atoms;
    for (short ii = 0; ii < 3; ii++)
        sum_forces[ii] *= inv_num_atoms;

    if (mmpi.mypeGlobal() == root)
    {
        if (num_movable)
        {
            double inv_num_movable = 1. / (double)num_movable;
            avg_forces[0]          = avg_forces[0] * inv_num_movable;
            avg_forces[1]          = avg_forces[1] * inv_num_movable;
            avg_forces[2]          = avg_forces[2] * inv_num_movable;

            os << std::scientific;
            os << " mean F on movable ions  = (" << avg_forces[0] << ","
               << avg_forces[1] << "," << avg_forces[2] << ")" << std::endl;
            os << " max F on movable ions   = (" << max_forces[0] << ","
               << max_forces[1] << "," << max_forces[2] << ")" << std::endl;
            os << " max |F| on movable ions = " << sqrt(maxf[0]) << std::endl;
        }
        os << " Sum forces on all ions  = (" << sum_forces[0] << ","
           << sum_forces[1] << "," << sum_forces[2] << ")" << std::endl;
        os << std::fixed;
    }
}

// Writes out the postions of the ions and the current forces on them
void Ions::printForces(std::ostream& os, const int root) const
{

    Control& ct(*(Control::instance()));

    if (num_ions_ < 512 || ct.verbose > 2)
        printForcesGlobal(os, root);
    else
        printForcesLocal(os, root);
}

int Ions::countIonsHere() const { return (int)local_ions_.size(); }

int Ions::countProjectorsHere() const
{
    int count = 0;
    for (auto& ion : local_ions_)
    {
        count += ion->nProjectors();
    }
    return count;
}

int Ions::countProjectors() const
{
    //    assert( setup_ );
    // std::cout<<"Num. local ions: "<<local_ions_.size()<<endl;

    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    int nproj = 0;
    for (auto& ion : local_ions_)
    {
        nproj += ion->nProjectors();
    }
    int tmp = nproj;
    MPI_Allreduce(&tmp, &nproj, 1, MPI_INT, MPI_SUM, myPEenv.comm());

    return nproj;
}

int Ions::countProjectorsSubdomain() const
{
    int nproj                             = 0;
    std::vector<Ion*>::const_iterator ion = overlappingNL_ions_.begin();
    while (ion != overlappingNL_ions_.end())
    {
        nproj += (*ion)->nProjectorsSubdomain();
        ion++;
    }
    return nproj;
}

Ion* Ions::findIon(const std::string& name) const
{
    std::vector<Ion*>::const_iterator ion = list_ions_.begin();
    while (ion != list_ions_.end())
    {
        bool same_name = (*ion)->compareName(name);
        if (same_name) return (*ion);

        ion++;
    }

    // return 0 if name not found in list_ions_
    return nullptr;
}

bool Ions::isLocal(const std::string& name) const
{
    for (auto ion : local_ions_)
    {
        bool same_name = ion->compareName(name);
        if (same_name) return true;
    }
    return false;
}

Ion* Ions::findIon(const int index) const
{
    for (auto ion : list_ions_)
    {
        if (ion->compareIndex(index)) return ion;
    }
    return nullptr;
}

Ion* Ions::findLocalIon(const int index) const
{
    for (auto ion : local_ions_)
    {
        if (ion->compareIndex(index)) return ion;
    }
    return nullptr;
}

void Ions::setLocalPositions(const std::vector<double>& positions)
{
    assert(positions.size() == 3 * local_ions_.size());

    int ia = 0;
    for (auto& ion : local_ions_)
    {
        ion->setPosition(positions[3 * ia + 0], positions[3 * ia + 1],
            positions[3 * ia + 2]);
        ia++;
    }

    setup_ = false;
}

int Ions::readAtoms(const std::string& filename, const bool cell_relative)
{

    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Ions::readAtoms() --- Read "
                         << " atomic positions from file " << filename
                         << std::endl;

    std::string strxyz(".xyz");
    size_t found = filename.find(strxyz);
    if (found != std::string::npos)
    {
        num_ions_ = readAtomsFromXYZ(filename, cell_relative);
    }
    else
    {
        num_ions_ = readNatoms(filename, cell_relative);
    }

    return num_ions_;
}

int Ions::readAtoms(std::ifstream* tfile, const bool cell_relative)
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    std::string first_string_read = "";
    if (mmpi.PE0())
    {
        (*tfile) >> first_string_read;
    }
    mmpi.bcastGlobal(first_string_read);

    std::string strxyz(".xyz");
    size_t found = first_string_read.find(strxyz);
    if (found != std::string::npos)
    {
        if (mmpi.PE0())
            (*MPIdata::sout) << "Read atomic positions from file "
                             << first_string_read << std::endl;
        num_ions_ = readAtomsFromXYZ(first_string_read, cell_relative);
    }
    else
    {
        if (mmpi.PE0())
        {
            for (unsigned short i = 0; i < first_string_read.size(); i++)
                tfile->unget();
        }
        num_ions_ = readNatoms(tfile, cell_relative);
    }

    return num_ions_;
}

int Ions::readAtomsFromXYZ(
    const std::string& filename, const bool cell_relative)
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    // set up list boundaries
    // get radius of projectors
    double rmax = getMaxListRadius();
    if (onpe0)
        (*MPIdata::sout) << "Max. radius for species in XYZ file: " << rmax
                         << std::endl;
    assert(rmax > 0.);

    setupListIonsBoundaries(rmax);

    int natoms = -1;

    std::ifstream* tfile = nullptr;
    if (mmpi.PE0())
    {
        tfile = new std::ifstream(filename.data(), std::ios::in);
        if (!(*tfile))
        {
            (*MPIdata::serr)
                << "Ions::readAtomsFromXYZ() --- ERROR: cannot open "
                << filename << std::endl;
        }
        else
        {
            std::string query;
            if (getline(*tfile, query))
            {
                std::stringstream na(query);
                na >> natoms;
            }
        }
        if (natoms < 1)
            (*MPIdata::sout)
                << "WARNING: Ions::readAtomsFromXYZ(), number of atoms read = "
                << natoms << std::endl;
    }
    mmpi.bcastGlobal(&natoms);
    if (natoms < 0) return natoms;

    std::vector<double> crds(3 * natoms);
    std::vector<short> spec(natoms);

    int count = 0;
    if (mmpi.PE0())
    {
        std::string query;
        getline(*tfile, query); // read comment line
        // read atomic species and positions
        for (int ia = 0; ia < natoms; ++ia)
        {
            if (!getline(*tfile, query))
            {
                break;
            }
            std::stringstream ss(query);
            std::string name_read;
            ss >> name_read;
            spec[ia] = map_species_.find(name_read)->second;
            for (int j = 0; j < 3; j++)
            {
                ss >> crds[3 * ia + j];
                crds[3 * ia + j] *= ang2bohr;
                assert(crds[3 * ia + j] > -100000.);
                assert(crds[3 * ia + j] < 100000.);
            }
            if (cell_relative)
            {
                crds[3 * ia + 0] *= lattice_[0];
                crds[3 * ia + 1] *= lattice_[1];
                crds[3 * ia + 2] *= lattice_[2];
            }
            ++count;
        }
        delete tfile;
    }

    mmpi.bcastGlobal(&count, 1);
    if (count < natoms) return count;

    mmpi.bcastGlobal(&crds[0], 3 * natoms);
    mmpi.bcastGlobal(&spec[0], natoms);

    return setAtoms(crds, spec);
}

int Ions::setAtoms(
    const std::vector<double>& crds, const std::vector<short>& anum)
{
    // we are setting up a new list of atoms and need an index starting at 0
    Ion::resetIndexCount();
    const int natoms = crds.size() / 3;

    double velocity[3] = { 0., 0., 0. };
    bool locked        = false;
    for (int ia = 0; ia < natoms; ++ia)
    {
        std::vector<Species>::const_iterator it = species_.begin();
        int isp                                 = -1;
        while (it != species_.end())
        {
            ++isp;
            if (it->getAtomicNumber() == anum[ia])
            {
                break;
            }
            ++it;
        }
        std::string spname("");
        for (std::map<std::string, short>::iterator itr = map_species_.begin();
             itr != map_species_.end(); ++itr)
        {
            if (itr->second == anum[ia])
            {
                spname = itr->first;
                break;
            }
        }
        if (spname.compare("") == 0)
        {
            (*MPIdata::serr) << "Ions::setAtoms() --- ERROR: unknown "
                                "species for atomic number "
                             << anum[ia] << std::endl;
            return -1;
        }

        // make a name for atom based on species and order of reading in
        std::string aname(spname);
        std::stringstream ss;
        ss << ia;
        if (ia < 10) aname.append("0");
        if (ia < 100) aname.append("0");
        if (ia < 1000) aname.append("0");
        if (ia < 10000) aname.append("0");
        aname.append(ss.str());

        addIonToList(species_[isp], aname, &crds[3 * ia], velocity, locked);
    }

    return natoms;
}

void Ions::addIonToList(const Species& sp, const std::string& name,
    const double crds[3], const double velocity[3], const bool locked)
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    Control& ct(*(Control::instance()));

    // create a new Ion
    Ion* new_ion = new Ion(sp, name, &crds[0], velocity, locked);
    // std::cout << "New Ion index = " << new_ion->index() << std::endl;
    new_ion->bcast(mmpi.commGlobal());

    if (inListIons(crds[0], crds[1], crds[2]))
    {
        list_ions_.push_back(new_ion);
        if (ct.verbose > 2)
            (*MPIdata::sout)
                << "Ion " << name << " at position " << crds[0] << ","
                << crds[1] << "," << crds[2] << " added to the list... on PE"
                << mmpi.mypeGlobal() << std::endl;

        // populate local_ions_ list
        if (inLocalIons(crds[0], crds[1], crds[2]))
        {
            (new_ion)->set_here(true);
            local_ions_.push_back(new_ion);
            if (onpe0 && ct.verbose > 2)
                (*MPIdata::sout) << "Ion " << name << " at position " << crds[0]
                                 << "," << crds[1] << "," << crds[2]
                                 << " added to the list of local ions... on PE"
                                 << mmpi.mypeGlobal() << std::endl;
        }
        else
            (new_ion)->set_here(false);
    }
    else
    {
        // delete Ion if not put in list
        delete new_ion;
    }
}

int Ions::readNatoms(const std::string& filename, const bool cell_relative)
{
    Control& ct(*(Control::instance()));

    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Ions::readNAtoms() --- Read "
                         << " atomic positions from file " << filename
                         << std::endl;

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    std::ifstream* tfile = nullptr;
    if (mmpi.instancePE0())
    {
        tfile = new std::ifstream(filename.data(), std::ios::in);
        if (!tfile->is_open())
        {
            (*MPIdata::serr)
                << " Unable to open file " << filename.data() << std::endl;
            return -1;
        }
        else
        {
            (*MPIdata::sout) << "Open " << filename.data() << std::endl;
        }
    }

    int nread = readNatoms(tfile, cell_relative);

    if (mmpi.instancePE0())
    {
        if (ct.verbose > 0)
            (*MPIdata::sout) << "Close " << filename.data() << std::endl;
        tfile->close();
        delete tfile;
    }

    return nread;
}

int Ions::readNatoms(std::ifstream* tfile, const bool cell_relative)
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    if (mmpi.PE0()) assert(tfile != nullptr);

    // set pointer to species object
    // set up list boundaries
    // get radius of projectors
    double rmax = getMaxListRadius();
    assert(rmax > 0.);
    setupListIonsBoundaries(rmax);

    // make sure all processors call this function
    mmpi.barrier();

    Control& ct(*(Control::instance()));
    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Ions::readNatoms() --- Try to read "
                         << " atomic positions..." << std::endl;
    int nread = 0;
    int iread = 1;
    while (iread > 0)
    {
        iread = read1atom(tfile, cell_relative);
        // if(onpe0)cout<<"iread="<<iread<<endl;
        if (iread < 0) return -100 - nread;
        nread += iread;
        if (onpe0 && ct.verbose > 0 && (nread % 1000 == 0))
            (*MPIdata::sout) << "Ions::readNatoms() --- read " << nread
                             << " atomic positions..." << std::endl;
    }
    if (onpe0) read_comments(*tfile);

    if (onpe0 && ct.verbose > 0)
        (*MPIdata::sout) << "Ions::readNatoms() --- read " << nread
                         << " atomic positions..." << std::endl;

    int siz = local_ions_.size();
    mmpi.allreduce(&siz, 1, MPI_SUM);
    assert(siz == nread);

    return nread;
}

// return number of atoms read, 0 if end of file, or -1 if failure occurs
int Ions::read1atom(std::ifstream* tfile, const bool cell_relative)
{
    std::string name_read = "";
    // short  isp=0;
    double crds[3];
    double velocity[3] = { 0., 0., 0. };

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    short movable = 0;
    std::string query;
    short count = 1;
    if (mmpi.PE0())
    {
        if (getline(*tfile, query))
        {
            if (query[0] == '#') count = 0; // comment
            if (query.empty()) count = 0; // end of line only
            if (query.size() < 3) count = 0; // white spaces only
            // std::cout<<"length="<<query.size()<<endl;
        }
        else
        {
            if (tfile->eof())
            {
                //(*MPIdata::serr)<<"Ions::read1atom() --- eof..."<<endl;
                count = 0;
            }
            else
            {
                (*MPIdata::serr)
                    << "ERROR: Ions::read1atom() --- getline() failed..."
                    << std::endl;
                count = -1;
            }
        }
        // std::cout<<"query="<<query<<endl;
        // std::cout<<"count="<<count<<endl;
    }

    mmpi.bcastGlobal(&count, 1);
    if (count < 1) return count;

    if (mmpi.PE0())
    {
        std::stringstream ss(query);
        ss >> name_read;
        if (!checkValidName(name_read))
        {
            std::cerr << "ERROR: Invalid name read in input file: " << name_read
                      << std::endl;
            mmpi.abort();
        }
        short dummy;
        ss >> dummy; // not used anymore (was species index)

        for (short j = 0; j < 3; j++)
        {
            ss >> crds[j];
            assert(crds[j] > -10000.);
            assert(crds[j] < 10000.);
        }

        if (cell_relative)
        {
            crds[0] *= lattice_[0];
            crds[1] *= lattice_[1];
            crds[2] *= lattice_[2];
        }

        if (!(ss >> movable)) movable = 1; // default value
        if (movable != 0 && movable != 1)
        {
            (*MPIdata::serr)
                << "Atom " << name_read << ", should be movable (1) or not(0)"
                << std::endl;
            return -1;
        }
#ifdef DEBUG
        (*MPIdata::sout) << "movable=" << movable << std::endl;
#endif
        int j = 0;
        if (movable)
        {
            while (ss >> velocity[j++])
                ;
        }

#ifndef NDEBUG
        (*MPIdata::sout) << "Ions::read1atom() --- Read Ion in position ("
                         << crds[0] << "," << crds[1] << "," << crds[2] << ")"
                         << " and velocity :(" << velocity[0] << ","
                         << velocity[1] << "," << velocity[2] << ")"
                         << std::endl;
#endif

    } // if onpe0

    double tmp[7] = { crds[0], crds[1], crds[2], velocity[0], velocity[1],
        velocity[2], (double)movable };
    mmpi.bcastGlobal(tmp, 7);

    crds[0]     = tmp[0];
    crds[1]     = tmp[1];
    crds[2]     = tmp[2];
    velocity[0] = tmp[3];
    velocity[1] = tmp[4];
    velocity[2] = tmp[5];

    movable = (int)tmp[6];

    mmpi.bcastGlobal(name_read);

    // find species based on name
    std::string name(name_read);
    stripName(name);

    auto search = map_species_.find(name);
    assert(search != map_species_.end());

    short spec_nb                           = search->second;
    std::vector<Species>::const_iterator it = species_.begin();
    int isp                                 = -1;
    while (it != species_.end())
    {
        ++isp;
        if (it->getAtomicNumber() == spec_nb)
        {
            break;
        }
        ++it;
    }

    bool locked = (!movable);
    // if(isp>=(int)species_.size())
    //{
    //    (*MPIdata::serr)<<"name="<<name<<", isp="<<isp<<" >= num. species ("
    //                    <<species_.size()<<")"
    //        <<endl;
    //    return -1;
    //}

#ifdef DEBUG
    if (onpe0) (*MPIdata::sout) << "Create new Ion..." << std::endl;
#endif

    // Ion needs to be created by each MPI task to set global ids
    Ion* new_ion = new Ion(species_[isp], name_read, crds, velocity, locked);

#ifdef DEBUG
    if (onpe0) (*MPIdata::sout) << "Ion read..." << std::endl;
#endif

    // Populate list_ions_ list
    if (inListIons(crds[0], crds[1], crds[2]))
    {
        list_ions_.push_back(new_ion);
        // populate local_ions_ list
        if (inLocalIons(crds[0], crds[1], crds[2]))
        {
            new_ion->set_here(true);
            local_ions_.push_back(new_ion);
        }
        else
            new_ion->set_here(false);
    }
    else
    {
        delete new_ion;
    }

    return 1;
}

int Ions::getNValenceElectrons() const
{
    double val = 0.;
    for (auto ion : local_ions_)
    {
        val += ion->getZion();
    }
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&val, 1, MPI_SUM);

    return (int)val;
}

double Ions::computeIonicCharge() const
{
    double ionic_charge = 0.;
    for (auto ion : local_ions_)
    {
        ionic_charge += (double)ion->getZion();
    }
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&ionic_charge, 1, MPI_SUM);
    return ionic_charge;
}

void Ions::setVelocities(const std::vector<double>& tau0,
    const std::vector<double>& taup, const double dt)
{
    assert(tau0.size() == 3 * local_ions_.size());
    assert(taup.size() == 3 * local_ions_.size());

    int ia = 0;
    for (auto& ion : local_ions_)
    {
        double v[3];
        for (short i = 0; i < 3; i++)
        {
            v[i] = taup[3 * ia + i];
            v[i] -= tau0[3 * ia + i];
            v[i] /= dt;
        }
        ion->setVelocity(v[0], v[1], v[2]);
        ia++;
    }
}

void Ions::getLocalPositions(std::vector<double>& pos) const
{
    assert(pos.size() == 3 * local_ions_.size());

    int ia = 0;
    for (auto& ion : local_ions_)
    {
        ion->getPosition(&pos[3 * ia]);
        ia++;
    }
}

void Ions::getLocalNames(std::vector<std::string>& names) const
{
    names.clear();
    for (const auto& ion : local_ions_)
    {
        names.push_back(ion->name());
    }
}

void Ions::getNames(std::vector<std::string>& names) const
{
    std::vector<std::string> local_names;
    getLocalNames(local_names);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherV(local_names, names);
}

void Ions::getPositions(std::vector<double>& pos) const
{
    std::vector<double> pos_local(3 * local_ions_.size());

    getLocalPositions(pos_local);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherV(pos_local, pos);
}

void Ions::getAtomicNumbers(std::vector<short>& atnumbers) const
{
    std::vector<short> local_atnumbers;
    for (auto& ion : local_ions_)
    {
        local_atnumbers.push_back(ion->atomic_number());
    }

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherV(local_atnumbers, atnumbers);
}

void Ions::getForces(std::vector<double>& forces) const
{
    std::vector<double> forces_local(3 * local_ions_.size());
    getLocalForces(forces_local);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allGatherV(forces_local, forces);
}

void Ions::setTau0()
{
    assert(tau0_.size() == 3 * local_ions_.size());

    int ia = 0;
    for (auto& ion : local_ions_)
    {
        ion->getPosition(&tau0_[3 * ia]);
        ia++;
    }
}

void Ions::setPositionsToTau0()
{
    assert(tau0_.size() == 3 * local_ions_.size());

    int ia = 0;
    for (auto& ion : local_ions_)
    {
        ion->setPosition(
            tau0_[3 * ia + 0], tau0_[3 * ia + 1], tau0_[3 * ia + 2]);
        ia++;
    }
}

void Ions::setPositions(
    const std::vector<double>& p, const std::vector<short>& anumbers)
{
    assert(p.size() == anumbers.size() * 3);

    // clear previous data
    clearLists();

    num_ions_ = setAtoms(p, anumbers);

    // setup required after updating local ions positions
    setup();
}

void Ions::setVelocitiesToVel()
{
    assert(velocity_.size() == 3 * local_ions_.size());

    int ia = 0;
    for (auto& ion : local_ions_)
    {
        ion->setVelocity(velocity_[3 * ia + 0], velocity_[3 * ia + 1],
            velocity_[3 * ia + 2]);
        ia++;
    }
}

void Ions::getLocalForces(std::vector<double>& f) const
{
    assert(f.size() == 3 * local_ions_.size());

    int ia = 0;
    for (const auto& ion : local_ions_)
    {
        assert(3 * ia + 2 < (int)f.size());
        ion->getForce(&f[3 * ia]);
        ia++;
    }
}

void Ions::setMapVL()
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    const double h0 = mygrid.hgrid(0);
    const double h1 = mygrid.hgrid(1);
    const double h2 = mygrid.hgrid(2);

    const double origin[3]
        = { mygrid.origin(0), mygrid.origin(1), mygrid.origin(2) };

    const int dim0  = mygrid.dim(0);
    const int dim1  = mygrid.dim(1);
    const int dim2  = mygrid.dim(2);
    const int gdim0 = mygrid.gdim(0);
    const int gdim1 = mygrid.gdim(1);
    const int gdim2 = mygrid.gdim(2);

    const int ilow = mygrid.istart(0);
    const int jlow = mygrid.istart(1);
    const int klow = mygrid.istart(2);
    const int ihi  = ilow + dim0 - 1;
    const int jhi  = jlow + dim1 - 1;
    const int khi  = klow + dim2 - 1;

    // Loop over list ions
    std::vector<Ion*>::iterator ion = list_ions_.begin();
    while (ion != list_ions_.end())
    {
        /* Generate range of indices over which the short-range difference */
        /* potential will be mapped onto the global grid.                  */
        (*ion)->set_lstart(0, origin[0], h0);
        (*ion)->set_lstart(1, origin[1], h1);
        (*ion)->set_lstart(2, origin[2], h2);

        // Generate indices
        std::vector<int> Ai0;
        (*ion)->get_Ai(Ai0, gdim0, 0);
        const int dimlx = Ai0.size();

        (*ion)->set_map_l(false);

        bool map0 = false;
        for (int idx = 0; idx < dimlx; idx++)
        {
            if ((Ai0[idx] >= ilow) && (Ai0[idx] <= ihi))
            {
                map0 = true;
                break;
            }
        }

        if (map0)
        {
            bool map1 = false;
            std::vector<int> Ai1;
            (*ion)->get_Ai(Ai1, gdim1, 1);
            const int dimly = Ai1.size();
            for (int idx = 0; idx < dimly; idx++)
            {
                if ((Ai1[idx] >= jlow) && (Ai1[idx] <= jhi))
                {
                    map1 = true;
                    break;
                }
            }
            if (map1)
            {
                bool map2 = false;
                std::vector<int> Ai2;
                (*ion)->get_Ai(Ai2, gdim2, 2);
                const int dimlz = Ai2.size();
                for (int idx = 0; idx < dimlz; idx++)
                {
                    if ((Ai2[idx] >= klow) && (Ai2[idx] <= khi))
                    {
                        map2 = true;
                        break;
                    }
                }

                if (map2)
                {
                    (*ion)->set_map_l(true);
                }
            }
        }
        ion++;
    }
}

double Ions::computeMaxVlRadius() const
{
    double radius = 0.;

    for (auto& ion : local_ions_)
    {
        double r = ion->computeRadiusVl();
        radius   = r > radius ? r : radius;
    }

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&radius, 1, MPI_MAX);

    assert(radius >= 0.);

    return radius;
}

double Ions::computeMaxNLprojRadius() const
{
    double radius = 0.;

    for (auto& iion : local_ions_)
    {
        double r = iion->radiusNLproj();
        radius   = r > radius ? r : radius;
    }

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&radius, 1, MPI_MAX);

    assert(radius >= 0.);

    return radius;
}

void Ions::gatherNames(std::map<int, std::string>& names, const int root,
    const MPI_Comm comm) const
{
    std::vector<int> indexes(num_ions_, 0);
    std::vector<int> local_indexes;
    std::vector<std::string> data;
    std::vector<std::string> local_names;

    for (auto& ion : local_ions_)
    {
        // get local name and index
        local_names.push_back(ion->name());
        local_indexes.push_back(ion->index());
    }

    // gather data to PE root
    mgmol_tools::gatherV(local_names, data, root, comm);
    mgmol_tools::gatherV(local_indexes, indexes, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    if (mype == root)
    {
        const unsigned int num_ions = data.size();
        assert(num_ions == indexes.size());
        for (unsigned int i = 0; i < num_ions; i++)
        {
            const int ion_index        = indexes[i];
            const std::string ion_name = data[i];
            names.insert(std::pair<int, std::string>(ion_index, ion_name));
        }
    }
}

void Ions::gatherNames(
    std::vector<std::string>& names, const int root, const MPI_Comm comm) const
{
    std::vector<std::string> local_names;

    for (auto& ion : local_ions_)
    {
        local_names.push_back(ion->name());
    }

    // gather data to PE root
    std::vector<std::string> data;
    mgmol_tools::gatherV(local_names, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    names.clear();
    if (mype == root) names = data;
}

void Ions::gatherLockedNames(
    std::vector<std::string>& names, const int root, const MPI_Comm comm) const
{
    std::vector<std::string> local_names;
    for (auto& ion : local_ions_)
    {
        if (ion->locked()) local_names.push_back(ion->name());
    }

    // gather data to PE root
    std::vector<std::string> data;
    mgmol_tools::gatherV(local_names, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    names.clear();
    if (mype == root) names = data;
}

void Ions::gatherIndexes(
    std::vector<int>& indexes, const int root, const MPI_Comm comm) const
{
    std::vector<int> local_indexes;
    for (auto& ion : local_ions_)
    {
        local_indexes.push_back(ion->index());
    }

    // gather data to PE root
    std::vector<int> data;
    mgmol_tools::gatherV(local_indexes, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    indexes.clear();
    if (mype == root) indexes = data;
}

void Ions::gatherNLprojIds(
    std::vector<int>& nlprojids, const int root, const MPI_Comm comm) const
{
    std::vector<int> local_nlprojids;
    for (auto& ion : local_ions_)
    {
        local_nlprojids.push_back(ion->nlprojid());
    }

    // gather data to PE root
    std::vector<int> data;
    mgmol_tools::gatherV(local_nlprojids, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    nlprojids.clear();
    if (mype == root) nlprojids = data;
}

void Ions::gatherAtomicNumbers(
    std::vector<int>& atnumbers, const int root, const MPI_Comm comm) const
{
    std::vector<int> local_atnumbers;
    for (auto& ion : local_ions_)
    {
        local_atnumbers.push_back(ion->atomic_number());
    }

    // gather data to PE root
    std::vector<int> data;
    mgmol_tools::gatherV(local_atnumbers, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    atnumbers.clear();
    if (mype == root) atnumbers = data;
}

void Ions::gatherRandStates(std::vector<unsigned short>& rstates,
    const int root, const MPI_Comm comm) const
{
    std::vector<unsigned short> local_rstates;
    for (auto& ion : local_ions_)
    {
        local_rstates.push_back(ion->randomState(0));
        local_rstates.push_back(ion->randomState(1));
        local_rstates.push_back(ion->randomState(2));
    }

    // gather data to PE root
    std::vector<unsigned short> data;
    mgmol_tools::gatherV(local_rstates, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    rstates.clear();
    if (mype == root) rstates = data;
}

void Ions::gatherPositions(
    std::vector<double>& positions, const int root, const MPI_Comm comm) const
{
    std::vector<double> local_positions;
    for (auto& ion : local_ions_)
    {
        double position[3];
        ion->getPosition(&position[0]);
        local_positions.push_back(position[0]);
        local_positions.push_back(position[1]);
        local_positions.push_back(position[2]);
    }

    // gather data to PE root
    std::vector<double> data;
    mgmol_tools::gatherV(local_positions, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    positions.clear();
    if (mype == root) positions = data;
}

void Ions::gatherPreviousPositions(
    std::vector<double>& positions, const int root, const MPI_Comm comm) const
{
    std::vector<double> local_positions;
    for (auto& ion : local_ions_)
    {
        local_positions.push_back(ion->getPreviousPosition(0));
        local_positions.push_back(ion->getPreviousPosition(1));
        local_positions.push_back(ion->getPreviousPosition(2));
    }

    // gather data to PE root
    std::vector<double> data;
    mgmol_tools::gatherV(local_positions, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    positions.clear();
    if (mype == root) positions = data;
}

void Ions::gatherForces(
    std::vector<double>& forces, const int root, const MPI_Comm comm) const
{
    std::vector<double> local_forces;
    for (auto& ion : local_ions_)
    {
        double force[3];
        ion->getForce(&force[0]);
        local_forces.push_back(force[0]);
        local_forces.push_back(force[1]);
        local_forces.push_back(force[2]);
    }
    // gather data to PE root
    std::vector<double> data;
    mgmol_tools::gatherV(local_forces, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    forces.clear();
    if (mype == root) forces = data;
}

void Ions::gatherVelocities(
    std::vector<double>& velocities, const int root, const MPI_Comm comm) const
{
    std::vector<double> local_velocities;
    for (auto& ion : local_ions_)
    {
        local_velocities.push_back(ion->velocity(0));
        local_velocities.push_back(ion->velocity(1));
        local_velocities.push_back(ion->velocity(2));
    }

    // gather data to PE root
    std::vector<double> data;
    mgmol_tools::gatherV(local_velocities, data, root, comm);

    int mype = 0;
    MPI_Comm_rank(comm, &mype);
    velocities.clear();
    if (mype == root) velocities = data;
}

void Ions::gatherPositions(std::vector<double>& positions, const int root) const
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    positions.resize(3 * num_ions_, 0.);

    for (auto& ion : local_ions_)
    {
        const int index = ion->index();
        ion->getPosition(&positions[3 * index]);
    }

    // gather data to PE root
    mmpi.reduce(&positions[0], 3 * num_ions_, MPI_SUM, root);
}

void Ions::gatherForces(std::vector<double>& forces, const int root) const
{
    assert(num_ions_ > 0);

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    forces.resize(3 * num_ions_, 0.);
    for (auto& ion : local_ions_)
    {
        const int index = ion->index();
        // std::cout << "index = " << index << std::endl;
        assert(index < num_ions_);
        assert((int)forces.size() >= 3 * index);
        assert(index < num_ions_);
        ion->getForce(&forces[3 * index]);
    }

    // gather data to PE root
    const int size = 3 * num_ions_;
    mmpi.reduce(&forces[0], size, MPI_SUM, root);
}

void Ions::gatherLockedData(std::vector<int>& locked_data, const int root) const
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    locked_data.resize(num_ions_, 0);
    for (auto& ion : local_ions_)
    {
        const int index = ion->index();
        if (ion->locked()) locked_data[index] = 1;
    }

    // gather data to PE root
    mmpi.reduce(&locked_data[0], num_ions_, MPI_SUM, root);
}

bool Ions::hasLockedAtoms() const
{
    short flag = 0;

    std::vector<Ion*>::const_iterator ion = local_ions_.begin();
    while (ion != local_ions().end())
    {
        if ((*ion)->locked())
        {
            flag = 1;
            break;
        }
        ++ion;
    }

    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&flag, 1, MPI_MAX);

    return (flag == 1);
}

double Ions::getSpeciesMaxNLradius() const
{
    double radius = 0.;
    for (const auto& spi : species_)
    {
        const double nlradius = spi.nlradius();
        radius                = radius > nlradius ? radius : nlradius;
    }
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&radius, 1, MPI_MAX);

    return radius;
}

double Ions::getSpeciesMaxLradius() const
{
    double radius = 0.;
    for (const auto& spi : species_)
    {
        const double lradius = spi.lradius();
        radius               = radius > lradius ? radius : lradius;
    }
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    mmpi.allreduce(&radius, 1, MPI_MAX);

    return radius;
}

void Ions::setupListIonsBoundaries(const double rmax)
{
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    const pb::Grid& mygrid   = mymesh->grid();

    double offset[3];
    for (short i = 0; i < 3; i++)
        offset[i] = (double)myPEenv.my_mpi(i) * div_lattice_[i];

    const double origin[3]
        = { mygrid.origin(0), mygrid.origin(1), mygrid.origin(2) };
    // const double end[3]    =
    // {origin[0]+lattice_[0],origin[1]+lattice_[1],origin[2]+lattice_[2]};

    // define extents for gathering ion data
    // define extents to be at least size of local domain and at most the size
    // of global domain
    double myrmax[3] = { rmax, rmax, rmax };
    for (short i = 0; i < 3; i++)
    {
        myrmax[i] = myrmax[i] >= div_lattice_[i] ? myrmax[i] : div_lattice_[i];
        myrmax[i] = myrmax[i] <= lattice_[i] ? myrmax[i] : lattice_[i];
    }
    // left / lower extent
    for (short i = 0; i < 3; i++)
    {
        list_boundary_left_[i] = origin[i] + offset[i] - myrmax[i];
    }

    // right/ upper extent
    for (short i = 0; i < 3; i++)
    {
        list_boundary_right_[i]
            = origin[i] + offset[i] + div_lattice_[i] + myrmax[i];
    }

    // setup and save data communication info
    /* get processor distribution */
    int nproc_xyz[3];
    nproc_xyz[0] = myPEenv.n_mpi_task(0);
    nproc_xyz[1] = myPEenv.n_mpi_task(1);
    nproc_xyz[2] = myPEenv.n_mpi_task(2);

    /* get domain info */
    double domain[3];
    domain[0] = mygrid.ll(0);
    domain[1] = mygrid.ll(1);
    domain[2] = mygrid.ll(2);

    /* compute processor width info */
    double proc_width[3];
    proc_width[0] = domain[0] / nproc_xyz[0];
    proc_width[1] = domain[1] / nproc_xyz[1];
    proc_width[2] = domain[2] / nproc_xyz[2];

    /* compute left and right steps in xyz directions */
    /* x-direction */
    lstep_[0]
        = std::min((int)(ceil(rmax / (proc_width[0]))), (nproc_xyz[0] - 1));
    rstep_[0] = std::min(lstep_[0], nproc_xyz[0] - lstep_[0] - 1);

    /* y-direction */
    lstep_[1]
        = std::min((int)(ceil(rmax / (proc_width[1]))), (nproc_xyz[1] - 1));
    rstep_[1] = std::min(lstep_[1], nproc_xyz[1] - lstep_[1] - 1);
    /* z-direction */
    lstep_[2]
        = std::min((int)(ceil(rmax / (proc_width[2]))), (nproc_xyz[2] - 1));
    rstep_[2] = std::min(lstep_[2], nproc_xyz[2] - lstep_[2] - 1);
}

// This function determines if an ion at position x,y,z is within range of
// the list of ions on this PE. This range is obtained from predetermined
// extents in list_boundary_left[x,y,z] and list_boundary_right[x,y,z].
bool Ions::inListIons(const double x, const double y, const double z)
{
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    const double origin[3]
        = { mygrid.origin(0), mygrid.origin(1), mygrid.origin(2) };
    const double end[3] = { origin[0] + lattice_[0], origin[1] + lattice_[1],
        origin[2] + lattice_[2] };

    // define unique position of ion in periodic domain
    double t[3] = { x, y, z };
    for (short i = 0; i < 3; i++)
    {
        while (t[i] >= end[i])
            t[i] -= lattice_[i];
        while (t[i] < origin[i])
            t[i] += lattice_[i];
    }

    bool inList = false;

    // check to see if ion is in list
    if (((t[0] >= list_boundary_left_[0] && t[0] <= list_boundary_right_[0])
            || ((t[0] - lattice_[0]) >= list_boundary_left_[0]
                && (t[0] - lattice_[0]) <= list_boundary_right_[0])
            || ((t[0] + lattice_[0]) >= list_boundary_left_[0]
                && (t[0] + lattice_[0]) <= list_boundary_right_[0]))
        && ((t[1] >= list_boundary_left_[1] && t[1] <= list_boundary_right_[1])
            || ((t[1] - lattice_[1]) >= list_boundary_left_[1]
                && (t[1] - lattice_[1]) <= list_boundary_right_[1])
            || ((t[1] + lattice_[1]) >= list_boundary_left_[1]
                && (t[1] + lattice_[1]) <= list_boundary_right_[1]))
        && ((t[2] >= list_boundary_left_[2] && t[2] <= list_boundary_right_[2])
            || ((t[2] - lattice_[2]) >= list_boundary_left_[2]
                && (t[2] - lattice_[2]) <= list_boundary_right_[2])
            || ((t[2] + lattice_[2]) >= list_boundary_left_[2]
                && (t[2] + lattice_[2]) <= list_boundary_right_[2])))
        inList = true;

    return inList;
}

bool Ions::inLocalIons(const double x, const double y, const double z)
{
    // std::cout<<"inLocalIons..."<<endl;
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();
    const pb::Grid& mygrid   = mymesh->grid();

    double offset[3];
    for (short i = 0; i < 3; i++)
        offset[i] = (double)myPEenv.my_mpi(i) * div_lattice_[i];

    const double origin[3]
        = { mygrid.origin(0), mygrid.origin(1), mygrid.origin(2) };
    const double end[3] = { origin[0] + lattice_[0], origin[1] + lattice_[1],
        origin[2] + lattice_[2] };

    double t[3] = { x, y, z };
    for (short i = 0; i < 3; i++)
    {
        while (t[i] >= end[i])
            t[i] -= lattice_[i];
        while (t[i] < origin[i])
            t[i] += lattice_[i];
        t[i] -= origin[i];
        t[i] -= offset[i];
    }

    bool inList = false;

    if ((t[0] >= 0. && t[0] < (div_lattice_[0]))
        && (t[1] >= 0. && t[1] < (div_lattice_[1]))
        && (t[2] >= 0. && t[2] < (div_lattice_[2])))
    {
        inList = true;
    }

    return inList;
}

// void Ions::checkUnicityLocalIons()
//{
//    //cout<<"Ions::checkUnicityLocalIons()..."<<endl;
//    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
//    //cout<<"size list = "<<list_ions_.size()<<endl;
//    for(auto& ion : list_ions_)
//    {
//        int here = ion->here();
//        mmpi.allreduce(&here, 1, MPI_SUM);
//        if(here != 1)
//        {
//            std::cout<<"Ion "<<ion->name()<<" is here on multiple
//            tasks"<<endl;
//        }
//    }
//}

int Ions::getNumIons(void)
{
    if (num_ions_ < 0) computeNumIons();

    return num_ions_;
}

void Ions::computeNumIons(void)
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));
    num_ions_ = (int)local_ions_.size();
    mmpi.allreduce(&num_ions_, 1, MPI_SUM);
}

double Ions::getMaxListRadius() const
{
    // get radius of projectors
    const double nlradius = getSpeciesMaxNLradius();
    const double lradius  = getSpeciesMaxLradius();

    double rmax = nlradius > lradius ? nlradius : lradius;

    Control& ct(*(Control::instance()));
    rmax
        = rmax > ct.maxDistanceAtomicInfo() ? rmax : ct.maxDistanceAtomicInfo();

    return rmax;
}

// augment Ions Data
void Ions::augmentIonsData(const int nsteps, const int dir, const int disp,
    const int locSize, const int maxLocSize, std::vector<IonData>& iondata,
    int* offset)
{
    // setup buffers for data transfer
    int csize              = IonData_MaxStrLength * locSize;
    const int cbuff_size_r = IonData_MaxStrLength * maxLocSize;
    char* cbuff            = new char[2 * cbuff_size_r];
    char* cbuff_r          = &cbuff[cbuff_size_r];

    // 7 int/atom in IonData
    int isize              = 7 * locSize + 1;
    const int ibuff_size_r = 7 * maxLocSize + 1;
    int* ibuff             = new int[2 * ibuff_size_r];
    int* ibuff_r           = &ibuff[ibuff_size_r];

    // 16 double/atom in IonData
    int dsize              = 16 * locSize;
    const int dbuff_size_r = 16 * maxLocSize;
    double* dbuff          = new double[2 * dbuff_size_r];
    double* dbuff_r        = &dbuff[dbuff_size_r];

    // pack data into buffers
    IonData::packIonData(cbuff, ibuff, dbuff, iondata);

    // transfer data in multiple directions
    MPI_Request request1[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    MPI_Request request2[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };
    MPI_Request request3[2] = { MPI_REQUEST_NULL, MPI_REQUEST_NULL };

    int src;
    int dst;
    if (disp == -1)
    {
        src = source_[dir];
        dst = dest_[dir];
    }
    else
    {
        src = dest_[dir];
        dst = source_[dir];
    }

    int step = 0;
    while (step < nsteps)
    {
        // chars data
        MPI_Irecv(
            cbuff_r, cbuff_size_r, MPI_CHAR, src, 0, cart_comm_, &request1[0]);
        MPI_Isend(cbuff, csize, MPI_CHAR, dst, 0, cart_comm_, &request1[1]);
        // integers data
        MPI_Irecv(
            ibuff_r, ibuff_size_r, MPI_INT, src, 0, cart_comm_, &request2[0]);
        MPI_Isend(ibuff, isize, MPI_INT, dst, 0, cart_comm_, &request2[1]);
        // doubles data
        MPI_Irecv(dbuff_r, dbuff_size_r, MPI_DOUBLE, src, 0, cart_comm_,
            &request3[0]);
        MPI_Isend(dbuff, dsize, MPI_DOUBLE, dst, 0, cart_comm_, &request3[1]);

        // update lists with most recent data
        std::vector<IonData>::iterator idata = ions_data_.begin() + *offset;
        while (idata != ions_data_.end())
        {
            addIon2Lists(*idata);

            idata++;
        }
        *offset = (int)ions_data_.size();

        // wait to complete data transfer
        MPI_Waitall(2, request1, MPI_STATUS_IGNORE);
        MPI_Waitall(2, request2, MPI_STATUS_IGNORE);
        MPI_Waitall(2, request3, MPI_STATUS_IGNORE);

        // unpack: update ions_data for next call to this function
        char* cptr   = &cbuff_r[0];
        int* iptr    = &ibuff_r[0];
        double* dptr = &dbuff_r[0];
        // first get data size
        const int rsize = *(iptr++);
        // unpack
        for (int i = 0; i < rsize; i++)
        {
            IonData newdata;

            newdata.unpack(cptr, iptr, dptr);

            // update ions_data if ion is in list_ions range
            if (inListIons(newdata.current_position[0],
                    newdata.current_position[1], newdata.current_position[2]))
            {
                ions_data_.push_back(newdata);
            }
        }
        // done with this phase of data transfer

        // update sizes and copy recv buffer to send buffer
        if (step < nsteps - 1)
        {
            csize = rsize * IonData_MaxStrLength;
            isize = 7 * rsize + 1;
            dsize = 16 * rsize;
            memcpy(cbuff, cbuff_r, csize * sizeof(char));
            memcpy(ibuff, ibuff_r, isize * sizeof(int));
            memcpy(dbuff, dbuff_r, dsize * sizeof(double));
        }
        step++;
    }

    delete[] cbuff;
    delete[] ibuff;
    delete[] dbuff;
}

void Ions::updateForcesInteractingIons()
{
    // get computed forces into fion_
    getLocalForces(fion_);

    updateDataInteractingIons(fion_, interacting_fion_);
}

void Ions::updateTaupInteractingIons()
{
    updateDataInteractingIons(taup_, interacting_taup_);
}

void Ions::updateDataInteractingIons(
    std::vector<double>& data, std::vector<double*>& interacting_data)
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    // initialize with local names and forces
    DistributedIonicData ddata(local_names_, data);

    for (short dir = 0; dir < 3; dir++)
    {
        // initial data
        const int lsize = ddata.size();
        int maxsize     = lsize;
        mmpi.allreduce(&maxsize, 1, MPI_MAX);

        DistributedIonicData data2send(ddata);

        // send right to left
        int disp = -1;
        ddata.augmentData(lstep_[dir], dir, disp, lsize, maxsize, data2send);

        // send left to right
        disp = 1;
        ddata.augmentData(rstep_[dir], dir, disp, lsize, maxsize, data2send);
    }

    int ia = 0;
    for (const auto& name : interacting_names_)
    {
        double* ptr = interacting_data[ia];
        ddata.getData(name, ptr);
        ++ia;
    }
}

void Ions::clearLists()
{
    local_ions_.clear();
    for (auto& ion : list_ions_)
    {
        delete ion;
    }
    list_ions_.clear();
}

// update list of local ions
void Ions::updateListIons()
{
    MGmol_MPI& mmpi(*(MGmol_MPI::instance()));

    // collect local_ions data
    // assume local_ions size is same as size of ions_names std::vector
    ions_data_.clear();
    {
        IonData idata;
        for (const auto& ion : local_ions_)
        {
            ion->getIonData(idata);

            // populate ions_data_ list
            ions_data_.push_back(idata);
        }
    }

    // update list ions from current ion positions
    // Note: this is based on data from MD std::vectors

    // First cleanup list_ions_
    clearLists();

    // Update list starting with local ion data.
    // This enables overlapping data accumulation with communication.
    // Most recent/ final data is accumulated later
    int offset = 0;
    for (short dir = 0; dir < 3; dir++)
    {
        // initial data
        const int lsize = (int)ions_data_.size();
        int maxsize     = lsize;
        mmpi.allreduce(&maxsize, 1, MPI_MAX);
        // if(onpe0)cout<<"lsize = "<<ions_data_.size()<<" list size =
        // "<<list_ions_.size()<<endl;
        std::vector<IonData> data2send(ions_data_);

        // send right to left
        int disp = -1;
        augmentIonsData(
            lstep_[dir], dir, disp, lsize, maxsize, data2send, &offset);
        // if(onpe0)cout<<"lsize2 = "<<ions_data_.size()<<" list size2 =
        // "<<list_ions_.size()<<endl;
        // send left to right
        disp = 1;
        augmentIonsData(
            rstep_[dir], dir, disp, lsize, maxsize, data2send, &offset);
        // if(onpe0)cout<<"lsize3 = "<<ions_data_.size()<<" list size3 =
        // "<<list_ions_.size()<<endl;
    }

    // perform last data accumulation here
    std::vector<IonData>::const_iterator idata = ions_data_.begin() + offset;
    while (idata != ions_data_.end())
    {
        addIon2Lists(*idata);

        idata++;
    }

    // clear ions_data_
    ions_data_.clear();
}

void Ions::addIon2Lists(const IonData& data)
{
    if (inListIons(data.current_position[0], data.current_position[1],
            data.current_position[2]))
    {
        Ion* newion = new Ion(getSpecies(data.atomic_num), data);

        // update list_ions_
        list_ions_.push_back(newion);

        // update local_ions_ list
        if (inLocalIons(data.current_position[0], data.current_position[1],
                data.current_position[2]))
        {
            newion->set_here(true);

            newion->setFromIonData(data);

            local_ions_.push_back(newion);
        }
        else
        {
            newion->set_here(false);
        }
    }
}

void Ions::clearStepperData()
{
    local_names_.clear();
    tau0_.clear();
    atmove_.clear();
    taum_.clear();
    taup_.clear();
    fion_.clear();
    velocity_.clear();
    rand_states_.clear();
    pmass_.clear();
    gids_.clear();
}

void Ions::initStepperData()
{
    clearStepperData();

    for (const auto& ion : local_ions_)
    {
        local_names_.push_back(ion->name());
        atmove_.push_back(!ion->locked());
        pmass_.push_back(ion->getMass());
        gids_.push_back(ion->index());

        for (short i = 0; i < 3; i++)
        {
            taum_.push_back(ion->getPreviousPosition(i));
            tau0_.push_back(ion->position(i));
            fion_.push_back(ion->force(i));
            velocity_.push_back(ion->velocity(i));
            rand_states_.push_back(ion->randomState(i));
        }
    }
    // initialize taup_ to enable computation of velocities
    // taup_ = tau0_ + (tau0_-taum_)
    int size_tau = (int)tau0_.size();
    taup_        = tau0_;

    int ione     = 1;
    double alpha = 1.;
    DAXPY(&size_tau, &alpha, &tau0_[0], &ione, &taup_[0], &ione);
    alpha = -1.;
    DAXPY(&size_tau, &alpha, &taum_[0], &ione, &taup_[0], &ione);
}

// update local ion data with ionic stepper data
// Note: forces have already been updated.
// fion may include Langevin forces (so ignore??)
void Ions::updateIons()
{
    assert(tau0_.size() == 3 * local_ions_.size());

    Control& ct(*(Control::instance()));

    // update local_ions data
    int ia = 0;
    for (auto& ion : local_ions_)
    {
        ion->setPosition(
            tau0_[3 * ia + 0], tau0_[3 * ia + 1], tau0_[3 * ia + 2]);

        ion->setRandomState(rand_states_[3 * ia + 0], rand_states_[3 * ia + 1],
            rand_states_[3 * ia + 2]);

        ia++;
    }

    // compute and set velocities if needed
    if (ct.dt > 0.) setVelocities(taum_, tau0_, ct.dt);

    // update various list of ions
    updateListIons();

    setup_ = false;
}

void Ions::shiftIons(const Vector3D& shift)
{
    // update local_ions data
    for (auto& ion : local_ions_)
    {
        ion->shiftPositionXLBOMDTest(shift);
    }

    // update various list of ions
    updateListIons();

    setup_ = false;
}

void Ions::rescaleVelocities(const double factor)
{
    if (fabs(factor - 1.) < 1.e-12) return;

    if (onpe0)
    {
        std::cout << "Ions::rescaleVelocities() with factor " << factor
                  << std::endl;
    }

    for (auto& ion : local_ions_)
    {
        ion->rescaleVelocity(factor);
    }
}

double Ions::computeMinLocalSpacing() const
{
    Control& ct     = *(Control::instance());
    double distance = 1.e6;

    std::vector<Ion*>::const_iterator ion1 = local_ions_.begin();
    std::vector<Ion*>::const_iterator ion2 = local_ions_.begin();
    while (ion1 != local_ions_.end())
    {
        while (ion2 != local_ions_.end())
        {
            if (ion1 != ion2)
            {
                double d = (*ion1)->minimage(**ion2, lattice_, ct.bcPoisson);
                if (d < distance) distance = d;
            }
            ion2++;
        }
        ion1++;
    }

    return distance;
}
