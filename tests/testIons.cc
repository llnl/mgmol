#include "Control.h"
#include "Ions.h"
#include "MGmol_MPI.h"
#include "Mesh.h"
#include "Species.h"

#include <random>

#include "catch.hpp"

// check that all forces components have integer values larger than 0
// and differ from each other
void checkForces(std::vector<double>& forces)
{
    const double tol = 1.e-14;

    for (auto f0 = forces.begin(); f0 != forces.end(); f0++)
    {
        std::cout << "f0 = " << *f0 << std::endl;
        for (auto f1 = f0 + 1; f1 != forces.end(); f1++)
        {
            // make sure each force component is different
            CHECK(std::abs(*f0 - *f1) > tol);
            CHECK(*f1 > tol);
            CHECK(*f0 > tol);
        }
    }
}

TEST_CASE("Ions", "[ions]")
{
    MPI_Comm comm = MPI_COMM_WORLD;

    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MGmol_MPI::setup(comm, std::cout);
    Control::setup(comm, false, 0.);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    // create a domain [0.10.]^3
    const double origin[3]  = { 0., 0., 0. };
    const double ll         = 10.;
    const double lattice[3] = { ll, ll, ll };
    const unsigned ngpts[3] = { 32, 24, 20 };
    pb::LapType lap_type    = pb::LapType::FD4th;

    Mesh::setup(comm, ngpts, origin, lattice, lap_type);

    const double h[3] = { ll / (double(ngpts[0])), ll / (double(ngpts[1])),
        ll / (double(ngpts[2])) };

    // random number generator
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    // create one species
    Species sp(MPI_COMM_WORLD);

    // read species info from pseudopotential file
    std::string filename("pseudo.C_ONCV_PBE_SG15");
    if (myrank == 0) std::cout << "Potential = " << filename << std::endl;

    sp.read_1species(filename);
    sp.set_dim_nl(h[0]);
    sp.set_dim_l(h[0]);
    sp.initPotentials('f', h[0], true);

    // put species into a vector
    std::vector<Species> vsp;
    vsp.push_back(sp);

    Ions ions(lattice, vsp);
    ions.setupListIonsBoundaries(10000.);

    double velocity[3] = { 0., 0., 0. };

    // set "na" atoms coordinates and add them to "ions"
    const int na = 10;
    for (int i = 0; i < na; i++)
    {
        double x[3] = { origin[0] + lattice[0] * dis(gen),
            origin[1] + lattice[1] * dis(gen),
            origin[2] + lattice[2] * dis(gen) };
        if (myrank == 0)
            std::cout << "x,y,z = " << x[0] << ", " << x[1] << ", " << x[2]
                      << std::endl;

        // set all x to the values of PE0
        MPI_Bcast(&x[0], 3, MPI_DOUBLE, 0, comm);

        // make a name for atom based on species and order of reading in
        std::string stri = std::to_string(i);
        std::string aname("C" + stri);

        ions.addIonToList(sp, aname, &x[0], velocity, false);
    }

    ions.setup();

    // verify sum of local ions adds up to total number of ions
    {
        std::vector<Ion*>& new_local_ions(ions.local_ions());

        int nlocal = new_local_ions.size();
        std::cout << "PE " << myrank << ", nlocal = " << nlocal << std::endl;

        int ntotal = 0;
        MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, comm);
        CHECK(ntotal == na);
    }
    mmpi.barrier();

    // verify some functionalities of class Ions
    {
        std::vector<double> positions;
        std::vector<short> anumbers;
        ions.getPositions(positions);
        ions.getAtomicNumbers(anumbers);
        if (myrank == 0)
        {
            std::cout << "Positions:" << std::endl;
            int i = 0;
            for (auto& position : positions)
            {
                std::cout << position;
                if (i % 3 == 2)
                    std::cout << std::endl;
                else
                    std::cout << "   ";
                i++;
            }
        }
        mmpi.barrier();

        // swap x and z
        for (size_t i = 0; i < positions.size() - 2; i += 3)
        {
            double x         = positions[i];
            double z         = positions[i + 2];
            positions[i]     = z;
            positions[i + 2] = x;
        }

        ions.setPositions(positions, anumbers);
    }

    mmpi.barrier();
    {
        std::vector<Ion*>& new_local_ions(ions.local_ions());

        int nlocal = new_local_ions.size();
        std::cout << "PE " << myrank << ", nlocal = " << nlocal << std::endl;

        int ntotal = 0;
        MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, comm);
        CHECK(ntotal == na);
    }

    // get the names of all the ions
    std::vector<std::string> names;
    ions.getNames(names);
    if (myrank == 0)
        for (auto& name : names)
            std::cout << "name = " << name << std::endl;
    CHECK(names.size() == na);

    mmpi.barrier();

    std::vector<double> forces(3 * na);
    // set forces to a different arbitrary value for each component
    int i = 1;
    for (auto& f : forces)
    {
        f = (double)i;
        i++;
    }
    ions.getNames(names);
    ions.setLocalForces(forces, names);

    ions.printForcesGlobal(std::cout);

    int nlocal = ions.getNumLocIons();
    std::vector<double> lforces(3 * nlocal);
    ions.getLocalForces(lforces);
    for (auto& f : lforces)
    {
        CHECK(std::fmod(f, 1.) < 1.e-14);
    }

    ions.getForces(forces);
    if (myrank == 0)
    {
        checkForces(forces);
    }

    // test Ions::setLocalForces based on coordinates matching
    {
        std::vector<double> positions;
        std::vector<short> anumbers;
        ions.getPositions(positions);

        ions.setLocalForces(forces, positions);

        ions.printForcesGlobal(std::cout);

        ions.getForces(forces);
        if (myrank == 0)
        {
            checkForces(forces);
        }
    }
}
