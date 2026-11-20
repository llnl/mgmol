// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "tools.h"
#include "Control.h"
#include "MGmol_MPI.h"
#include "MPIdata.h"
#include "Mesh.h"
#include "Vector3D.h"

#ifdef MGMOL_USE_SCALAPACK
#include "SparseDistMatrix.h"
#endif

#include "MGmol_abort.h"
#include <sys/stat.h>
#include <time.h>

const int im_    = 2147483647;
const int iq_    = 127773;
const int mask_  = 123459876;
const double am_ = (1.0 / im_);

// function to call if operator new can't allocate enough memory
void noMoreMemory()
{
    std::cerr << "Unable to satisfy request for memory for MPI task " << mype
              << std::endl;
    mgmol_abort("Unable to satisfy request for memory");
}

// an atom name should start with a capital letter and end with a number
bool checkValidName(const std::string& name)
{
    if (name.empty()) return false;

    unsigned found = name.find_last_of("0123456789");
    if (name[found] != (*name.rbegin())) return false;

    found = name.find_first_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ");
    if (name[found] != (*name.begin())) return false;

    return true;
}

void stripLeadingAndTrailingBlanks(std::string& stringToModify)
{
    if (stringToModify.empty()) return;

    int startIndex = stringToModify.find_first_not_of(' ');
    // int endIndex   = stringToModify.find_last_not_of(" ");
    int endIndex = stringToModify.find_last_of("0123456789");
    // cerr<<"startIndex="<<startIndex<<endl;
    // cerr<<"endIndex="<<endIndex<<endl;
    std::string tempString(stringToModify);
    stringToModify.erase();
    stringToModify = tempString.substr(startIndex, (endIndex - startIndex + 1));
}

void stripName(std::string& stringToModify)
{
    if (stringToModify.empty()) return;

    // make new string with 1st two characters
    std::string tempString(stringToModify.substr(0, 2));
    int endIndex = tempString.find_first_of("_0123456789");
    // cout<<"endIndex="<<endIndex<<endl;
    if (endIndex > 0)
        stringToModify = tempString.substr(0, endIndex);
    else
        stringToModify = tempString;
}

void appendNumber(std::string& name, const int number)
{
    char extension[4];
    sprintf(extension, "%d", number);
    if (number < 10) name.append("0");
    if (number < 100) name.append("0");
    if (number < 1000) name.append("0");
    int l = 1;
    if (number > 9) l++;
    if (number > 99) l++;
    if (number > 999) l++;
    name.append(extension, l);
}

void read_comments(std::ifstream& tfile)
{
    // while( tfile.get()!='\n');
    // string str;
    // getline(tfile,str);

    char cc = (char)tfile.peek();
    while (cc == ('#') || (cc == '\n') || cc == ' ')
    {
        while (tfile.get() != '\n')
            ;
        cc = (char)tfile.peek(); // look at next character
    }
}

#ifdef MGMOL_USE_SCALAPACK
void setSparseDistMatriConsolidationNumber(const int npes)
{
    int consolidation_number = 9;
    int remainder            = (npes % consolidation_number);
    // find consolidation number with max. or 0 remainder;
    if (remainder > 0)
        for (int i = consolidation_number - 2; i < consolidation_number + 8;
             i++)
        {
            if ((npes % i) == 0)
            {
                consolidation_number = i;
                break;
            }
            int new_remainder = npes % i;
            if ((new_remainder) > remainder)
            {
                consolidation_number = i;
                remainder            = new_remainder;
            }
        }
    dist_matrix::SparseDistMatrix<DISTMATDTYPE>::setConsolidationNumber(
        consolidation_number);

    if (onpe0)
        dist_matrix::SparseDistMatrix<DISTMATDTYPE>::printConsolidationNumber(
            *MPIdata::sout);
}
#endif

void reduceBytes(std::vector<char>& val, const MPI_Comm comm)
{
    assert(sizeof(char) == 1);

    const int nn = (int)val.size();

    // pack data into bits
    int nc = nn / 4;
    if (nc * 4 < nn) nc++;

    std::vector<char> bb(nc, 0);
    for (int i = 0; i < nn; i++)
    {
        if (val[i] == 1)
        {
            int ic = i >> 2;
            int ib = i - 4 * ic;
            switch (ib)
            {
                case 0:
                    bb[ic] = bb[ic] | 1; // set bit 0
                    break;
                case 1:
                    bb[ic] = bb[ic] | 2; // set bit 1
                    break;
                case 2:
                    bb[ic] = bb[ic] | 4; // set bit 2
                    break;
                case 3:
                    bb[ic] = bb[ic] | 8; // set bit 3
                    break;
                default:
                    break;
            }
        }
    }

    // transfer data
    std::vector<char> bbsum(nc, 0);
    MPI_Allreduce(&bb[0], &bbsum[0], nc, MPI_CHAR, MPI_BOR, comm);
    // cout<<"Allreduce terminated..."<<endl;

    // unpack data
    int i   = 0;
    int nn4 = nn / 4;
    for (int ic = 0; ic < nn4; ic++)
    {
        val[i++] = (char)((bbsum[ic] & 1) > 0);
        val[i++] = (char)((bbsum[ic] & 2) > 0);
        val[i++] = (char)((bbsum[ic] & 4) > 0);
        val[i++] = (char)((bbsum[ic] & 8) > 0);
    }
    for (int ic = nn4; ic < nc; ic++)
    {
        if (i < nn) val[i++] = (char)((bbsum[ic] & 1) > 0);
        if (i < nn) val[i++] = (char)((bbsum[ic] & 2) > 0);
        if (i < nn) val[i++] = (char)((bbsum[ic] & 4) > 0);
        if (i < nn) val[i++] = (char)((bbsum[ic] & 8) > 0);
    }
}

/////////////////////////////////////////////////////////////////////////////////////
// function add or subtract arrays
void arrayops(const double* const a, const double* const b, const double s,
    const double e, const int dim, double* result)
{
    for (int i = 0; i < dim; i++)
    {
        result[i] = e * (a[i] + s * b[i]);
    }
}

void printWithTimeStamp(const std::string& string2print, std::ostream& os)
{
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.barrier();
    if (onpe0)
    {
        struct tm* local_tm;
        time_t t = time(nullptr);
        local_tm = localtime(&t);

        char* dateString = asctime(local_tm);
        char* c          = index(dateString, '\n');
        if (c != nullptr) *c = (char)'\0';
        std::string date(dateString);

        os << date << ": " << string2print << std::endl;
        os.flush();
    }
#if 0
    static int s=0;
    int r=0;
    int mpierr=MPI_Reduce(&s, &r, 1, MPI_INT, MPI_SUM, 0, mmpi.commGlobal());
    if( mpierr!=MPI_SUCCESS )
    {
        cerr << " Error in MPI!!!" << std::endl;
        MPI_Abort(mmpi.commGlobal(),EXIT_FAILURE);
    }
    if( r!=mmpi.size()*s && onpe0 )
    {
        cerr << " Error in barrier: "<<r<<"!="<<(mmpi.size()*s)<<endl;
        MPI_Abort(mmpi.commGlobal(),EXIT_FAILURE);
    }
    s++;
    s=(s%128);
#endif
}

bool isOverlaping(const Vector3D& center, const float radius)
{
    Mesh* mymesh             = Mesh::instance();
    const pb::Grid& mygrid   = mymesh->grid();
    const pb::PEenv& myPEenv = mymesh->peenv();

    const double origin[3]
        = { mygrid.origin(0), mygrid.origin(1), mygrid.origin(2) };
    double div_lattice[3];
    for (short i = 0; i < 3; i++)
        div_lattice[i] = mygrid.ll(i) / (double)(myPEenv.n_mpi_task(i));

    double ll[3];
    for (short i = 0; i < 3; i++)
        ll[i] = (double)myPEenv.my_mpi(i) * div_lattice[i] + origin[i];
    // double  ur[3];
    // for(short i=0;i<3;i++)
    //    ur[i] = ll[i] + div_lattice[i];

    const double h[3] = { mygrid.hgrid(0), mygrid.hgrid(1), mygrid.hgrid(2) };

    // correct div_lattice to include only mesh points, not domain
    div_lattice[0] -= h[0];
    div_lattice[1] -= h[1];
    div_lattice[2] -= h[2];

    assert(radius > 1.);

    // subdomain center
    double cc[3] = { 0.5 * (2. * ll[0] + div_lattice[0]),
        0.5 * (2. * ll[1] + div_lattice[1]),
        0.5 * (2. * ll[2] + div_lattice[2]) };
    double t[3]  = { center[0], center[1], center[2] };
    for (short i = 0; i < 3; i++)
    {
        while (t[i] >= cc[i] + 0.5 * mygrid.ll(i))
            t[i] -= mygrid.ll(i);
        while (t[i] < cc[i] - 0.5 * mygrid.ll(i))
            t[i] += mygrid.ll(i);

        t[i] -= ll[i];
    }

    // find "p" closest point within subdomain (projection)
    double p[3] = { t[0], t[1], t[2] };
    if (p[0] < 0.) p[0] = 0.;
    if (p[1] < 0.) p[1] = 0.;
    if (p[2] < 0.) p[2] = 0.;
    if (p[0] > div_lattice[0]) p[0] = div_lattice[0];
    if (p[1] > div_lattice[1]) p[1] = div_lattice[1];
    if (p[2] > div_lattice[2]) p[2] = div_lattice[2];
    double d2 = (p[0] - t[0]) * (p[0] - t[0]) + (p[1] - t[1]) * (p[1] - t[1])
                + (p[2] - t[2]) * (p[2] - t[2]);
    if (d2 < radius * radius)
    {
        return true;
    }

    return false;
}

void exitWithErrorMessage(const std::string& name)
{
    std::cerr << "Function " << name
              << " not implemented and should not be called!!!" << std::endl;
    mgmol_abort();
}

bool fileExists(const char* file)
{
    struct stat buf;
    return (stat(file, &buf) == 0);
}

//  Random number generator of Park and Miller
//  adapted from "Numerical Recipes in C", 2nd. Edition, p.279
//  Returns a uniform random deviate between 0. and 1.
double ran0()
{
    // Initialize the random number generator
    static int idum_ = 3356;

    idum_ ^= mask_;
    long k = idum_ / iq_;
    idum_  = 16807 * (idum_ - k * iq_) - 2836 * k;
    if (idum_ < 0) idum_ += im_;
    double ans = am_ * (double)(idum_);
    idum_ ^= mask_;
    return ans;
}

double minQuadPolynomial(const double e0, const double e1, const double de0,
    const bool print_flag, std::ostream& os)
{
    double x0 = -de0 / (2. * (e1 - de0 - e0));
    if (x0 < 0. && e1 < e0) x0 = 1.;
    x0 = std::min(x0, 1.);

    if (onpe0 && print_flag)
    {
        os << std::setprecision(12) << std::fixed << "x0=" << x0 << std::endl;
        double val = x0 * x0 * (e1 - e0 - de0) + x0 * de0 + e0;
        os << "Predicted E: " << val << std::endl;
    }
    return x0;
}

double minQuadPolynomialFrom3values(const double e0, const double e1,
    const double ehalf, const bool print_flag, std::ostream& os)
{
    double b = -e1 + 4 * ehalf - 3. * e0;
    double a = -4. * ehalf + 2. * e1 + 2. * e0;
    assert((0.25 * a + 0.5 * b + e0 - ehalf) < 1.e-8);

    if (a < 0.)
    {
        if (e1 < e0)
            return 1.;
        else
            return -1.; // failed to find minimum
    }

    double x0 = -b / (2. * a);
    x0        = std::min(x0, 1.);

    if (onpe0 && print_flag)
    {
        os << std::setprecision(12) << std::fixed << "x0=" << x0 << std::endl;
        double val = a * x0 * x0 + b * x0 + e0;
        // os<<"a="<<a<<", b="<<b<<std::endl;
        os << "Predicted E: " << val << std::endl;
    }
    return x0;
}

// compute vector of 3 k-values within cube of size kmax
void getkvector(const int index, const int kmax, int kvector[3])
{
    kvector[0] = index / (kmax * kmax);
    kvector[1] = (index - kvector[0] * kmax * kmax) / kmax;
    kvector[2] = index % kmax;

    // if (onpe0)
    //    std::cout << " k=(" << kvector[0] << "," << kvector[1] << ","
    //              << kvector[2] << ")" << std::endl;
}

double getCharge(double* rho)
{
    Control& ct            = *(Control::instance());
    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    double charge = mygrid.integralOverMesh(rho);

    if (onpe0 && ct.verbose > 0)
        std::cout << std::setprecision(8) << std::fixed << "Charge: " << charge
                  << std::endl;

    return charge;
}
