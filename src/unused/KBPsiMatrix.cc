// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "KBPsiMatrix.h"
#include "Control.h"
#include "Ions.h"
#include "LocGridOrbitals.h"
#include "MGmol_MPI.h"
#include "Mesh.h"
#include "ProjectedMatrices.h"

#include <limits.h>

// #define USE_OLD_ALGO  1

Timer KBPsiMatrix::allreduce_tm_("KBPsiMatrix::allreduce");
Timer KBPsiMatrix::global_sum_tm_("KBPsiMatrix::global_sum");
Timer KBPsiMatrix::compute_kbpsi_tm_("KBPsiMatrix::compute_kbpsi");
Timer KBPsiMatrix::computeHvnlMatrix_tm_("KBPsiMatrix::computeHvnlMatrix");
Timer KBPsiMatrix::allGatherNonzeroElements_tm_(
    "KBPsiMatrix::allGatherNonzeroElements");

set<INDEX_TYPE> KBPsiMatrix::nonzero_elements_;

static const double tolKBpsi = 1.e-12;
// static const int sparse_distmatrix_tasks_per_partitions=128;
static const int sparse_distmatrix_tasks_per_partitions = 256;

dist_matrix::RemoteTasksDistMatrix<DISTMATDTYPE>*
    KBPsiMatrix::remote_tasks_DistMatrix_
    = 0;

KBPsiMatrix::KBPsiMatrix(pb::Lap<ORBDTYPE>* lapop) : lapop_(lapop)
{
    is_setup_ = false;

    storage_ = 0;

    setIterativeIndex(-1);
    count_proj_subdomain_ = -1;

    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    // build sub-communicator to gather one direction at a time
    for (int dir = 0; dir < 3; dir++)
    {
        int color = myPEenv.my_mpi((dir + 1) % 3)
                    + (myPEenv.my_mpi((dir + 2) % 3) << 16);
        int key = myPEenv.my_mpi(dir);

        int rc = MPI_Comm_split(myPEenv.comm(), color, key, &comm_dir_[dir]);
        if (rc != MPI_SUCCESS)
        {
            (*MPIdata::serr) << "Error calling MPI_Comm_split for dir=" << dir
                             << ", color=" << color << ", key=" << key << endl;
            exit(0);
        }
    }
};

KBPsiMatrix::~KBPsiMatrix()
{
    clear();
    for (int dir = 0; dir < 3; dir++)
        MPI_Comm_free(&comm_dir_[dir]);
}

void KBPsiMatrix::clear()
{
    ptr_kbpsi_.clear();
    ptr_kbBpsi_.clear();

    if (storage_ != NULL)
    {
        delete[] storage_;
        storage_ = NULL;
    }

    // nonzero_elements_.clear();
}

void KBPsiMatrix::setup(const Ions& ions, const LocGridOrbitals& orbitals)
{
    allocate(ions, orbitals.numst());
    computeSetNonZeroElements(ions, orbitals);

    is_setup_ = true;

    setOutdated();
}

// allocate storage only for data associated to ions with nl projector
// overlaping with subdomain
void KBPsiMatrix::allocate(const Ions& ions, const int numst)
{
    assert(numst >= 0);
    numst_ = numst;

    clear();

    if (numst == 0) return;

    int count_proj        = ions.countProjectors();
    count_proj_subdomain_ = ions.countProjectorsSubdomain();

    ptr_kbpsi_.resize(count_proj);
    ptr_kbBpsi_.resize(count_proj);
#ifdef USE_OLD_ALGO
    size_kbpsi_ = count_proj * numst_;
#else
    size_kbpsi_ = count_proj_subdomain_ * numst_;
#endif
    //(*MPIdata::sout)<<"KBPsiMatrix::count_proj          ="<<count_proj<<endl;
    //(*MPIdata::sout)<<"count_proj_subdomain_="<<count_proj_subdomain_<<endl;

    if (size_kbpsi_ > 0)
    {
        storage_ = new double[2 * size_kbpsi_];
        memset(storage_, 0, 2 * size_kbpsi_ * sizeof(double));
    }

    // initialize pointers to storage
    double* ptr_storage  = storage_;
    double* ptr_storageB = storage_ + size_kbpsi_;

    int n = 0;

    // loop over ions
    const vector<Ion*>& list_ions    = ions.list_ions();
    vector<Ion*>::const_iterator ion = list_ions.begin();
    while (ion != list_ions.end())
    {
        // allocate storage only for ions with nl projector
        // overlaping with subdomain
#ifndef USE_OLD_ALGO
        if ((*ion)->map_nl())
#endif
        {
            vector<int> gids;
            (*ion)->getGidsNLprojs(gids);

            const short nprojs = (short)gids.size();
            for (short i = 0; i < nprojs; i++)
            {
                const int gid = gids[i];

                if (gid >= ptr_kbpsi_.size())
                {
                    cerr << "gid=" << gid
                         << ", ptr_kbpsi_.size()=" << ptr_kbpsi_.size() << endl;
                }
                assert(gid < ptr_kbpsi_.size());
                assert(size_kbpsi_ > 0);

                // set class member pointers
                ptr_kbpsi_[gid]  = ptr_storage;
                ptr_kbBpsi_[gid] = ptr_storageB;

                // increment pointers
                ptr_storage += numst_;
                ptr_storageB += numst_;

                n += numst_;
            }
        }
        ion++;
    }

    assert(n <= size_kbpsi_);
}

void KBPsiMatrix::globalSumKBpsi(const Ions& ions)
{
    assert(is_setup_);
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

#ifdef PRINT_OPERATIONS
    if (onpe0) (*MPIdata::sout) << "KBPsiMatrix::globalSumKBpsi()" << endl;
#endif

    global_sum_tm_.start();

#ifndef USE_OLD_ALGO // old algorithm: communicate the whole <KB|psi> matrix
    int data_size = 0;
    for (set<INDEX_TYPE>::const_iterator it = nonzero_elements_.begin();
         it != nonzero_elements_.end(); it++)
    {
        const int ion_index = index2ion(*it);

        // const Ion& ion=ions.getIon(ion_index);
        const Ion* ion = ions.findLocalIon(ion_index);
        if (ion != NULL)
        {
            data_size += ion->nProjectors();
        }
    }
    data_size *= 2;

    int data_size_tmp = data_size;
    MPI_Allreduce(
        &data_size_tmp, &data_size, 1, MPI_INT, MPI_SUM, myPEenv.comm());
#endif

    Control& ct = *(Control::instance());

#ifdef USE_OLD_ALGO // old algorithm: communicate the whole <KB|psi> matrix
    int ione      = 1;
    int data_size = 2 * size_kbpsi_;
    double* tmp   = new double[data_size];

    int mpirc = MPI_Allreduce(
        storage_, tmp, data_size, MPI_DOUBLE, MPI_SUM, myPEenv.comm());
    if (mpirc != MPI_SUCCESS)
        mgmol_abort("globalSumKBpsi: MPI_allreduce kbpsi failed!!!");

    memcpy(storage_, tmp, data_size * sizeof(double));
    delete[] tmp;

#else // new algorithm: communicate only non-zero elements
    // if( onpe0 )
    //    (*MPIdata::sout)<<"KBPsiMatrix::globalSumKBpsi() with new
    //    algorithm..."<<endl;

    if (data_size > 0)
    {
        double* sendbuf    = new double[2 * data_size];
        double* recvbuffer = sendbuf + data_size;
        memset(sendbuf, 0, data_size * sizeof(double));

        // pack data for MPI_Allreduce
        int idatas = 0;

        for (set<INDEX_TYPE>::const_iterator it = nonzero_elements_.begin();
             it != nonzero_elements_.end(); it++)
        {
            const int ion_index = index2ion(*it);
            const int st        = index2st(*it);

            const Ion* ion = ions.findIon(ion_index);
            if (ion != NULL)
            {
                vector<int> gids;
                ion->getGidsNLprojs(gids);

                const short nprojs = (short)gids.size();
                for (short i = 0; i < nprojs; i++)
                {
                    const int gid = gids[i];
                    assert(idatas + 1 < data_size);
                    if (ion->map_nl())
                    {
                        assert(ptr_kbpsi_[gid] != NULL);
                        sendbuf[idatas]     = ptr_kbpsi_[gid][st];
                        sendbuf[idatas + 1] = ptr_kbBpsi_[gid][st];
                    }
                    idatas += 2;
                }
            }
        }

        //        assert( idatas==data_size );
        allreduce_tm_.start();
        int mpirc = MPI_Allreduce(sendbuf, recvbuffer, data_size, MPI_DOUBLE,
            MPI_SUM, myPEenv.comm());
        allreduce_tm_.stop();
        if (mpirc != MPI_SUCCESS)
            mgmol_abort("globalSumKBpsi: MPI_allreduce kbpsi failed!!!");

        // unpack data out of MPI_Allreduce
        int idatar = 0;
        for (set<INDEX_TYPE>::const_iterator it = nonzero_elements_.begin();
             it != nonzero_elements_.end(); it++)
        {
            const int ion_index = index2ion(*it);
            const int st        = index2st(*it);

            const Ion* ion = ions.findIon(ion_index);
            if (ion != NULL)
            {
                vector<int> gids;
                ion->getGidsNLprojs(gids);

                const short nprojs = (short)gids.size();
                for (short i = 0; i < nprojs; i++)
                {
                    const int gid = gids[i];
                    assert(idatar + 1 < data_size);
                    if (ion->map_nl())
                    {
                        assert(ptr_kbpsi_[gid] != NULL);
                        ptr_kbpsi_[gid][st]  = recvbuffer[idatar];
                        ptr_kbBpsi_[gid][st] = recvbuffer[idatar + 1];
                    }
                    idatar += 2;
                }
            }
        }

        delete[] sendbuf;
    }

#endif

    global_sum_tm_.stop();

    return;
}

// Loop over the ions with projectors overlapping with local subdomain
// and evaluate <KB|psi> for some state.
void KBPsiMatrix::computeKBpsi(Ions& ions, LocGridOrbitals& orbitals,
    const int first_color, const int nb_colors, const bool flag)
{
    assert(first_color >= 0);
    assert(first_color < 100000);
    assert(nb_colors > 0);
    assert(nb_colors < 100000);
    if (flag) assert(lapop_ != 0);

    compute_kbpsi_tm_.start();

    ORBDTYPE* ppsi;
    const int ldsize = orbitals.getLda();
    assert(ldsize > 0);
    assert(ldsize < 1e8);

    if (flag)
    {
        ppsi = new ORBDTYPE[nb_colors * ldsize];
        // orbitals.setDataWithGhosts();
        // orbitals.trade_boundaries();
        for (int color = 0; color < nb_colors; color++)
        {
            lapop_->rhs(orbitals.getFuncWithGhosts(first_color + color),
                ppsi + color * ldsize);
        }
    }
    else
    {
        ppsi = orbitals.getPsi(first_color);
    }

    // Loop over states, subdomains and ions
    const vector<Ion*>::const_iterator iend = ions.overlappingNL_ions().end();
    Mesh* mymesh                            = Mesh::instance();
    const int subdivx                       = mymesh->subdivx();
    for (int color = 0; color < nb_colors; color++)
        for (int iloc = 0; iloc < subdivx; iloc++)
        {
            const int gid = orbitals.getGlobalIndex(iloc, first_color + color);

            // Loop over the ions
            if (gid != -1)
            {
                vector<Ion*>::const_iterator ion
                    = ions.overlappingNL_ions().begin();
                while (ion != iend)
                {
                    computeLocalElement(
                        **ion, gid, iloc, ppsi + color * ldsize, flag);
                    ion++;
                }
            }
        }

    if (flag) delete[] ppsi;

    compute_kbpsi_tm_.stop();
}

void KBPsiMatrix::computeKBpsi(
    Ions& ions, pb::GridFunc<ORBDTYPE>* phi, const int istate, const bool flag)
{
    if (flag) assert(lapop_ != 0);

    compute_kbpsi_tm_.start();

    Mesh* mymesh           = Mesh::instance();
    const pb::Grid& mygrid = mymesh->grid();

    const int ldsize = mygrid.size();

    ORBDTYPE* ppsi = new ORBDTYPE[ldsize];

    if (flag)
    {
        lapop_->rhs(*phi, ppsi);
    }
    else
    {
        phi->init_vect(ppsi, 'd');
    }

    // Loop over states, subdomains and ions
    const vector<Ion*>::const_iterator iend = ions.overlappingNL_ions().end();
    const int subdivx                       = mymesh->subdivx();
    for (int iloc = 0; iloc < subdivx; iloc++)
    {

        // Loop over the ions
        vector<Ion*>::const_iterator ion = ions.overlappingNL_ions().begin();
        while (ion != iend)
        {
            computeLocalElement(**ion, istate, iloc, ppsi, flag);
            ion++;
        }
    }

    delete[] ppsi;

    compute_kbpsi_tm_.stop();
}

void KBPsiMatrix::allGatherNonzeroElements(MPI_Comm comm, const int ntasks)
{
    allGatherNonzeroElements_tm_.start();

    int nel     = (int)nonzero_elements_.size();
    int max_nel = -1;
    MPI_Allreduce(&nel, &max_nel, 1, MPI_INT, MPI_MAX, comm);

    if (max_nel <= 0) return;

    //(*MPIdata::sout)<<"max_nel="<<max_nel<<endl;
    INDEX_TYPE* elements = new INDEX_TYPE[max_nel];
    memset(elements, 0, max_nel * sizeof(INDEX_TYPE));
    int i = 0;
    for (set<INDEX_TYPE>::const_iterator it = nonzero_elements_.begin();
         it != nonzero_elements_.end(); it++)
    {
        elements[i] = *it;
        i++;
    }
    nonzero_elements_.clear();

    const int ngather_elements  = max_nel * ntasks;
    INDEX_TYPE* gather_elements = new INDEX_TYPE[ngather_elements];
    MPI_Allgather(elements, max_nel, MPI_INDEX_TYPE, gather_elements, max_nel,
        MPI_INDEX_TYPE, comm);
    delete[] elements;

    // put elements collected into set (-> sorted, unique)
    for (int j = 0; j < ngather_elements; j++)
    {
        nonzero_elements_.insert(gather_elements[j]);
    }
    nonzero_elements_.erase(0);

    delete[] gather_elements;

    allGatherNonzeroElements_tm_.stop();
}

void KBPsiMatrix::computeSetNonZeroElements(
    const Ions& ions, const LocGridOrbitals& orbitals)
{
    // compute "local" set
    nonzero_elements_.clear();

    Mesh* mymesh                            = Mesh::instance();
    const int ncolors                       = orbitals.chromatic_number();
    const int subdivx                       = mymesh->subdivx();
    const vector<Ion*>::const_iterator iend = ions.overlappingNL_ions().end();
    for (int color = 0; color < ncolors; color++)
        for (int iloc = 0; iloc < subdivx; iloc++)
        {
            const int st = orbitals.getGlobalIndex(iloc, color);

            if (st >= 0)
            {
                // Loop over the ions
                vector<Ion*>::const_iterator ion
                    = ions.overlappingNL_ions().begin();
                while (ion != iend)
                {
                    nonzero_elements_.insert(index(st, (*ion)->index()));
                    ion++;
                }
            }
        }

    // complete with sets from other tasks
    const pb::PEenv& myPEenv = mymesh->peenv();

    // gather one direction at a time
    for (int dir = 0; dir < 3; dir++)
    {
        allGatherNonzeroElements(comm_dir_[dir], myPEenv.n_mpi_task(dir));
    }

    if (onpe0)
    {
        (*MPIdata::sout) << "Number of nonzero pairs in KBPsiMatrix: "
                         << nonzero_elements_.size() << endl;
        //(*MPIdata::sout)<<"sizeof(INDEX_TYPE): "<<sizeof(INDEX_TYPE)<<endl;
        // set<INDEX_TYPE>::const_iterator it=nonzero_elements_.end();
        // it--;
        //(*MPIdata::sout)<<"Largest element in nonzero_elements_: "<<*it<<endl;
    }
}

void KBPsiMatrix::scaleWithKBcoeff(const Ions& ions)
{
    int ione                          = 1;
    vector<Ion*>::const_iterator ion  = ions.overlappingNL_ions().begin();
    vector<Ion*>::const_iterator iend = ions.overlappingNL_ions().end();
    while (ion != iend)
    {
        vector<int> gids;
        (*ion)->getGidsNLprojs(gids);
        vector<double> kbcoeffs;
        (*ion)->getKBcoeffs(kbcoeffs);

        const short nprojs = (short)gids.size();
        assert(nprojs == gids.size());
        assert(nprojs == kbcoeffs.size());

        for (short i = 0; i < nprojs; i++)
        {
            const int gid = gids[i];
            double coeff  = kbcoeffs[i];
            assert(gid < ptr_kbpsi_.size());
            assert(ptr_kbpsi_[gid] != NULL);
            assert(ptr_kbBpsi_[gid] != NULL);
            dscal(&numst_, &coeff, ptr_kbpsi_[gid], &ione);
            dscal(&numst_, &coeff, ptr_kbBpsi_[gid], &ione);
        }

        ion++;
    }
}

// computeHvnlMatrix:
//    Get the elements of the Hamiltonian matrix due to the non-local
//    potential, and add them into Aij.
// Note: neglecting the small matrix elements reduces the size of hnlij and thus
//       reduces the size of communications later on.
void KBPsiMatrix::computeHvnlMatrix(const KBPsiMatrix* const kbpsi2,
    const Ion& ion, dist_matrix::SparseDistMatrix<DISTMATDTYPE>& hnlij) const
{
    assert(ion.here());
    vector<int> gids;
    ion.getGidsNLprojs(gids);
    vector<short> kbsigns;
    ion.getKBsigns(kbsigns);
    const short nprojs = (short)gids.size();
    for (short i = 0; i < nprojs; i++)
    {
        const int gid       = gids[i];
        const double kbsign = (double)kbsigns[i];

        const double* const pkbpsi  = kbpsi2->ptr_kbpsi_[gid];
        const double* const pkbBpsi = ptr_kbBpsi_[gid];

        assert(pkbpsi != NULL);
        assert(pkbBpsi != NULL);

        for (int st1 = 0; st1 < numst_; st1++)
        {
            const double kbpsi1 = pkbBpsi[st1] * kbsign;
            if (fabs(kbpsi1) > tolKBpsi)
            {
                // Important test for scaling of very large systems!!

                for (int st2 = 0; st2 < numst_; st2++)
                {
                    const double alpha = kbpsi1 * pkbpsi[st2];
                    if (fabs(alpha) > tolKBpsi)
                        hnlij.push_back(st1, st2, alpha);
                }
            }
        }
    }
}

void KBPsiMatrix::computeHvnlMatrix(const KBPsiMatrix* const kbpsi2,
    const Ion& ion, ProjectedMatricesInterface* proj_matrices) const
{
    assert(ion.here());

    vector<int> gids;
    ion.getGidsNLprojs(gids);
    vector<short> kbsigns;
    ion.getKBsigns(kbsigns);
    const short nprojs = (short)gids.size();
    for (short i = 0; i < nprojs; i++)
    {
        const int gid       = gids[i];
        const double kbsign = (double)kbsigns[i];

        const double* const pkbpsi  = kbpsi2->ptr_kbpsi_[gid];
        const double* const pkbBpsi = ptr_kbBpsi_[gid];

        assert(pkbpsi != NULL);
        assert(pkbBpsi != NULL);

        for (int st1 = 0; st1 < numst_; st1++)
        {

            const double kbpsi1 = pkbBpsi[st1] * kbsign;
            if (fabs(kbpsi1) > tolKBpsi)
            {
                // Important test for scaling of very large systems!!

                for (int st2 = 0; st2 < numst_; st2++)
                {

                    const double alpha = kbpsi1 * pkbpsi[st2];
                    if (fabs(alpha) > tolKBpsi)
                        proj_matrices->addMatrixElementSparseH(st1, st2, alpha);
                }
            }
        }
    }
}

void KBPsiMatrix::computeHvnlMatrix(
    const Ions& ions, dist_matrix::SparseDistMatrix<DISTMATDTYPE>& Aij) const
{
    computeHvnlMatrix(this, ions, Aij);
}

void KBPsiMatrix::computeHvnlMatrix(
    const Ions& ions, ProjectedMatricesInterface* proj_matrices) const
{
    computeHvnlMatrix(this, ions, proj_matrices);
}

void KBPsiMatrix::computeHvnlMatrix(const KBPsiMatrixInterface* const kbpsi2,
    const Ions& ions, dist_matrix::SparseDistMatrix<DISTMATDTYPE>& Aij) const
{
    computeHvnlMatrix_tm_.start();

    // Loop over ions centered on current PE only
    // (distribution of work AND Hvnlij contributions)
    vector<Ion*>::const_iterator ion = ions.local_ions().begin();
    while (ion != ions.local_ions().end())
    {
        computeHvnlMatrix((KBPsiMatrix*)kbpsi2, **ion, Aij);
        ion++;
    }

    computeHvnlMatrix_tm_.stop();
}

void KBPsiMatrix::computeHvnlMatrix(const KBPsiMatrixInterface* const kbpsi2,
    const Ions& ions, ProjectedMatricesInterface* proj_matrices) const
{
    computeHvnlMatrix_tm_.start();

    // Loop over ions centered on current PE only
    // (distribution of work AND Hvnlij contributions)
    vector<Ion*>::const_iterator ion = ions.local_ions().begin();
    while (ion != ions.local_ions().end())
    {
        computeHvnlMatrix((KBPsiMatrix*)kbpsi2, **ion, proj_matrices);
        ion++;
    }

    computeHvnlMatrix_tm_.stop();
}

// build elements of matrix <phi_i|Vnl|phi_j> (assumed to be symmetric)
void KBPsiMatrix::getPsiKBPsiSym(
    const Ion& ion, dist_matrix::SparseDistMatrix<DISTMATDTYPE>& sm)
{
    vector<int> gids;
    ion.getGidsNLprojs(gids);

    vector<short> kbsigns;
    ion.getKBsigns(kbsigns);

    const short nprojs = (short)gids.size();
    for (short i = 0; i < nprojs; i++)
    {
        const int gid       = gids[i];
        const double kbsign = (double)kbsigns[i];

        // loop over all the states
        for (int st1 = 0; st1 < numst_; st1++)
        {

            double kbpsi_st1 = getValIonState(gid, st1);
            if (fabs(kbpsi_st1) > 1.e-12)
            {
                // Important test for scaling of very large systems!!

                kbpsi_st1 *= kbsign;

                for (int st2 = 0; st2 < st1; st2++)
                {
                    const double alpha = getValIonState(gid, st2) * kbpsi_st1;
                    if (fabs(alpha) > 1.e-12)
                    {
                        sm.push_back(st2, st1, alpha);
                        sm.push_back(st1, st2, alpha);
                    }
                }
                const double kbpsi_st2 = getValIonState(gid, st1);
                sm.push_back(st1, st1, kbpsi_st1 * kbpsi_st2);
            }
        }
    }
}

void KBPsiMatrix::getPsiKBPsiSym(
    const Ions& ions, dist_matrix::SparseDistMatrix<DISTMATDTYPE>& sm)
{
    // loop over all the ions (parallelization over ions)
    const vector<Ion*>::const_iterator iend = ions.local_ions().end();
    vector<Ion*>::const_iterator ion        = ions.local_ions().begin();
    while (ion != iend)
    {
        getPsiKBPsiSym(**ion, sm);
        ion++;
    }
}

// build H elements
void KBPsiMatrix::getPsiKBPsiSym(
    const Ions& ions, dist_matrix::DistMatrix<DISTMATDTYPE>& Aij)
{
    assert(remote_tasks_DistMatrix_ != 0);

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    MPI_Comm comm   = mmpi.commSameSpin();
    dist_matrix::SparseDistMatrix<DISTMATDTYPE> sm(comm, Aij,
        remote_tasks_DistMatrix_, sparse_distmatrix_tasks_per_partitions);

    getPsiKBPsiSym(ions, sm);

    // sum contributions from all processors into Aij
    sm.parallelSumToDistMatrix();
}

void KBPsiMatrix::computeAll(Ions& ions, LocGridOrbitals& orbitals)
{
    if (getIterativeIndex() >= 0)
        if (orbitals.getIterativeIndex() == getIterativeIndex())
        {
#ifdef PRINT_OPERATIONS
            if (onpe0)
                (*MPIdata::sout) << "KBPsi coeff. up to date, "
                                    "KBPsiMatrix::computeAll() skipped"
                                 << endl;
#endif
            return;
        }

#ifdef PRINT_OPERATIONS
    if (onpe0) (*MPIdata::sout) << "KBPsiMatrix::computeAll()" << endl;
#endif
    reset();

    const int iinit = 0;
    const int iend  = orbitals.chromatic_number();
    const int bsize = 32;

    orbitals.setDataWithGhosts();
    orbitals.trade_boundaries();

    setIterativeIndex(orbitals.getIterativeIndex());

    for (int color = iinit; color < iend; color += bsize)
    {
        int ncolors = min(bsize, iend - color);
        computeKBpsi(ions, orbitals, color, ncolors, 0);
        computeKBpsi(ions, orbitals, color, ncolors, 1);
    }

    setIterativeIndex(orbitals.getIterativeIndex());

    globalSumKBpsi(ions);

    scaleWithKBcoeff(ions);
}

void KBPsiMatrix::printTimers(ostream& os)
{
    KBPsiMatrixInterface::printTimers(os);

    allreduce_tm_.print(os);
    computeHvnlMatrix_tm_.print(os);
    global_sum_tm_.print(os);
    compute_kbpsi_tm_.print(os);
    allGatherNonzeroElements_tm_.print(os);
}

double KBPsiMatrix::getEvnl(const Ions& ions, LocGridOrbitals& orbitals,
    ProjectedMatrices* proj_matrices)
{
    const int numst = orbitals.numst();
    if (numst == 0) return 0.;

    dist_matrix::DistMatrix<DISTMATDTYPE> Aij("A", numst, numst);

    getPsiKBPsiSym(ions, Aij);

    double evnl = proj_matrices->getExpectation(Aij);

    return evnl;
}
