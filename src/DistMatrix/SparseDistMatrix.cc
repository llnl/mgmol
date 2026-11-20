// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "SparseDistMatrix.h"

#include <iostream>
#include <map>
#include <string>

#if ZOLTAN
#include "zoltan_comm_cpp.h"
#endif

#include "MGmol_abort.h"

#define MEMMON 0

#define SHIFT 16
#define BIG_DOUBLE 65536. // = 2**16

#define MPIALLTOALL 0

#ifdef __KCC
#define TEMP_DECL /**/
#else
#define TEMP_DECL template <>
#endif

#if MEMMON
extern "C"
{
    void memmon_trace_on(int* rank_p);
    void memmon_print_usage();
}
#endif

namespace dist_matrix
{

const unsigned short reserve_size = 8;

template <class T>
std::shared_ptr<RemoteTasksDistMatrix<T>>
    SparseDistMatrix<T>::rtasks_distmatrix_;

template <>
int SparseDistMatrix<double>::Alltoallv(const void* sendbuf,
    const int* sendcounts, const int* sdispls, void* recvbuf,
    const int* recvcounts, const int* rdispls, MPI_Comm comm)
{
    return MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_DOUBLE, recvbuf,
        recvcounts, rdispls, MPI_DOUBLE, comm);
}

template <>
int SparseDistMatrix<float>::Alltoallv(const void* sendbuf,
    const int* sendcounts, const int* sdispls, void* recvbuf,
    const int* recvcounts, const int* rdispls, MPI_Comm comm)
{
    return MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_FLOAT, recvbuf,
        recvcounts, rdispls, MPI_FLOAT, comm);
}

template <>
int SparseDistMatrix<double>::Alltoall(const void* sendbuf, int sendcount,
    void* recvbuf, int recvcount, MPI_Comm comm)
{
    return MPI_Alltoall(
        sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcount, MPI_DOUBLE, comm);
}

template <>
int SparseDistMatrix<float>::Alltoall(const void* sendbuf, int sendcount,
    void* recvbuf, int recvcount, MPI_Comm comm)
{
    return MPI_Alltoall(
        sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcount, MPI_FLOAT, comm);
}

template <>
int SparseDistMatrix<double>::Isend(const void* buf, int count, int dest,
    int tag, MPI_Comm comm, MPI_Request* request)
{
    return MPI_Isend(buf, count, MPI_DOUBLE, dest, tag, comm, request);
}

template <>
int SparseDistMatrix<float>::Isend(const void* buf, int count, int dest,
    int tag, MPI_Comm comm, MPI_Request* request)
{
    return MPI_Isend(buf, count, MPI_FLOAT, dest, tag, comm, request);
}

template <>
int SparseDistMatrix<double>::Irecv(void* buf, int count, int source, int tag,
    MPI_Comm comm, MPI_Request* request)
{
    return MPI_Irecv(buf, count, MPI_DOUBLE, source, tag, comm, request);
}

template <>
int SparseDistMatrix<float>::Irecv(void* buf, int count, int source, int tag,
    MPI_Comm comm, MPI_Request* request)
{
    return MPI_Irecv(buf, count, MPI_FLOAT, source, tag, comm, request);
}

template <class T>
SparseDistMatrix<T>::SparseDistMatrix(
    MPI_Comm comm, DistMatrix<T>& mat, const int target_nb_tasks_per_partition)
    : comm_global_(comm), mat_(mat)
{
    ntasks_per_partition_ = -1;
    npartitions_          = -1;

    MPI_Comm_rank(comm_global_, &mype_);
    MPI_Comm_size(comm_global_, &npes_);
    nprow_          = mat_.nprow();
    const int npcol = mat_.npcol();
    assert(nprow_ > 0);
    assert(npcol > 0);
    ntasks_mat_ = nprow_ * npcol;

    index_and_val_.resize(ntasks_mat_);
    map_val_.resize(ntasks_mat_);

    if (!rtasks_distmatrix_)
    {
        rtasks_distmatrix_.reset(new RemoteTasksDistMatrix<T>(mat));
    }

    int target_nb_tasks = target_nb_tasks_per_partition;
    if (target_nb_tasks <= 0) target_nb_tasks = npes_; // default

    setPartitioning(target_nb_tasks);
}

template <class T>
SparseDistMatrix<T>::SparseDistMatrix(MPI_Comm comm, DistMatrix<T>& mat)
    : comm_global_(comm), mat_(mat)
{
    ntasks_per_partition_ = -1;
    npartitions_          = -1;

    MPI_Comm_rank(comm_global_, &mype_);
    MPI_Comm_size(comm_global_, &npes_);
    nprow_          = mat_.nprow();
    const int npcol = mat_.npcol();
    assert(nprow_ > 0);
    assert(npcol > 0);
    ntasks_mat_ = nprow_ * npcol;

    index_and_val_.resize(ntasks_mat_);
    map_val_.resize(ntasks_mat_);

    if (!rtasks_distmatrix_)
    {
        rtasks_distmatrix_.reset(new RemoteTasksDistMatrix<T>(mat));
    }

    int target_nb_tasks = sparse_distmatrix_nb_partitions_;
    if (target_nb_tasks <= 0) target_nb_tasks = npes_; // default

    setPartitioning(target_nb_tasks);
}

template <class T>
SparseDistMatrix<T>::SparseDistMatrix(const SparseDistMatrix<T>& spdistmat)
    : comm_global_(spdistmat.comm_global_), mat_(spdistmat.mat_)
{
    mype_       = spdistmat.mype_;
    npes_       = spdistmat.npes_;
    nprow_      = spdistmat.nprow_;
    ntasks_mat_ = spdistmat.ntasks_mat_;

    partition_comm_ = spdistmat.partition_comm_;

    ntasks_per_partition_ = spdistmat.ntasks_per_partition_;
    npartitions_          = spdistmat.npartitions_;

    index_and_val_ = spdistmat.index_and_val_;
    map_val_       = spdistmat.map_val_;
}

template <class T>
SparseDistMatrix<T>::~SparseDistMatrix<T>()
{
}

template <class T>
void SparseDistMatrix<T>::addData(const std::vector<T>& data, const int ld,
    const int ilow, const int ihi, const int jlow, const int jhi)
{
    for (int j = jlow; j < jhi; j++)
    {
        for (int i = ilow; i < ihi; i++)
        {
            push_back(i, j, data[i + j * ld]);
        }
    }
}

template <class T>
void SparseDistMatrix<T>::addData(const std::vector<T>& data, const int ld,
    const int ilow, const int ihi, const int jlow, const int jhi,
    const std::vector<int>& gids)
{
    for (int j = jlow; j < jhi; j++)
    {
        int gidj = gids[j];
        if (gidj > -1)
            for (int i = ilow; i < ihi; i++)
            {
                int gidi = gids[j];
                if (gidi > -1) push_back(gidi, gidj, data[i + j * ld]);
            }
    }
}

template <class T>
void SparseDistMatrix<T>::addData(
    const SquareSubMatrix<T>& mat, const double tol)
{
    const std::vector<int>& gid(mat.getGids());
    const int n = gid.size();

    for (int j = 0; j < n; j++)
    {
        assert(gid[j] >= 0);

        for (int i = 0; i < n; i++)
        {
            const double val = mat.getLocalValue(i, j);
            if (std::abs(val) > tol) push_back(gid[i], gid[j], val);
        }
    }
}

template <class T>
void SparseDistMatrix<T>::scal(const T alpha)
{
    int n = (int)map_val_.size();
    for (int i = 0; i < n; i++)
    {
        std::map<int, T>& ref_map_val(map_val_[i]);
        for (auto p : ref_map_val)
        {
            p.second *= alpha;
        }
    }
}

template <class T>
void SparseDistMatrix<T>::print(std::ostream& os) const
{
    const int n = (int)index_and_val_.size();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (int)index_and_val_[i].size(); j = j + 2)
        {
            const double v  = index_and_val_[i][j];
            const double di = floor(v / BIG_DOUBLE);
            os << "PE " << mype_ << ": (" << (int)di << ","
               << (int)(v - di * BIG_DOUBLE) << "), "
               << index_and_val_[i][j + 1] << std::endl;
        }
}

template <class T>
size_t SparseDistMatrix<T>::size() const
{
    size_t size_data = 0;
    const int n      = (int)map_val_.size();
    for (int i = 0; i < n; i++)
        size_data += map_val_[i].size();

    return size_data;
}

void myMPI_AlltoallDouble(
    double* sendbuf, double* recvbuf, const int count, MPI_Comm comm)
{
    int mype, npes;
    MPI_Comm_rank(comm, &mype);
    MPI_Comm_size(comm, &npes);

    MPI_Request req[2];

    // copy local data
    memcpy(
        &recvbuf[mype * count], &sendbuf[mype * count], count * sizeof(double));

    int src = mype;

    for (int p = 1; p < npes; p++)
    {
        int dst = (mype + p) % npes;
        src--;
        if (src < 0) src += npes;

        MPI_Irecv(
            &recvbuf[src * count], count, MPI_DOUBLE, src, p, comm, &req[0]);
        MPI_Isend(
            &sendbuf[dst * count], count, MPI_DOUBLE, dst, p, comm, &req[1]);

        // complete send/recv
        MPI_Status status;
        MPI_Wait(&req[1], &status);
        MPI_Wait(&req[0], &status);
    }
}

template <class T>
void SparseDistMatrix<T>::push_back(
    const int index1, const int index2, const T val)
{
    const int pr = mat_.pr(index1);
    const int pc = mat_.pc(index2);
    const int pe = pr + pc * nprow_; // destination PE
    assert(pe >= 0);
    assert(pe < (int)map_val_.size());
    assert(!std::isnan(val));

    map_val_[pe][(index1 << SHIFT) + index2] += val;
}

template <class T>
void SparseDistMatrix<T>::maptoarray()
{
    maptoarray_tm_.start();
    // std::cout<<"SparseDistMatrix<T>::maptoarray()"<<endl;
    const int imax = map_val_.size();
    for (int pe = 0; pe < imax; pe++)
    {
        std::vector<T>& index_and_val_pe(index_and_val_[pe]);
        index_and_val_pe.clear();
        const std::map<int, T>& ref_map_val(map_val_[pe]);
        if (ref_map_val.size() > 1) index_and_val_pe.reserve(reserve_size);
        for (auto p : ref_map_val)
        {
            assert(p.first >= 0);
            index_and_val_pe.push_back((T)p.first);
            index_and_val_pe.push_back(p.second);
        }
    }
    maptoarray_tm_.stop();
}

template <class T>
void SparseDistMatrix<T>::array2map()
{
    map_val_.clear();
    map_val_.resize(ntasks_mat_);
    for (int pe = 0; pe < ntasks_mat_; pe++)
    {
        const int size_index_and_val_pe        = index_and_val_[pe].size();
        std::map<int, T>& ref_map_val_pe       = map_val_[pe];
        const std::vector<T>& index_and_val_pe = index_and_val_[pe];
        for (int i = 0; i < size_index_and_val_pe; i = i + 2)
        {
            ref_map_val_pe[(int)index_and_val_pe[i]] += index_and_val_pe[i + 1];
        }
    }
}

template <class T>
void SparseDistMatrix<T>::consolidateArray()
{
    consolidateArray_tm_.start();

    // if( mype_==0 )
    //   std::cout<<"Consolidate,
    //   consolidation_number="<<consolidation_number_<<endl;

    maptoarray();

    map_val_.clear(); // to save memory

    unsigned short num_neighbors = consolidation_number_;
    if (num_neighbors > ntasks_mat_) num_neighbors = ntasks_mat_;

    const int npes_color = num_neighbors;

    // special case
    const unsigned short nremaining_tasks = (npes_ % num_neighbors);
    if (nremaining_tasks)
    {
        if (mype_ >= (npes_ - nremaining_tasks))
        {
            num_neighbors = nremaining_tasks;
            // std::cout<<"mype="<<mype_<<",
            // num_neighbors="<<num_neighbors<<endl;
        }
    }

    MPI_Comm sub_comm;
    const int color = mype_ / npes_color;
    const int key   = mype_ % npes_color;
    MPI_Comm_split(comm_global_, color, key, &sub_comm);

    int fraction_tasks = (ntasks_mat_ / num_neighbors);
    if (fraction_tasks * num_neighbors < ntasks_mat_) fraction_tasks++;

    {
        const int npes_mat = num_neighbors * fraction_tasks;
        assert(npes_mat > 0);
        assert(npes_mat >= ntasks_mat_);

        // first send size of arrays to be sent later
        int* ndata2send = new int[npes_mat];
        for (int dst = 0; dst < num_neighbors; dst++)
        {
            const int pe_init = dst * fraction_tasks;
            const int pe_end  = std::min(pe_init + fraction_tasks, ntasks_mat_);
            for (int pe = pe_init; pe < pe_end; pe++)
                ndata2send[pe] = (int)index_and_val_[pe].size();
        }
        for (int i = ntasks_mat_; i < npes_mat; i++)
            ndata2send[i] = 0;

        int* ndata2recv = new int[npes_mat];
#ifndef NDEBUG
        int rc =
#endif
            MPI_Alltoall(ndata2send, fraction_tasks, MPI_INT, ndata2recv,
                fraction_tasks, MPI_INT, sub_comm);
        assert(rc == MPI_SUCCESS);

        int* sendcounts = new int[num_neighbors];
        for (int dst = 0; dst < num_neighbors; dst++)
        {
            sendcounts[dst]   = 0;
            const int pe_init = dst * fraction_tasks;
            const int pe_end  = pe_init + fraction_tasks;
            for (int pe = pe_init; pe < pe_end; pe++)
            {
                sendcounts[dst] += ndata2send[pe];
            }
        }
        delete[] ndata2send;

        int total_ndata2send = 0;
        for (int dst = 0; dst < num_neighbors; dst++)
        {
            total_ndata2send += sendcounts[dst];
        }

        int* recvcounts = new int[num_neighbors];
        for (int src = 0; src < num_neighbors; src++)
        {
            recvcounts[src] = 0;
            const int iinit = src * fraction_tasks;
            const int iend  = iinit + fraction_tasks;
            for (int i = iinit; i < iend; i++)
            {
                recvcounts[src] += ndata2recv[i];
            }
        }
        int total_ndata2recv = 0;
        for (int src = 0; src < num_neighbors; src++)
        {
            total_ndata2recv += recvcounts[src];
        }

        int* rdispls = new int[num_neighbors];
        rdispls[0]   = 0;
        for (int i = 1; i < num_neighbors; i++)
        {
            rdispls[i] = rdispls[i - 1] + recvcounts[i - 1];
            assert(rdispls[i] >= 0);
        }

        int* sdispls = new int[num_neighbors];
        sdispls[0]   = 0;
        for (int i = 1; i < num_neighbors; i++)
        {
            sdispls[i] = sdispls[i - 1] + sendcounts[i - 1];
            assert(sdispls[i] >= 0);
        }

        // pack and erase data to send
        T* send_buffer = new T[std::max(1, total_ndata2send)];
        for (int dst = 0; dst < num_neighbors; dst++)
        {
            const int pe_init = dst * fraction_tasks;
            const int pe_end  = std::min(pe_init + fraction_tasks, ntasks_mat_);
            assert(sdispls[dst] <= total_ndata2send);
            T* psend_buffer = send_buffer + sdispls[dst];
            int offset      = 0;
            for (int pe = pe_init; pe < pe_end; pe++)
            {
                size_t ndata = index_and_val_[pe].size();
                memcpy(psend_buffer + offset, &index_and_val_[pe][0],
                    sizeof(T) * ndata);
                index_and_val_[pe].clear(); // remove data we send out
                offset += ndata;
            }
            assert(offset == sendcounts[dst]);
        }

        T* recv_buffer = new T[std::max(total_ndata2recv, 1)];
        recv_buffer[0] = -1.;

#ifndef NDEBUG
        rc =
#endif
            Alltoallv(send_buffer, sendcounts, sdispls, recv_buffer, recvcounts,
                rdispls, sub_comm);
        assert(rc == MPI_SUCCESS);

        delete[] send_buffer;
        delete[] sdispls;
        delete[] sendcounts;

        // unpack data just received
        for (int src = 0; src < num_neighbors; src++)
        {
            const int src_init          = src * fraction_tasks;
            const int src_end           = src_init + fraction_tasks;
            const T* const precv_buffer = recv_buffer + rdispls[src];
            int offset                  = 0;
            for (int i = src_init; i < src_end; i++)
            {
                const int ndata = ndata2recv[i];
                for (int j = 0; j < ndata; j++)
                {
                    const int index = key * fraction_tasks + i - src_init;
                    // const int index=i;
                    assert(static_cast<unsigned int>(index)
                           < index_and_val_.size());
                    index_and_val_[index].push_back(precv_buffer[offset + j]);
                }
                offset += ndata;
            }
            if (offset != recvcounts[src])
                std::cout << "src=" << src << ", offset=" << offset
                          << ", recvcounts=" << recvcounts[src] << std::endl;
            assert(offset == recvcounts[src]);
        }

        delete[] recv_buffer;
        delete[] recvcounts;
        delete[] ndata2recv;
        delete[] rdispls;

        array2map(); // to sum up elements with same index
    }
    MPI_Comm_free(&sub_comm);
    consolidateArray_tm_.stop();
}

template <class T>
void SparseDistMatrix<T>::assign(const int size, const T* const val)
{
    assign_tm_.start();
    // std::cout<<"SparseDistMatrix<T>::assign()"<<endl;
    // std::cout<<"size="<<size<<endl;
    for (int k = 0; k < size; k = k + 2)
    {
        const int vi = (int)val[k];
        if (vi > -1)
        {
            const T vv  = val[k + 1];
            const int i = (vi >> SHIFT);
            const int j = vi - (i << SHIFT);
#ifndef NDEBUG
            const int mypr = mat_.myrow();
            const int mypc = mat_.mycol();
            const int pri  = mat_.pr(i);
            if (pri != mypr)
            {
                std::cerr << "mype_=" << mype_ << ", size=" << size
                          << ", vi=" << vi << ", val[" << k << "]=" << val[k]
                          << ", i=" << i << ", j=" << j << ", pri=" << pri
                          << ",mypr=" << mypr << ", vv=" << vv << std::endl;
                MPI_Barrier(comm_global_);
            }
            const int pcj = mat_.pc(j);
            if (pcj != mypc)
            {
                std::cerr << "mype_=" << mype_ << ", size=" << size
                          << ", vi=" << vi << ", val[" << k << "]=" << val[k]
                          << ", i=" << i << ", j=" << j << ", pcj=" << pcj
                          << ",mypc=" << mypc << ", vv=" << vv << std::endl;
                MPI_Barrier(comm_global_);
            }
#endif
            mat_.addval(i, j, vv);
#if !ZOLTAN
        }
        else
        {
            break;
#endif
        }
    }
    assign_tm_.stop();
}

template <class T>
void SparseDistMatrix<T>::setPartitioning(
    const int target_nb_tasks_per_partition)
{
    partition_comm_.reset(new MPI_DistMatrixCommunicator());
    int maxnpartitions = std::max(npes_ / target_nb_tasks_per_partition, 1);

    npartitions_ = maxnpartitions;
    while (npes_ % npartitions_ != 0)
    {
        npartitions_--;
    }

    // if( mype_==0 )
    //    std::cout<<"npes_="<<npes_
    //        <<", npartitions_="<<npartitions_<<endl;
    ntasks_per_partition_ = npes_ / npartitions_;

    const int color      = mype_ / ntasks_per_partition_;
    const int first_task = ntasks_per_partition_ * color;
    int key              = mype_ - first_task;
    int rc = MPI_Comm_split(comm_global_, color, key, &partition_comm_->comm());
    if (rc != MPI_SUCCESS)
    {
        std::cerr << "Error calling MPI_Comm_split:"
                  << " color=" << color << ", key=" << key << std::endl;
        mgmol_abort();
    }
}

#if MPIALLTOALL

template <class T>
void SparseDistMatrix<T>::parallelSumToDistMatrix()
{
#ifndef NDEBUG
    // print(cout);
#endif
    pSumToDistMatrix_tm_.start();

    maptoarray();

    int max_val_size   = 0;
    const int val_size = (int)index_and_val_.size();
    for (int i = 0; i < val_size; i++)
    {
        max_val_size = max(max_val_size, (int)index_and_val_[i].size());
        assert((int)index_and_val_[i].size() % 2 == 0);
    }
    int maxsize = max_val_size;

    mat_.clear();
    int rc;
    if (npes_ > 1)
        rc = MPI_Allreduce(
            &max_val_size, &maxsize, 1, MPI_INT, MPI_MAX, comm_global_);

    const size_t maxsized = maxsize * sizeof(T);
    const size_t sizeT    = sizeof(T);

    // put data into tmp array
    std::vector<T> tmp_val(npes_ * maxsize, -1.);
    for (int dest = 0; dest < npes_; dest++)
    {
        const int inddest = other_tasks_indexes_[dest];
        if (inddest > -1)
            memcpy(&tmp_val[0] + dest * maxsize, &index_and_val_[inddest][0],
                sizeT * index_and_val_[inddest].size());
    }

    T* buf_val = new T[npes_ * maxsize];
    pSumSendRecv_tm_.start();
    if (my_task_index_ == 0)
        std::cout << "Call MPI_Alltoall, size=" << maxsize << std::endl;
    Alltoall(&tmp_val[0], maxsize, buf_val, maxsize, comm_global_);
    // if(my_task_index_==0)
    //    std::cout<<"Call myMPI_Alltoall, size="<<maxsize<<endl;
    // myMPI_AlltoallDouble(&tmp_val[0], buf_val,maxsize,comm_global_);
    pSumSendRecv_tm_.stop();

    if (my_task_index_ > -1)
        for (int i = 0; i < npes_; i++)
            assign(maxsize, &buf_val[i * maxsize]);

    delete[] buf_val;

    pSumToDistMatrix_tm_.stop();
}

#else

#if ZOLTAN

TEMP_DECL
void SparseDistMatrix<double>::parallelSumToDistMatrix()
{
    // if( my_task_index_==0 )cout<<"parallelSumToDistMatrix() using
    // ZOLTAN"<<endl;

    assert(&mat_ != nullptr);
#ifdef DEBUG
    print(cout);
#endif
    pSumToDistMatrix_tm_.start();

    maptoarray();

    int max_index_and_val_size   = 0;
    const int index_and_val_size = (int)index_and_val_.size();
    for (int i = 0; i < index_and_val_size; i++)
        max_index_and_val_size
            = max(max_index_and_val_size, (int)index_and_val_[i].size());

    const bool recv = (my_task_index_ > -1);

    // make all data to transfer same size
    int newsize = max_index_and_val_size;
    if (npes_ > 1)
        MPI_Allreduce(&max_index_and_val_size, &newsize, 1, MPI_INT, MPI_MAX,
            comm_global_);
    assert(newsize > 0);
    for (int i = 0; i < index_and_val_size; i++)
    {
        index_and_val_[i].resize(newsize, -1.);
    }

    // Setup array of destination processor numbers for each of the objects to
    // be sent.
    int* proclist = new int[ntasks_mat_];
    int nsend     = 0;
    for (int dst = 0; dst < npes_; dst++)
    {
        const int dst_task_index = other_tasks_indexes_[dst];
        assert(dst_task_index < ntasks_mat_);
        if (dst_task_index >= 0)
            if (index_and_val_[dst_task_index].size() > 0)
            {
                proclist[nsend] = dst;
                nsend++;
            }
    }
    assert(nsend <= ntasks_mat_);

    const int tag = 17;
    int nreturn   = 0;
    Zoltan_Comm zcom(nsend, proclist, comm_global_, tag, &nreturn);
    if (recv)
    {
        assert(nreturn <= npes_);
        assert(nreturn > 0);
    }
    else
        assert(nreturn == 0);

    double* recvbuf;
    if (nreturn > 0) recvbuf = new double[newsize * nreturn];
    double* sendbuf;
    if (nsend > 0) sendbuf = new double[newsize * nsend];
    for (int i = 0; i < nsend; i++)
    {
        const int dst            = proclist[i];
        const int dst_task_index = other_tasks_indexes_[dst];
        assert(dst_task_index < ntasks_mat_);
        memcpy(&sendbuf[i * newsize], &index_and_val_[dst_task_index][0],
            newsize * sizeof(double));
    }

    // communicate data
    pSumSendRecv_tm_.start();

    const int nbytes = newsize * sizeof(double);
    zcom.Do(tag, (char*)sendbuf, nbytes, (char*)recvbuf);

    pSumSendRecv_tm_.stop();

    mat_.clear();
    if (recv)
        assign(newsize * nreturn,
            &recvbuf[0]); // assign values received to DistMatrix

    if (nsend > 0) delete[] sendbuf;
    if (nreturn > 0) delete[] recvbuf;
    delete[] proclist;

    pSumToDistMatrix_tm_.stop();
}

TEMP_DECL
void SparseDistMatrix<float>::parallelSumToDistMatrix()
{
    // if( my_task_index_==0 )cout<<"parallelSumToDistMatrix() using
    // ZOLTAN"<<endl;

    assert(&mat_ != nullptr);
#ifdef DEBUG
    print(cout);
#endif
    pSumToDistMatrix_tm_.start();

    maptoarray();

    int max_index_and_val_size   = 0;
    const int index_and_val_size = (int)index_and_val_.size();
    for (int i = 0; i < index_and_val_size; i++)
        max_index_and_val_size
            = max(max_index_and_val_size, (int)index_and_val_[i].size());

    const bool recv = (my_task_index_ > -1);

    // make all data to transfer same size
    int newsize = max_index_and_val_size;
    if (npes_ > 1)
        MPI_Allreduce(&max_index_and_val_size, &newsize, 1, MPI_INT, MPI_MAX,
            comm_global_);
    assert(newsize > 0);
    for (int i = 0; i < index_and_val_size; i++)
    {
        index_and_val_[i].resize(newsize, -1.);
    }

    // Setup array of destination processor numbers for each of the objects to
    // be sent.
    int* proclist = new int[ntasks_mat_];
    int nsend     = 0;
    for (int dst = 0; dst < npes_; dst++)
    {
        const int dst_task_index = other_tasks_indexes_[dst];
        assert(dst_task_index < ntasks_mat_);
        if (dst_task_index >= 0)
            if (index_and_val_[dst_task_index].size() > 0)
            {
                proclist[nsend] = dst;
                nsend++;
            }
    }
    assert(nsend <= ntasks_mat_);

    const int tag = 17;
    int nreturn   = 0;
    Zoltan_Comm zcom(nsend, proclist, comm_global_, tag, &nreturn);
    if (recv)
    {
        assert(nreturn <= npes_);
        assert(nreturn > 0);
    }
    else
        assert(nreturn == 0);

    float* recvbuf;
    if (nreturn > 0) recvbuf = new float[newsize * nreturn];
    float* sendbuf;
    if (nsend > 0) sendbuf = new float[newsize * nsend];
    for (int i = 0; i < nsend; i++)
    {
        const int dst            = proclist[i];
        const int dst_task_index = other_tasks_indexes_[dst];
        assert(dst_task_index < ntasks_mat_);
        memcpy(&sendbuf[i * newsize], &index_and_val_[dst_task_index][0],
            newsize * sizeof(float));
    }

    // communicate data
    pSumSendRecv_tm_.start();

    const int nbytes = newsize * sizeof(float);
    zcom.Do(tag, (char*)sendbuf, nbytes, (char*)recvbuf);

    pSumSendRecv_tm_.stop();

    mat_.clear();
    if (recv)
        assign(newsize * nreturn,
            &recvbuf[0]); // assign values received to DistMatrix

    if (nsend > 0) delete[] sendbuf;
    if (nreturn > 0) delete[] recvbuf;
    delete[] proclist;

    pSumToDistMatrix_tm_.stop();
}

#else // !ZOLTAN

template <class T>
void SparseDistMatrix<T>::sendRecvData()
{
    pSumSendRecv_tm_.start();

    mat_.clear();

#if MEMMON
    if (mype_ == 0)
    {
        std::cout << "sendRecvData" << std::endl;
        memmon_print_usage();
    }
#endif
    const bool recv = rtasks_distmatrix_->isMyTaskActive();

    int max_val_size   = 0;
    const int val_size = (int)index_and_val_.size();
    for (int i = 0; i < val_size; i++)
        max_val_size = std::max(max_val_size, (int)index_and_val_[i].size());

    int newsize = max_val_size;

    int color            = mype_ / ntasks_per_partition_;
    const int first_task = ntasks_per_partition_ * color;
    if (ntasks_per_partition_ > 1)
    {
        assert(partition_comm_->comm() != MPI_COMM_NULL);
        MPI_Allreduce(&max_val_size, &newsize, 1, MPI_INT, MPI_MAX,
            partition_comm_->comm());
    }
    MPI_Request rr   = MPI_REQUEST_NULL;
    MPI_Request sr1  = MPI_REQUEST_NULL;
    MPI_Request sr2  = MPI_REQUEST_NULL;
    MPI_Request* psr = (&sr1);

    // int dest=first_task+(mype_+1)%ntasks_per_partition_;
    int key       = mype_ - first_task;
    int dest_key  = (key + 1) % ntasks_per_partition_;
    int dest_task = first_task + dest_key;
    int src_key   = key - 1;
    if (src_key < 0) src_key += ntasks_per_partition_;

    const size_t newsized = newsize * sizeof(T);

    if (newsize == 0) return;
    T* tmp_val = recv ? new T[4 * newsize] : new T[2 * newsize];

    T* sbuf1_val = tmp_val;
    T* sbuf2_val = tmp_val + newsize;

    T* tbuf_val = nullptr;
    T* rbuf_val = nullptr;
    if (recv)
    {
        rbuf_val = tmp_val + 2 * newsize;
        tbuf_val = tmp_val + 3 * newsize;
    }
    T* sbuf_val = sbuf1_val;

    if (ntasks_per_partition_ > 1)
    {
        assert(partition_comm_->comm() != MPI_COMM_NULL);
        if (recv)
        {
            Irecv(rbuf_val, newsize, src_key, 0, partition_comm_->comm(), &rr);
        }
        if (rtasks_distmatrix_->isRemoteTaskActive(dest_task))
        {
            const int dst = rtasks_distmatrix_->getRemoteTask(dest_task);

            size_t size_index_and_val = index_and_val_[dst].size();
            assert(!(size_index_and_val % 2));
            assert(size_index_and_val <= static_cast<size_t>(newsize));
            if (size_index_and_val > 0)
                memcpy(sbuf_val, &index_and_val_[dst][0],
                    size_index_and_val * sizeof(T));
            if (size_index_and_val < static_cast<size_t>(newsize))
            {
                sbuf_val[size_index_and_val] = -1.;
            }
            Isend(sbuf_val, newsize, dest_key, 0, partition_comm_->comm(), psr);
        }
    }

    // local data
    if (recv)
    {
        const int mytask_index = rtasks_distmatrix_->getMyTaskIndex();
        if (index_and_val_[mytask_index].size() > 0)
            assign((int)index_and_val_[mytask_index].size(),
                &index_and_val_[mytask_index][0]);
    }

    for (int p = 0; p < ntasks_per_partition_ - 1; p++)
    {

        int tag = p + 1;
        // compute next destination/source
        dest_key  = (key + p + 2) % ntasks_per_partition_;
        dest_task = first_task + dest_key;
        src_key--;
        if (src_key < 0) src_key += ntasks_per_partition_;

        // copy received data into temporary arrays
        if (recv)
        {
            MPI_Wait(&rr, MPI_STATUS_IGNORE);
            // copy data only if size relevant data>0
            if (rbuf_val[0] > -0.5)
                memcpy(tbuf_val, rbuf_val, newsized);
            else
            {
                tbuf_val[0] = -1.;
            }
        }

        // do another send/recv if not last PE
        if (p < ntasks_per_partition_ - 2)
        {
            assert(dest_key < ntasks_per_partition_);
            assert(dest_key >= 0);
            assert(src_key < ntasks_per_partition_);
            assert(src_key >= 0);

            // post next receive for next iteration
            if (recv)
            {
                // receive
                // std::cout<<"PE: "<<mype_<<", receive data from PE
                // "<<src_key<<endl;
                assert(partition_comm_->comm() != MPI_COMM_NULL);
                Irecv(rbuf_val, newsize, src_key, tag, partition_comm_->comm(),
                    &rr);
            }

            // send data
            if (rtasks_distmatrix_->isRemoteTaskActive(dest_task))
            {
                // std::cout<<"PE: "<<mype_<<", send data to PE "<<dest<<endl;
                const int dst = rtasks_distmatrix_->getRemoteTask(dest_task);

                size_t size_index_and_val = index_and_val_[dst].size();
                assert(!(size_index_and_val % 2));
                assert(!(newsize % 2));
                assert(size_index_and_val <= static_cast<size_t>(newsize));

                if (p % 2)
                {
                    sbuf_val = sbuf1_val;
                    psr      = &sr1;
                }
                else
                {
                    sbuf_val = sbuf2_val;
                    psr      = &sr2;
                }
#ifndef NDEBUG
                int rc =
#endif
                    MPI_Wait(psr,
                        MPI_STATUS_IGNORE); // wait for previous send to
                                            // complete before reusing buffer
                assert(rc == MPI_SUCCESS);

                if (size_index_and_val > 0)
                    memcpy(&sbuf_val[0], &index_and_val_[dst][0],
                        size_index_and_val * sizeof(T));
                if (size_index_and_val < static_cast<size_t>(newsize))
                {
                    sbuf_val[size_index_and_val] = -1.;
                }

                assert(partition_comm_->comm() != MPI_COMM_NULL);
#ifndef NDEBUG
                rc =
#endif
                    Isend(sbuf_val, newsize, dest_key, tag,
                        partition_comm_->comm(), psr);
                assert(rc == MPI_SUCCESS);
            }
        }

        if (recv)
            assign(newsize, tbuf_val); // assign values received to DistMatrix
    }

#if MEMMON
    if (mype_ == 0)
    {
        std::cout << "sendRecvData end" << std::endl;
        memmon_print_usage();
    }
#endif

    // make sure the last send is completed before deleting buffer!
    MPI_Wait(&sr1, MPI_STATUS_IGNORE);
    MPI_Wait(&sr2, MPI_STATUS_IGNORE);
    delete[] tmp_val;

    // MPI_Barrier(comm_global_);
    pSumSendRecv_tm_.stop();
}

template <class T>
void SparseDistMatrix<T>::parallelSumToDistMatrix1()
{
#ifdef DEBUG
    print(std::cout);
#endif
    pSumToDistMatrix_tm_.start();

    maptoarray();

    ntasks_per_partition_ = npes_;

    sendRecvData();

    pSumToDistMatrix_tm_.stop();
}

template <class T>
void SparseDistMatrix<T>::parallelSumToDistMatrix2()
{
    assert(npartitions_ >= 0);
    assert(npartitions_ <= npes_);
#ifdef DEBUG
    print(std::cout);
#endif
    pSumToDistMatrix_tm_.start();

    maptoarray();

    MPI_Request rr1 = MPI_REQUEST_NULL;
    MPI_Request rr2 = MPI_REQUEST_NULL;
    MPI_Request sr1 = MPI_REQUEST_NULL;
    MPI_Request sr2 = MPI_REQUEST_NULL;

    // duplicate data among some processors
    // so that subsequently communication is needed only inside a partition

    if (npartitions_ > 1)
    {
        int* my_data_desc = new int[ntasks_mat_];

        int my_data_size = 0;
        for (int i = 0; i < ntasks_mat_; i++)
        {
            my_data_desc[i] = (int)index_and_val_[i].size();
            my_data_size += my_data_desc[i];
        }
        // if( my_task_index_==0 )cout<<"my_data_size="<<my_data_size<<endl;

        // pack local data for communication
        T* my_data  = new T[my_data_size];
        T* pmy_data = &my_data[0];
        for (int i = 0; i < ntasks_mat_; i++)
        {
            memcpy(
                pmy_data, &index_and_val_[i][0], my_data_desc[i] * sizeof(T));
            pmy_data += my_data_desc[i];
        }

        // exchange data with task mype+i*ntasks_per_partition_
        std::vector<int> remote_data_desc(ntasks_mat_);
        std::vector<T> remote_data;
        for (int i = 1; i < npartitions_; i++)
        {
            int dest = (mype_ + i * ntasks_per_partition_) % npes_;
            int src  = mype_ - i * ntasks_per_partition_;
            if (src < 0) src += npes_;

            int tag = 2 * i;
            MPI_Irecv(&remote_data_desc[0], ntasks_mat_, MPI_INT, src, tag,
                comm_global_, &rr1);
            MPI_Isend(&my_data_desc[0], ntasks_mat_, MPI_INT, dest, tag,
                comm_global_, &sr1);

            MPI_Wait(&rr1, MPI_STATUS_IGNORE);
            int remote_data_size = 0;
            for (int pe = 0; pe < ntasks_mat_; pe++)
            {
                remote_data_size += remote_data_desc[pe];
                static bool first_time = true;
                if (remote_data_size > 250000 && first_time) // 2MB
                {
                    std::cout
                        << "WARNING: mype_=" << mype_ << ", pe=" << pe
                        << ", remote_data_desc[pe]=" << remote_data_desc[pe]
                        << ", remote_data_size=" << remote_data_size
                        << std::endl;
                    first_time = false;
                }
            }
            if (remote_data_size > 0)
            {
                remote_data.resize(remote_data_size);
                Irecv(&remote_data[0], remote_data_size, src, tag + 1,
                    comm_global_, &rr2);
            }
            if (my_data_size > 0)
                Isend(&my_data[0], my_data_size, dest, tag + 1, comm_global_,
                    &sr2);

            // append received data to index_and_val_
            if (remote_data_size > 0)
            {
                T* premote_data = &remote_data[0];
                MPI_Wait(&rr2, MPI_STATUS_IGNORE);
                // if( my_task_index_==0 )cout<<"append received data to
                // index_and_val_ ..."<<endl;
                for (int pe = 0; pe < ntasks_mat_; pe++)
                {
#if 1
                    std::copy(premote_data, premote_data + remote_data_desc[pe],
                        back_inserter(index_and_val_[pe]));
#else
                    for (int i = 0; i < remote_data_desc[pe]; i++)
                        index_and_val_[pe].push_back(premote_data[i]);
#endif
                    premote_data += remote_data_desc[pe];
                }
            }
        }
        remote_data.clear();

        MPI_Barrier(comm_global_); // to make sure all the communications are
                                   // completed and memory can be released
        delete[] my_data;
        delete[] my_data_desc;
    }

    sendRecvData();

    pSumToDistMatrix_tm_.stop();
}

template <class T>
void SparseDistMatrix<T>::parallelSumToDistMatrix()
{
#if MEMMON
    if (mype_ == 0)
    {
        std::cout << "parallelSumToDistMatrix, npartitions_=" << npartitions_
                  << std::endl;
        memmon_print_usage();
    }
#endif
    // if( mype_==0 )cout<<"parallelSumToDistMatrix,
    // npartitions_="<<npartitions_<<endl; std::cout<<"My task = "<<mype_<<",
    // size(before)="<<size()<<endl;
    consolidateArray();
    // std::cout<<"My task = "<<mype_<<", size(after)="<<size()<<endl;

#if MEMMON
    if (mype_ == 0)
    {
        std::cout << "parallelSumToDistMatrix2" << std::endl;
        memmon_print_usage();
    }
#endif

    parallelSumToDistMatrix2();

#if MEMMON
    if (mype_ == 0)
    {
        memmon_print_usage();
    }
#endif
}

#endif // ZOLTAN

#endif

template class SparseDistMatrix<DISTMATDTYPE>;

} // namespace
