// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef GRIDFUNCVECTOR_H
#define GRIDFUNCVECTOR_H

#include "FDkernels.h"
#include "GridFunc.h"
#include "Lap.h"
#include "Map2Masks.h"
#include "memory_space.h"

#include <map>
#include <memory>
#include <type_traits>
#include <vector>

namespace pb
{
#ifdef HAVE_OPENMP_OFFLOAD
template <typename ScalarType, typename MemorySpaceType = MemorySpace::Device>
#else
template <typename ScalarType, typename MemorySpaceType = MemorySpace::Host>
#endif
class GridFuncVector
{
    static Timer trade_bc_tm_;
    static Timer trade_bc_colors_tm_;
    static Timer prod_tm_;
    static Timer finishExchangeNorthSouth_tm_;
    static Timer finishExchangeUpDown_tm_;
    static Timer finishExchangeEastWest_tm_;
    static Timer wait_north_south_tm_;
    static Timer wait_up_down_tm_;
    static Timer wait_east_west_tm_;
    static Timer copy_tm_;

    static Map2Masks* map2masks_;

    // block of memory for all GridFunc
    std::unique_ptr<ScalarType> memory_;

    using MemoryST = MemorySpace::Memory<ScalarType, MemorySpaceType>;

    // global id for functions in each subdivision
    const std::vector<std::vector<int>>& gid_;

    // gids to communicate to neighbors in each of 6 directions
    std::vector<std::vector<int>> remote_gids_;

    const Grid& grid_;
    const MPI_Comm comm_;

    std::vector<GridFunc<ScalarType>*> functions_;
    std::map<int, short> gid2lid_;

    // number of functions in functions_ (functions_.size())
    int nfunc_;

    int nfunc_max_global_;
    short nsubdivx_;

    bool up_;
    bool down_;
    bool east_;
    bool west_;
    bool north_;
    bool south_;

    bool updated_boundaries_;
    int bc_[3];
    short nghosts_;

    int dimx_;
    int dimy_;
    int dimz_;

    int incx_;
    int incy_;
    int dimxy_;
    int incxy_;
    size_t east_west_size_;
    size_t south_north_size_;
    size_t up_down_size_;

    const bool skinny_stencil_;

    int nfunc4buffers_;

    // block of memory for device memory
    std::unique_ptr<ScalarType, void (*)(ScalarType*)> memory_dev_;

    std::vector<std::vector<ScalarType>> comm_buf1_;
    std::vector<std::vector<ScalarType>> comm_buf2_;
    std::vector<std::vector<ScalarType>> comm_buf3_;
    std::vector<std::vector<ScalarType>> comm_buf4_;

    MPI_Request req_east_west_[4];
    MPI_Request req_north_south_[4];
    MPI_Request req_up_down_[4];

    void wait_north_south();
    void wait_east_west();
    void wait_up_down();

    void allocate_buffers(const int nfunc);

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void initiateNorthSouthComm(const int begin_color, const int end_color);

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void initiateNorthSouthComm(const int begin_color, const int end_color);

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void finishNorthSouthComm();

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void finishNorthSouthComm();

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void initiateUpDownComm(const int begin_color, const int end_color);

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void initiateUpDownComm(const int begin_color, const int end_color);

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void finishUpDownComm();

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void finishUpDownComm();

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void initiateEastWestComm(const int begin_color, const int end_color);

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void initiateEastWestComm(const int begin_color, const int end_color);

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void finishEastWestComm();

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void finishEastWestComm();

    void communicateRemoteGids(const int begin_color, const int end_color);

    void allocate(const int n);

public:
    GridFuncVector(const Grid& my_grid, const int px, const int py,
        const int pz, const std::vector<std::vector<int>>& gid,
        const bool skinny_stencil = false)
        : gid_(gid),
          grid_(my_grid),
          comm_(my_grid.mype_env().comm()),
          skinny_stencil_(skinny_stencil),
          nfunc4buffers_(0),
// just for now
#ifdef HAVE_OPENMP_OFFLOAD
          memory_dev_(MemoryST::allocate(gid[0].size() * my_grid.sizeg()),
              MemoryST::free)
#else
          memory_dev_(nullptr, nullptr)
#endif
    {
        bc_[0] = px;
        bc_[1] = py;
        bc_[2] = pz;

        updated_boundaries_ = false; // boundaries not initialized

        allocate(gid[0].size());

        setup();
    }

    ~GridFuncVector()
    {
        assert(static_cast<int>(functions_.size()) == nfunc_);
        for (int i = 0; i < nfunc_; i++)
        {
            assert(functions_[i] != 0);
            delete functions_[i];
        }
    }

    void setup();

    static void setMasks(Map2Masks* map2masks) { map2masks_ = map2masks; }

    const Grid& grid() const { return grid_; }

    ScalarType* data() { return memory_.get(); }

    ScalarType* getDataPtr(const int ifunc, const int index = 0) const
    {
        return memory_.get() + ifunc * grid_.sizeg() + index;
    }

    bool getUpdatedBoundariesFlag() const { return updated_boundaries_; }

    // assign values to one GridFunc from values in array src
    // (without ghosts)
    template <typename ScalarType2>
    void assign(const int i, const ScalarType2* const src, const char dis = 'd')
    {
        assert(i < static_cast<int>(functions_.size()));
        assert(functions_[i] != nullptr);

        functions_[i]->assign(src, dis);
        updated_boundaries_ = false;
    }

    // extract values from one GridFunc into array dst
    // (without ghosts)
    void init_vect(const int k, ScalarType* dst, const char dis) const
    {
        functions_[k]->init_vect(dst, dis);
    }

#ifdef HAVE_OPENMP_OFFLOAD
    void copyHtoD(int size)
    {
        MemorySpace::copy_to_dev(memory_.get(), size, memory_dev_.get());
    }

    void copyDtoH(int size)
    {
        MemorySpace::copy_to_host(memory_dev_.get(), size, memory_.get());
    }
#endif

    GridFunc<ScalarType>& getGridFunc(const int k)
    {
        assert(k < static_cast<int>(functions_.size()));
        assert(functions_[k] != nullptr);

        return *functions_[k];
    }

    const GridFunc<ScalarType>& getGridFunc(const int k) const
    {
        assert(k < static_cast<int>(functions_.size()));
        assert(functions_[k] != nullptr);

        return *functions_[k];
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void del2_4th(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelDel2_4th(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void del2_4th(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        // assume the CPU data is uptodate for now...
        FDkernelDel2_4th(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void del2_4th_Mehr(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelDel2_4th_Mehr(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void del2_4th_Mehr(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        // assume the CPU data is uptodate for now...
        FDkernelDel2_4th_Mehr(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void rhs_4th_Mehr1(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelRHS_4th_Mehr1(grid(), data(), rhs.data(), grid().ghost_pt(),
            size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void rhs_4th_Mehr1(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        // assume the CPU data is uptodate for now...
        FDkernelRHS_4th_Mehr1(grid(), data(), rhs.data(), grid().ghost_pt(),
            size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void rhs_4th_Mehr1(ScalarType* rhs)
    {
        trade_boundaries();

        FDkernelRHS_4th_Mehr1(
            grid(), data(), rhs, 0, size(), MemorySpace::Host());
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void del2_2nd(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelDel2_2nd(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void del2_2nd(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        // assume the CPU data is uptodate for now...
        FDkernelDel2_2nd(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void del2_6th(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelDel2_6th(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void del2_6th(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelDel2_6th(
            grid(), data(), rhs.data(), size(), MemorySpace::Device());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Host, MST>::value>::type* = nullptr>
    void del2_8th(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelDel2_8th(
            grid(), data(), rhs.data(), size(), MemorySpace::Host());

        rhs.set_updated_boundaries(0);
    }

    template <typename MST = MemorySpaceType,
        typename std::enable_if<
            std::is_same<MemorySpace::Device, MST>::value>::type* = nullptr>
    void del2_8th(GridFuncVector<ScalarType, MemorySpaceType>& rhs)
    {
        trade_boundaries();

        FDkernelDel2_8th(
            grid(), data(), rhs.data(), size(), MemorySpace::Device());

        rhs.set_updated_boundaries(0);
    }

    void trade_boundaries();
    void trade_boundaries_colors(const short, const short);

    size_t size() const { return nfunc_; }

    // pointwise products this=A*B for each vector in this
    template <typename ScalarType2>
    void pointwiseProduct(GridFuncVector<ScalarType, MemorySpaceType>& A,
        const GridFunc<ScalarType2>& B);

    void extend3D(GridFuncVector<ScalarType, MemorySpaceType>&);
    void restrict3D(GridFuncVector<ScalarType, MemorySpaceType>&);
    void resetData()
    {
        memset(memory_.get(), 0, nfunc_ * grid_.sizeg() * sizeof(ScalarType));
        updated_boundaries_ = true;
    }
    void set_updated_boundaries(const bool flag) { updated_boundaries_ = flag; }
    GridFuncVector<ScalarType, MemorySpaceType>& operator-=(
        const GridFuncVector<ScalarType, MemorySpaceType>& func);

    template <typename ScalarType2>
    void axpy(const ScalarType2 alpha,
        const GridFuncVector<ScalarType, MemorySpaceType>& func);

    template <typename ScalarType2>
    void copyFrom(const GridFuncVector<ScalarType2, MemorySpaceType>& src);

    template <typename InputScalarType>
    void getValues(const int k, InputScalarType* vv) const;

    void applyLap(
        const LapType type, GridFuncVector<ScalarType, MemorySpaceType>& rhs);
    void applyRHS(
        const LapType type, GridFuncVector<ScalarType, MemorySpaceType>& rhs);
    void jacobi(const LapType type,
        const GridFuncVector<ScalarType, MemorySpaceType>& B,
        GridFuncVector<ScalarType, MemorySpaceType>& w, const double scale);

    void app_mask(const short level);

    static void printTimers(std::ostream& os)
    {
        trade_bc_tm_.print(os);
        trade_bc_colors_tm_.print(os);
        prod_tm_.print(os);
        wait_north_south_tm_.print(os);
        wait_up_down_tm_.print(os);
        wait_east_west_tm_.print(os);
        finishExchangeNorthSouth_tm_.print(os);
        finishExchangeUpDown_tm_.print(os);
        finishExchangeEastWest_tm_.print(os);
        copy_tm_.print(os);
    }
};

template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::trade_bc_tm_(
    "GridFuncVector::trade_bc_" + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::trade_bc_colors_tm_(
    "GridFuncVector::trade_bc_colors" + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::prod_tm_(
    "GridFuncVector::prod" + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::finishExchangeNorthSouth_tm_(
    "GridFuncVector::finishExNorthSouth_"
    + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::finishExchangeUpDown_tm_(
    "GridFuncVector::finishExUpDown_" + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::finishExchangeEastWest_tm_(
    "GridFuncVector::finishExEastWest_"
    + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::wait_north_south_tm_(
    "GridFuncVector::waitNS_" + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::wait_up_down_tm_(
    "GridFuncVector::waitUD_" + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::wait_east_west_tm_(
    "GridFuncVector::waitEW_" + std::to_string(sizeof(ScalarType) * 8));
template <typename ScalarType, typename MemorySpaceType>
Timer GridFuncVector<ScalarType, MemorySpaceType>::copy_tm_(
    "GridFuncVector::copy_" + std::to_string(sizeof(ScalarType) * 8));

template <typename ScalarType, typename MemorySpaceType>
Map2Masks* GridFuncVector<ScalarType, MemorySpaceType>::map2masks_(nullptr);
} // namespace pb

#endif
