// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "Preconditioning.h"
#include "LapFactory.h"

template <typename T>
Preconditioning<T>::Preconditioning(const pb::LapType lap_type,
    const short maxlevels, const short npresmooth, const short npostsmooth,
    const pb::Grid& grid, const short bcWF[3])
    : max_levels_(maxlevels), npresmooth_(npresmooth), npostsmooth_(npostsmooth)
{
    assert(npresmooth_ >= 0);
    assert(npostsmooth_ >= 0);
    assert(npresmooth_ < 100);
    assert(npostsmooth_ < 100);

    for (short i = 0; i < 3; i++)
        bc_[i] = bcWF[i];

    pb::Grid* mygrid = new pb::Grid(grid);
    grid_.push_back(mygrid);

    pb::Lap<T>* myoper = LapFactory<T>::createLap(*grid_[0], lap_type);
    jacobi_factor_.push_back(myoper->jacobiFactor());
}

template <typename T>
Preconditioning<T>::~Preconditioning()
{
    clear();
}

template <typename T>
void Preconditioning<T>::clear()
{
    for (short i = 0; i < (short)gf_work_.size(); i++)
    {
        assert(gf_work_[i] != nullptr);
        delete gf_work_[i];
    }
    for (short i = 0; i < (short)gf_rcoarse_.size(); i++)
    {
        assert(gf_rcoarse_[i] != nullptr);
        delete gf_rcoarse_[i];
    }
    for (short i = 0; i < (short)gf_newv_.size(); i++)
    {
        assert(gf_newv_[i] != nullptr);
        delete gf_newv_[i];
    }
    // delete grids after pb::GridFunc<T> objects since those
    // have data members references to grids
    for (short i = 0; i < (short)grid_.size(); i++)
    {
        delete grid_[i];
    }
    grid_.clear();
    gf_work_.clear();
    gf_rcoarse_.clear();
    gf_newv_.clear();

    for (unsigned short l = 0; l < gfv_work_.size(); l++)
    {
        delete gfv_work_[l];
    }
    for (unsigned short l = 0; l < gfv_rcoarse_.size(); l++)
    {
        delete gfv_rcoarse_[l];
        delete gfv_newv_[l];
    }
    gfv_work_.clear();
    gfv_rcoarse_.clear();
    gfv_newv_.clear();
}

template <typename T>
void Preconditioning<T>::reset(
    const std::vector<std::vector<int>>& overlapping_gids)
{
    clear();

    setup(overlapping_gids);
}

template <typename T>
void Preconditioning<T>::setup(
    const std::vector<std::vector<int>>& overlapping_gids)
{
    assert(overlapping_gids.size() > 0);

    // fine level
    pb::GridFunc<T>* gf_work
        = new pb::GridFunc<T>(*grid_[0], bc_[0], bc_[1], bc_[2]);
    gf_work_.push_back(gf_work);

    pb::GridFuncVector<T, memory_space_type>* gfv_work
        = new pb::GridFuncVector<T, memory_space_type>(
            *grid_[0], bc_[0], bc_[1], bc_[2], overlapping_gids);
    gfv_work_.push_back(gfv_work);

    // coarse levels
    pb::Grid* mygrid = grid_[0];
    for (short ln = 1; ln <= max_levels_; ln++)
    {
        pb::Grid* coarse_grid = new pb::Grid(mygrid->coarse_grid());
        grid_.push_back(coarse_grid);

        // use 2nd order FD operator
        pb::Lap<T>* myoper
            = LapFactory<T>::createLap(*coarse_grid, pb::LapType::FD2nd);
        jacobi_factor_.push_back(myoper->jacobiFactor());

        gf_work = new pb::GridFunc<T>(*coarse_grid, bc_[0], bc_[1], bc_[2]);
        gf_work_.push_back(gf_work);

        pb::GridFunc<T>* gf_rcoarse
            = new pb::GridFunc<T>(*coarse_grid, bc_[0], bc_[1], bc_[2]);
        gf_rcoarse_.push_back(gf_rcoarse);
        pb::GridFunc<T>* gf_newv
            = new pb::GridFunc<T>(*coarse_grid, bc_[0], bc_[1], bc_[2]);
        gf_newv_.push_back(gf_newv);

        mygrid = coarse_grid;

        gfv_work = new pb::GridFuncVector<T, memory_space_type>(
            *coarse_grid, bc_[0], bc_[1], bc_[2], overlapping_gids);
        gfv_work_.push_back(gfv_work);

        pb::GridFuncVector<T, memory_space_type>* gfv_rcoarse
            = new pb::GridFuncVector<T, memory_space_type>(
                *coarse_grid, bc_[0], bc_[1], bc_[2], overlapping_gids);
        gfv_rcoarse_.push_back(gfv_rcoarse);

        pb::GridFuncVector<T, memory_space_type>* gfv_newv
            = new pb::GridFuncVector<T, memory_space_type>(
                *coarse_grid, bc_[0], bc_[1], bc_[2], overlapping_gids);
        gfv_newv_.push_back(gfv_newv);
    }
}

// MG V-cycle
template <typename T>
void Preconditioning<T>::mg(pb::GridFuncVector<T, memory_space_type>& gfv_v,
    const pb::GridFuncVector<T, memory_space_type>& gfv_f,
    const pb::LapType lap_type, const short level)
{
#ifdef PRINT_OPERATIONS
    if (onpe0)
        (*MPIdata::sout) << "Preconditioning::mg() at level " << level << endl;
#endif
    assert(static_cast<int>(gfv_work_.size()) > level);
    assert(gfv_work_[level] != nullptr);

    short ncycl = npresmooth_;
    if (level == max_levels_) ncycl = npresmooth_ + npostsmooth_;

    const double jacobi_factor = jacobi_factor_[level];

    // SMOOTHING
    for (short it = 0; it < ncycl; it++)
    {
        gfv_v.jacobi(lap_type, gfv_f, *gfv_work_[level], jacobi_factor);
        gfv_v.app_mask(level);
    }

    if (level == max_levels_) return;

    // COARSE GRID CORRECTION

    // LOCALIZATION
    gfv_work_[level]->app_mask(level);

    // restrictions
    pb::GridFuncVector<T, memory_space_type>* rcoarse = gfv_rcoarse_[level];
    assert(rcoarse != nullptr);
    gfv_work_[level]->restrict3D(*rcoarse);

    // LOCALIZATION
    rcoarse->app_mask(level + 1);

    // storage functions for coarse grid
    pb::GridFuncVector<T, memory_space_type>* newv = gfv_newv_[level];

    // call mgrid solver on a coarser level
    newv->resetData();
    mg(*newv, *rcoarse, pb::LapType::FD2nd, level + 1);

    gfv_work_[level]->extend3D(*newv);

    // LOCALIZATION
    gfv_work_[level]->app_mask(level);

    gfv_v -= (*gfv_work_[level]);

    // post-smoothing
    for (short it = 0; it < npostsmooth_; it++)
    {
        gfv_v.jacobi(lap_type, gfv_f, *gfv_work_[level], jacobi_factor);
        gfv_v.app_mask(level);
    }

    if (bc_[0] != 1 || bc_[2] != 1 || bc_[2] != 1) gfv_v.trade_boundaries();
}

template class Preconditioning<float>;
template class Preconditioning<double>;
