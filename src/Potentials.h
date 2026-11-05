// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_POTENTIALS_H
#define MGMOL_POTENTIALS_H

#include "HDFrestart.h"
#include "Rho.h"
#include "TriCubic.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

class Ions;
class Species;
template <class T>
class GridFunc;

class Potentials
{
    int size_;
    int gdim_[3];
    int dim_[3];
    bool diel_;

    double scf_dvrho_;
    double scf_dv_;

    double background_charge_;
    double charge_in_cell_;
    double ionic_charge_;

    /*!
     * Total KS potential seen by electrons
     */
    std::vector<POTDTYPE> vtot_;
    std::vector<POTDTYPE> vtot_old_;

    std::vector<POTDTYPE> vepsilon_;
    std::vector<POTDTYPE> vh_rho_;
    std::vector<POTDTYPE> vxc_rho_;

    /*
     * Potential contribution from atomic cores (local pseudopotential)
     */
    std::vector<POTDTYPE> v_nuc_;

    /*!
     * Optional external potential (read from input)
     * Used only in special cases.
     */
    std::vector<POTDTYPE> v_ext_;
#ifdef HAVE_TRICUBIC
    pb::TriCubic<POTDTYPE>* vext_tricubic_;
#endif

    /*!
     * Optional external potential from electrostatic field
     */
    std::vector<POTDTYPE> v_efield_;

    /*!
     * Potential associated with the sum of Gaussian charge distributions
     * compensating  the Coulomb potential of each atom
     */
    std::vector<POTDTYPE> v_comp_;

    /*!
     * Sum of Gaussian charge distributions compensating the Coulomb potential
     * of each atom
     */
    std::vector<RHODTYPE> rho_comp_;

    std::vector<POTDTYPE> dv_;

    /*!
     * Backpup copy of Hartree potential to save previous state
     */
    std::vector<POTDTYPE> vh_rho_backup_;

    /*!
     * Electrostatic field
     */
    double efield_[3];

    int itindex_vxc_;
    int itindex_vh_;

    short verbosity_level_;

    // filenames for various input potentials and their type
    // 0 = radial potential
    // 1 = filtered radial potential
    // 2 = xyz potential
    std::vector<std::string> pot_filenames_;
    std::vector<char> pot_types_;

    void evalNormDeltaVtotRho(const std::vector<std::vector<RHODTYPE>>& rho);

    void initializeRadialDataOnMesh(
        const Vector3D& position, const Species& sp);
    void initializeSupersampledRadialDataOnMesh(
        const Vector3D& position, const Species& sp);

    void initializeRadialDataOnSampledPts(const Vector3D& position,
        const Species& sp, const std::vector<int>& local_idx,
        std::vector<RHODTYPE>& sampled_rhoc);

    void rescaleRhoComp();

    void addBackgroundToRhoComp();

    void initBackground();

public:
    Potentials();

    ~Potentials();

    void setVerbosity(const short vlevel) { verbosity_level_ = vlevel; }

    void registerName(const std::string& filename, const char flag)
    {
        pot_filenames_.push_back(filename);
        pot_types_.push_back(flag);
    }

    void writeNames(std::ostream& os)
    {
        std::vector<std::string>::iterator p = pot_filenames_.begin();
        while (p != pot_filenames_.end())
        {
            os << " Potential file: " << *p << std::endl;
            p++;
        }
    }

    char pot_type(const int isp) const
    {
        assert(isp < (int)pot_types_.size());
        return pot_types_[isp];
    }

    int getIterativeIndex() const
    {
        assert(itindex_vxc_ >= 0);
        assert(itindex_vh_ >= 0);
        assert(itindex_vxc_ == itindex_vh_);
        return itindex_vh_;
    }

    void turnOnDiel() { diel_ = true; }

    int write(HDFrestart& h5f_file);
    int read(HDFrestart& h5f_file);

    int size() const { return size_; }

    double scf_dvrho(void) const { return scf_dvrho_; }
    double scf_dv(void) const { return scf_dv_; }
    POTDTYPE* vtot() { return vtot_.data(); }
    RHODTYPE* rho_comp() { return rho_comp_.data(); }

    const std::vector<POTDTYPE>& dv() { return dv_; }

    const std::vector<POTDTYPE>& vnuc() const { return v_nuc_; }
    const std::vector<POTDTYPE>& vh_rho() const { return vh_rho_; }

    POTDTYPE* vepsilon() { return vepsilon_.data(); }

    void axpVcompToVh(const double alpha);

    bool diel() const { return diel_; }

    double getChargeInCell() const { return charge_in_cell_; }

    const double getBackgroundCharge() const { return background_charge_; }

    void getVofRho(std::vector<POTDTYPE>& vrho) const;

    double getEfield(short int dir) { return efield_[dir]; }

    /*!
     * evaluate potential correction associated with a new rho
     */
    double computeDeltaV(const std::vector<std::vector<RHODTYPE>>& rho);

    /*!
     * update total potential with updated components
     */
    double updateVtot(const std::vector<std::vector<RHODTYPE>>& rho);

    /*!
     * update potentials based on potential correction delta v and mixing
     * parameter
     */
    void updateVtot(const double mix);

    double max() const;
    double min() const;
    void readAll(std::vector<Species>& sp);

    template <typename T>
    void setVxc(const T* const vxc, const int iterativeIndex);
    void setVh(const pb::GridFunc<POTDTYPE>& vh, const int iterativeIndex);

    void initialize(Ions& ions);

    /*!
     * Setup linear potential based on values of exetrostatic field ex, ey, ez
     */
    void setupVefield(const double ex, const double ey, const double ez);

    /*!
     * Save current Hartree potential into backup array
     */
    void backupVh();

    void resetVhRho2Backup() { vh_rho_ = vh_rho_backup_; }

    void evalIonDensityOnSamplePts(Ions& ions,
        const std::vector<int>& local_idx, std::vector<RHODTYPE>& sampled_rhoc);

#ifdef HAVE_TRICUBIC
    void readExternalPot(const string filename, const char type);
    void setupVextTricubic();
    bool withVext() const;
    void getGradVext(const double[3], double[3]) const;
    void getValVext(
        const std::vector<double>& r, std::vector<double>& val) const;
#endif
};

#endif
