// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "LBFGS_IonicStepper.h"
#include "MGmol_MPI.h"
#include "MGmol_blas1.h"
#include "MPIdata.h"
#include "hdf5.h"
#include "mcstep.h"

#include <cmath>
#include <iomanip>
#include <iostream>

static const int nb_attri = 11;

#define MGMOL_LBFGS_FAIL(X)                                                    \
    {                                                                          \
        (*MPIdata::sout) << "LBFGS failure:" << std::endl;                     \
        (*MPIdata::sout) << "Error Message: " << X << std::endl;               \
    }

LBFGS_IonicStepper::LBFGS_IonicStepper(const double dt,
    const std::vector<short>& atmove, std::vector<double>& tau0,
    std::vector<double>& taup, std::vector<double>& fion,
    const std::vector<int>& gid, const int m, double* etot)
    : IonicStepper(dt, atmove, tau0, taup), fion_(fion), gids_(gid), etot_(etot)
{
    assert(gid.size() == atmove.size());
    assert(3 * gid.size() == fion.size());

    MGmol_MPI& mmpi = *(MGmol_MPI::instance());

    eps_                = 2.e-6;
    xtol_               = 1.e-15;
    last_step_accepted_ = 1;

    info_  = 0;
    iflag_ = 0;
    point_ = 0;
    iter_  = 0;
    npt_   = 0;

    // count total number of atoms
    int na  = (int)atmove_.size();
    int tmp = 0;
    mmpi.allreduce(&na, &tmp, 1, MPI_SUM);
    na = tmp;

    for (int i = 0; i < 3; i++)
        etot_[i] = 0.;

    // count total number of d.o.f.
    const int nmove = (int)atmove_.size();
    gndofs_         = 0;
    for (int ia = 0; ia < nmove; ia++)
    {
        // if ( atmove_[ia] )
        gndofs_ += 3;
    }

    mmpi.allreduce(&gndofs_, &tmp, 1, MPI_SUM);
    gndofs_ = tmp;

    m_ = std::min(m, gndofs_);
    if (onpe0) (*MPIdata::sout) << " LBFGS with m=" << m_ << std::endl;

    freeze_geom_center_ = (3 * na == gndofs_);
    if (onpe0)
        if (freeze_geom_center_)
            (*MPIdata::sout)
                << " LBFGS with geometry center frozen" << std::endl;

    // allocate duplicated arrays used in LBFGS computations
    xcurrent_.resize(gndofs_);
    g_.resize(gndofs_);
    work_.resize(gndofs_ * 2 * (m_ + 1) + 2 * m_);
    xref_.resize(gndofs_);
    diag_.resize(gndofs_);

    setDataFromLocalData(xref_, tau0);

    //(*MPIdata::sout)<<"gndofs_="<<gndofs_<<", m_="<<m_<<endl;
    assert(gndofs_ > 0);
    assert(m_ > 0);
}

int LBFGS_IonicStepper::writeDoubleAtt(hid_t dataset_id) const
{
    // attribute: 16 double
    std::string attname = "double parameters";
    hsize_t dim         = 16;

    // Create dataset attribute.
    hid_t dataspace_id = H5Screate_simple(1, &dim, nullptr);
    if (dataspace_id < 0)
    {
        MGMOL_LBFGS_FAIL("Attribute " << attname << ": H5Screate failed!!!");
        return -1;
    }

    hid_t attribute_id = H5Acreate2(dataset_id, attname.c_str(),
        H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute_id < 0)
    {
        MGMOL_LBFGS_FAIL("Attribute " << attname << ": H5Acreate failed!!!");
        return -1;
    }

    std::vector<double> attr_d(dim);
    attr_d[0]  = fx_;
    attr_d[1]  = fy_;
    attr_d[2]  = stx_;
    attr_d[3]  = sty_;
    attr_d[4]  = dgx_;
    attr_d[5]  = dgy_;
    attr_d[6]  = stmin_;
    attr_d[7]  = stmax_;
    attr_d[8]  = width_;
    attr_d[9]  = width1_;
    attr_d[10] = finit_;
    attr_d[11] = dginit_;
    attr_d[12] = etot_[0];
    attr_d[13] = etot_[1];
    attr_d[14] = etot_[2];
    attr_d[15] = stp_;

    //(*MPIdata::sout)<<"Write attribute "<<attname<<endl;
    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &attr_d[0]);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL("Attribute " << attname << ": H5Awrite failed!!!");
        return -1;
    }

    status = H5Aclose(attribute_id);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL(
            "LBFGS_IonicStepper::writeDoubleAtt(): H5Aclose failed!!!");
        return -1;
    }

    status = H5Sclose(dataspace_id);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL(
            "LBFGS_IonicStepper::writeDoubleAtt(): H5Sclose failed!!!");
        return -1;
    }

    return 0;
}

void LBFGS_IonicStepper::setDataFromLocalData(
    std::vector<double>& x, const std::vector<double>& tau)
{
    // set to 0. before summing up across MPI tasks
    memset(&x[0], 0, gndofs_ * sizeof(double));

    const int na = (int)atmove_.size();
    for (int ia = 0; ia < na; ia++)
    {
        const int gid = gids_[ia];
        for (short j = 0; j < 3; j++)
            x[3 * gid + j] = tau[3 * ia + j];
    }

    // consolidate data across all pes
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&x[0], gndofs_, MPI_SUM);
}

int LBFGS_IonicStepper::writeIntAtt(hid_t dataset_id) const
{
    // attribute: nb_attri int
    std::string attname = "int parameters";
    hsize_t dim         = nb_attri;

    // Create dataspace and attribute
    hid_t dataspace_id = H5Screate_simple(1, &dim, nullptr);
    if (dataspace_id < 0)
    {
        MGMOL_LBFGS_FAIL("Attribute " << attname << ": H5Screate failed!!!");
        return -1;
    }

    hid_t attribute_id = H5Acreate2(dataset_id, attname.c_str(), H5T_NATIVE_INT,
        dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
    if (attribute_id < 0)
    {
        MGMOL_LBFGS_FAIL("Attribute " << attname << ": H5Acreate failed!!!");
        return -1;
    }

    std::vector<int> attr_i(dim);
    attr_i[0]  = iflag_;
    attr_i[1]  = m_;
    attr_i[2]  = iter_;
    attr_i[3]  = point_;
    attr_i[4]  = info_;
    attr_i[5]  = infoc_;
    attr_i[6]  = nfev_;
    attr_i[7]  = npt_;
    attr_i[8]  = (int)brackt_;
    attr_i[9]  = (int)stage1_;
    attr_i[10] = last_step_accepted_;

    herr_t status = H5Awrite(attribute_id, H5T_NATIVE_INT, &attr_i[0]);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL("Attribute " << attname << ": H5Awrite failed!!!");
        return -1;
    }

    status = H5Aclose(attribute_id);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL("Attribute " << attname << ": H5Aclose failed!!!");
        return -1;
    }

    status = H5Sclose(dataspace_id);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL(
            "LBFGS_IonicStepper::writeIntAtt(): H5Sclose failed!!!");
        return -1;
    }

    return 0;
}

int LBFGS_IonicStepper::writeLBFGSinfo(HDFrestart& h5f_file)
{
    assert(gndofs_ == (int)diag_.size());
    assert(gndofs_ == (int)xref_.size());

    hid_t file_id = h5f_file.file_id();
    if (file_id < 0) return 0;

    hsize_t dim = work_.size() + 2 * gndofs_;

    // Create the data space for the dataset.
    hid_t dataspace_id = H5Screate_simple(1, &dim, nullptr);
    if (dataspace_id < 0)
    {
        MGMOL_LBFGS_FAIL("writeLBFGSinfo(), H5Screate_simple failed!!!");
        return -1;
    }

    // Create dataset.
    std::string name("/LBFGS");
    hid_t dataset_id = H5Dcreate2(file_id, name.c_str(), H5T_NATIVE_DOUBLE,
        dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0)
    {
        MGMOL_LBFGS_FAIL(
            "writeLBFGSinfo(), H5Dcreate2 " << name << " failed!!!");
        return -1;
    }

    // Write the dataset.
    if (onpe0)
    {
        std::vector<double> u_buffer(dim);
        double* u = u_buffer.data();
        memcpy(u, &work_[0], work_.size() * sizeof(double));
        memcpy(u + work_.size(), &diag_[0], gndofs_ * sizeof(double));
        memcpy(u + work_.size() + gndofs_, &xref_[0], gndofs_ * sizeof(double));

        herr_t status = H5Dwrite(
            dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, u);
        if (status < 0)
        {
            MGMOL_LBFGS_FAIL("writeLBFGSinfo(): H5Dwrite failed!!!");
            return status;
        }
    }

    int return_status = writeDoubleAtt(dataset_id);
    if (return_status != 0) return return_status;

    return_status = writeIntAtt(dataset_id);
    if (return_status != 0) return return_status;

    herr_t status = H5Sclose(dataspace_id);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL("writeLBFGSinfo(): H5Sclose failed!!!");
        return -1;
    }

    status = H5Dclose(dataset_id);
    if (status < 0)
    {
        MGMOL_LBFGS_FAIL("writeLBFGSinfo(): H5Dclose failed!!!");
        return -1;
    }

    if (onpe0) (*MPIdata::sout) << "LBFGS restart info written" << std::endl;

    return return_status;
}

// return 1 if no data read
// return 0 if restart data read
// return -1 if error occurs
int LBFGS_IonicStepper::read_lbfgs(HDFrestart& h5f_file)
{
    if (onpe0) (*MPIdata::sout) << "LBFGS_IonicStepper::read_lbfgs()...\n";

    hid_t file_id = h5f_file.file_id();

    hsize_t dim_dset;
    hsize_t maxdim;
    hid_t dataset_id = H5P_DEFAULT;
    herr_t status;
    const std::string error_string
        = "!!!Error in LBFGS_IonicStepper::read_lbfgs(): ";

    short check_data = 0;
    std::vector<double> attr_d(16);
    // Open an existing dataset.
    std::string datasetname("/LBFGS");
    int err_id = h5f_file.checkDataExists(datasetname);
    if (onpe0)
    {
        if (err_id < 0)
        { // dataset does not exists
            (*MPIdata::sout) << "Warning: no dataset " << datasetname
                             << " -> no restart info for LBFGS" << std::endl;
        }
        else
        {
            dataset_id = H5Dopen2(file_id, "/LBFGS", H5P_DEFAULT);
            if (dataset_id < 0)
            {
                (*MPIdata::sout) << "Warning: H5Dopen failed for /LBFGS-> "
                                    "no restart info for LBFGS"
                                 << std::endl;
            }
            else
            {
                (*MPIdata::sout)
                    << "Read LBFGS information from PE 0" << std::endl;
                check_data = 1;

                // 1st attribute: 16 double
                std::string attname = "double parameters";

                //  Open a dataset attribute.
                hid_t attribute_id = H5Aopen_name(dataset_id, attname.c_str());
                if (attribute_id < 0)
                {
                    (*MPIdata::serr) << "PE " << mype << ": " << error_string
                                     << "H5Aopen failed for " << attname
                                     << "!!!" << std::endl;
                    return -1;
                }

                hid_t attdataspace = H5Aget_space(attribute_id);
                int rank           = H5Sget_simple_extent_dims(
                    attdataspace, &dim_dset, &maxdim);
                if (rank != 1)
                {
                    MGMOL_LBFGS_FAIL(error_string
                                     << "error in rank for attribute "
                                     << attname);
                    return -1;
                }
                if (dim_dset != 16)
                {
                    MGMOL_LBFGS_FAIL("error in dim for attribute " << attname);
                    return -1;
                }

                // read attribute
                status = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &attr_d[0]);
                if (status < 0)
                {
                    MGMOL_LBFGS_FAIL(error_string << "H5Aread failed!!!");
                    return -1;
                }

                // close ressources
                status = H5Aclose(attribute_id);
                if (status < 0)
                {
                    MGMOL_LBFGS_FAIL("H5Aclose failed!!!");
                    return -1;
                }
                status = H5Sclose(attdataspace);
                if (status < 0)
                {
                    MGMOL_LBFGS_FAIL("H5Sclose failed!!!");
                    return -1;
                }
            }
        }
    }
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.bcast(&check_data, 1);

    // return 1 if no data read
    if (!check_data) return 1;

    mmpi.bcast(&attr_d[0], 16);

    if (onpe0)
        (*MPIdata::sout) << "Setup LBFGS using restart info" << std::endl;

    fx_      = attr_d[0];
    fy_      = attr_d[1];
    stx_     = attr_d[2];
    sty_     = attr_d[3];
    dgx_     = attr_d[4];
    dgy_     = attr_d[5];
    stmin_   = attr_d[6];
    stmax_   = attr_d[7];
    width_   = attr_d[8];
    width1_  = attr_d[9];
    finit_   = attr_d[10];
    dginit_  = attr_d[11];
    etot_[0] = attr_d[12];
    etot_[1] = attr_d[13];
    etot_[2] = attr_d[14];
    stp_     = attr_d[15];

    std::vector<int> attr_i;
    attr_i.resize(nb_attri);
    if (onpe0)
    {
        // 2nd attribute: nb_attri int
        std::string attname = "int parameters";

        //  Open a dataset attribute.
        hid_t attribute_id = H5Aopen_name(dataset_id, attname.c_str());
        if (attribute_id < 0)
        {
            MGMOL_LBFGS_FAIL("H5Aopen failed for " << attname << "!!!");
            return -1;
        }

        hid_t attdataspace = H5Aget_space(attribute_id);
        int rank = H5Sget_simple_extent_dims(attdataspace, &dim_dset, &maxdim);
        if (rank != 1)
        {
            MGMOL_LBFGS_FAIL("error in rank for attribute " << attname);
            return -1;
        }
        if (dim_dset == (nb_attri - 1))
        { // for backward compatibility with older restart files
            attr_i[10] = true;
        }
        else
        {
            if (dim_dset != nb_attri)
            {
                MGMOL_LBFGS_FAIL("error in dim for attribute " << attname);
                return -1;
            }
        }

        // read attribute
        status = H5Aread(attribute_id, H5T_NATIVE_INT, &attr_i[0]);
        if (status < 0)
        {
            MGMOL_LBFGS_FAIL("H5Aread failed!!!");
            return -1;
        }

        // close ressources
        status = H5Aclose(attribute_id);
        if (status < 0)
        {
            MGMOL_LBFGS_FAIL("H5Aclose failed!!!");
            return -1;
        }
        status = H5Sclose(attdataspace);
        if (status < 0)
        {
            MGMOL_LBFGS_FAIL("H5Sclose failed!!!");
            return -1;
        }
    }
    mmpi.bcast(&attr_i[0], nb_attri);
    iflag_              = attr_i[0];
    m_                  = attr_i[1];
    iter_               = attr_i[2];
    point_              = attr_i[3];
    info_               = attr_i[4];
    infoc_              = attr_i[5];
    nfev_               = attr_i[6];
    npt_                = attr_i[7];
    brackt_             = attr_i[8];
    stage1_             = attr_i[9];
    last_step_accepted_ = attr_i[10];

    work_.resize(gndofs_ * 2 * (m_ + 1) + 2 * m_);
    int dim = work_.size() + 2 * gndofs_;
    std::vector<double> u_buffer(dim);
    double* u = u_buffer.data();
    if (onpe0)
    {
        hid_t dset_space = H5Dget_space(dataset_id);

        status = H5Dread(
            dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, u);
        if (status < 0)
        {
            MGMOL_LBFGS_FAIL("read_lbfgs(): H5Dread failed!!!");
            return -1;
        }

        status = H5Sclose(dset_space);
        if (status < 0)
        {
            MGMOL_LBFGS_FAIL("read_lbfgs(): H5Sclose failed!!!");
            return -1;
        }

        // Close the dataset.
        status = H5Dclose(dataset_id);
        if (status < 0)
        {
            MGMOL_LBFGS_FAIL("read_lbfgs(): H5Dclose failed!!!");
            return -1;
        }
    }

    mmpi.bcast(&u[0], dim);
    int wsize = work_.size();
    int ione  = 1;
    DCOPY(&wsize, u, &ione, &work_[0], &ione);
    DCOPY(&gndofs_, u + wsize, &ione, &diag_[0], &ione);
    // test for compatibility with previous version restart files
    if ((int)dim_dset >= dim)
    {
        memcpy(&xref_[0], u + work_.size() + diag_.size(),
            xref_.size() * sizeof(double));
    }
    else
    {
        if (onpe0)
            (*MPIdata::sout)
                << "LBFGS Warning: No reference ionic positions in restart file"
                << std::endl;
    }

    return 0;
}

int LBFGS_IonicStepper::init(HDFrestart& h5f_file)
{
    if (onpe0) (*MPIdata::sout) << "LBFGS_IonicStepper::init()...\n";
    hid_t file_id = h5f_file.file_id();

    if (file_id >= 0)
    {
        readPositions_hdf5(h5f_file, std::string("/Ionic_positions"));

        // Open dataset
        htri_t exists = H5Lexists(file_id, "/Ionic_velocities", H5P_DEFAULT);
        if (!exists)
        {
            if (onpe0)
            {
                (*MPIdata::sout) << "Warning: LBFGS_IonicStepper::init(), "
                                    "H5Dopen2 failed for /Ionic_velocities!!!"
                                 << std::endl;
                (*MPIdata::sout) << "Set velocities to zero" << std::endl;
            }
            memset(&taup_[0], 0, taup_.size() * sizeof(double));
        }
        else
        {
            hid_t dataset_id
                = H5Dopen2(file_id, "/Ionic_velocities", H5P_DEFAULT);
            herr_t status = 0;

            // Read velocities equal to (taup-tau0)/dt
            status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                H5P_DEFAULT, &taup_[0]);
            if (status < 0)
            {
                (*MPIdata::serr) << "Error: LBFGS_IonicStepper::init: H5Dread "
                                    "failed for taup!!!"
                                 << std::endl;
                return -1;
            }

            // close dataset
            status = H5Dclose(dataset_id);
            if (status < 0)
            {
                MGMOL_LBFGS_FAIL("init(): H5Dclose failed!!!");
                return -1;
            }
        }
    }

    int n = (int)tau0_.size(), ione = 1;
    DAXPY(&n, &dt_, &taup_[0], &ione, &tau0_[0], &ione);

    int rlbfgs = read_lbfgs(h5f_file);

    return rlbfgs;
}

void LBFGS_IonicStepper::restart()
{
    info_  = 0;
    iflag_ = 0;
    point_ = 0;
    iter_  = 0;
    npt_   = 0;
}

int LBFGS_IonicStepper::write_hdf5(HDFrestart& h5f_file)
{
    int status = 0;

    if (iflag_ < 0)
    {
        if (onpe0)
        {
            (*MPIdata::sout)
                << "LBFGS algorithm ended with failure" << std::endl;
            (*MPIdata::sout) << "Do NOT write LBFGS info!" << std::endl;
        }
        return 0;
    }

    hid_t file_id = h5f_file.file_id();
    if (file_id < 0) return 0;

#ifdef MGMOL_USE_HDF5P
    const bool write_flag = (onpe0 || h5f_file.useHdf5p());
#else
    const bool write_flag = onpe0;
#endif

    if (write_flag)
    {
        status = writeLBFGSinfo(h5f_file);
    }

    return status;
}

int LBFGS_IonicStepper::run()
{
    const int na = (int)atmove_.size();
    assert((int)fion_.size() == 3 * na);
    assert(gids_.size() == atmove_.size());

    double step = 2.74; // safe step for H2 molecule

    // set xcurrent_
    setDataFromLocalData(xcurrent_, tau0_);

    // reset data to 0
    memset(&g_[0], 0, gndofs_ * sizeof(double));

    for (int ia = 0; ia < na; ia++)
    {
        const int gid = gids_[ia];
        if (atmove_[ia])
        {
            for (short j = 0; j < 3; j++)
            {
                assert(gid < gndofs_ / 3);

                g_[3 * gid + j] = -fion_[3 * ia + j];
            }
        }
    }

    bool diagco;
    if (iflag_ == 1)
    {
        diagco = false;
    }
    else
    {
        diagco = true;
        for (int k = 0; k < gndofs_; k++)
            diag_[k] = step;
    }

    // consolidate data across all pes
    MGmol_MPI& mmpi = *(MGmol_MPI::instance());
    mmpi.allreduce(&g_[0], gndofs_, MPI_SUM);

    if (onpe0) iflag_ = lbfgs(etot_[2], diagco, iflag_);

    // sync data across all pes
    mmpi.bcast(&xcurrent_[0], gndofs_);
    mmpi.bcast(&iflag_, 1);
    mmpi.bcast(&last_step_accepted_, 1);

    if (iflag_ == 0) return 1;
    if (iflag_ == -1)
    {
        if (onpe0)
        {
            MGMOL_LBFGS_FAIL("Line search failed, possibly due to inaccurate "
                             "energies/forces");
        }
        return -1;
    }
    if (iflag_ == -2)
    {
        if (onpe0)
            MGMOL_LBFGS_FAIL("inverse Hessian approximation is not positive");
        return -2;
    }
    if (iflag_ == -3)
    {
        if (onpe0) MGMOL_LBFGS_FAIL("yy is too small");
        return -3;
    }

    // update local taup_
    for (int ia = 0; ia < na; ia++)
    {
        if (atmove_[ia])
        {
            const int gid = gids_[ia];
            for (int j = 0; j < 3; j++)
            {
                taup_[3 * ia + j] = xcurrent_[3 * gid + j];
            }
        }
        else
        {
            for (int j = 0; j < 3; j++)
                taup_[3 * ia + j] = tau0_[3 * ia + j];
        }
    }

    return 0;
}

double LBFGS_IonicStepper::etol(void) const
{
    double emax    = std::max(etot_[2], etot_[1]);
    emax           = std::max(emax, etot_[0]);
    double emin    = std::min(etot_[2], etot_[1]);
    emin           = std::min(emin, etot_[0]);
    double espread = emax - emin;

    return std::min(espread * 1.e-4, 1.e-8);
}

// lbfgs() returns an integer between -3 and +2:
//    iflag   is an integer variable that must be set to 0 on initial entry
//            to the subroutine.
//    diagco  is a boolean variable that must be set to true if the
//            user  wishes to provide the diagonal matrix Hk0 at each
//            iteration. Otherwise it should be set to false, in which
//            case  LBFGS will use a default value. If
//            diagco is set to true the routine will return 2 at each
//            iteration of the algorithm, and the diagonal
//             matrix Hk0  must be provided in the array diag_.
// A negative return value indicates an error,
// and 0 indicates that the routine has terminated without
// detecting errors. On a return value 1, the user must
// evaluate the function F and gradient G. On a return value
// 2, the user must provide the diagonal matrix Hk0.
//
// The following negative values, detecting an error,
// are possible:
//
//  -1  The line search routine mcsrch failed. The
//      parameter info_ provides more detailed information
//      (see the documentation of mcsrch)
//  -2  The i-th diagonal element of the diagonal inverse
//      Hessian approximation, given in diag_, is not
//      positive.
//  -3  Numerical problem with yy (division by 0).
int LBFGS_IonicStepper::lbfgs(
    const double f, const bool diagco, const int iflag)
{
    const int ispt = gndofs_ + 2 * m_;
    const int iypt = ispt + gndofs_ * m_;

    int inc     = 1;
    bool finish = true;

    if (iflag == 0)
    {
        iter_  = 0;
        point_ = 0;
        finish = false;
        if (diagco)
        {
            for (int i = 0; i < gndofs_; i++)
                if (diag_[i] <= 0.)
                {
                    return -2;
                }
        }
        else
        {
            for (int i = 0; i < gndofs_; i++)
                diag_[i] = 1.;
        }
        // the work std::vector work_ is divided as follows:
        // ---------------------------------------
        // the first n locations are used to store the gradient and
        //     other temporary information.
        // locations n...n+m-1 store the scalars rho.
        // locations n+m...n+2m-1 store the numbers alpha used
        //     in the formula that computes h*g.
        // locations n+2m...n+2m+nm-1 store the last m search
        //     steps.
        // locations n+2m+nm...n+2m+2nm-1 store the last m
        //     gradient differences.
        //
        // the search steps and gradient differences are stored in a
        // circular order controlled by the parameter point.
        //
        for (int i = 0; i < gndofs_; i++)
            work_[ispt + i] = -g_[i] * diag_[i];
    }

    // Main iteration loop
    do
    {
        if ((iflag == 0) || (!finish))
        {
            info_ = 0;
            iter_++;
            int bound = iter_ - 1;
            if (iter_ != 1)
            {
                if (iter_ > m_) bound = m_;
                double ys = DDOT(&gndofs_, &work_[iypt + npt_], &inc,
                    &work_[ispt + npt_], &inc);
                if (!diagco)
                {
                    double yy = DDOT(&gndofs_, &work_[iypt + npt_], &inc,
                        &work_[iypt + npt_], &inc);
                    if (yy < 1.e-14)
                    {
                        if (onpe0)
                            (*MPIdata::sout) << "LBFGS: yy=" << yy << std::endl;
                        return -3;
                    }
                    double ratio = ys / yy;
                    for (int i = 0; i < gndofs_; i++)
                        diag_[i] = ratio;
                }
                else
                {
                    return 2;
                }

                // Compute -H*g using the formula given in: Nocedal, J. 1980,
                // "Updating quasi-Newton matrices with limited storage",
                // Mathematics of Computation, Vol.24, No.151, pp. 773-782.
                minushg(bound, ys);

                // Fix geometry center by setting average displacement to 0.
                if (freeze_geom_center_)
                {
                    double ax = 0.;
                    double ay = 0.;
                    double az = 0.;
                    for (int i = 0; i < gndofs_; i += 3)
                    {
                        ax += work_[i];
                        ay += work_[i + 1];
                        az += work_[i + 2];
                    }
                    ax *= (3. / (double)gndofs_);
                    ay *= (3. / (double)gndofs_);
                    az *= (3. / (double)gndofs_);
                    for (int i = 0; i < gndofs_; i += 3)
                    {
                        work_[i] -= ax;
                        work_[i + 1] -= ay;
                        work_[i + 2] -= az;
                    }
                    //(*MPIdata::sout)<<"lbfgs:shift search std::vector by
                    //"<<ax<<",
                    //"<<ay<<", "<<az<<endl;
                }

                // Restrict the new search direction to small displacements (0.1
                // a.u.)
                const double dismax = 0.1;
                int maxw            = (IDAMAX(&gndofs_, &work_[0], &inc) - 1);
                if (fabs(work_[maxw]) > dismax)
                {
                    double alpha = dismax / fabs(work_[maxw]);
                    (*MPIdata::sout) << "LBFGS: rescale search std::vector by "
                                     << alpha << std::endl;
                    DSCAL(&gndofs_, &alpha, &work_[0], &inc);
                }

                // Store the new search direction
                memcpy(&work_[ispt + point_ * gndofs_], &work_[0],
                    gndofs_ * sizeof(double));
            }

            nfev_ = 0;
            stp_  = 1.;
            DCOPY(&gndofs_, &g_[0], &inc, &work_[0], &inc);
        }

        // obtain the one-dimensional minimizer of the function
        // by using the line search routine mcsrch
        mcsrch(f, &work_[ispt + point_ * gndofs_], stp_);
        if (info_ == -1)
        {
            return 1;
        }
        if (info_ != 1)
        {
            return -1;
        }

        // Compute the new step and gradient change
        npt_ = point_ * gndofs_;
        DSCAL(&gndofs_, &stp_, &work_[ispt + npt_], &inc);
        for (int i = 0; i < gndofs_; i++)
        {
            work_[iypt + npt_ + i] = g_[i] - work_[i];
        }
        point_++;
        if (point_ == m_) point_ = 0;

        // Termination test
        double gnorm = sqrt(DDOT(&gndofs_, &g_[0], &inc, &g_[0], &inc));
        (*MPIdata::sout) << std::setprecision(2) << std::scientific;
        if (onpe0)
            (*MPIdata::sout)
                << "LBFGS: Norm of gradient=" << gnorm << std::endl;
        finish = (gnorm <= eps_);

    } while (!finish);

    (*MPIdata::sout) << "The minimization terminated without detecting errors."
                     << std::endl;

    return 0;
}

// Compute -H*g using the formula given in: Nocedal, J. 1980,
// "Updating quasi-Newton matrices with limited storage",
// Mathematics of Computation, Vol.24, No.151, pp. 773-782.
void LBFGS_IonicStepper::minushg(const int bound, const double ys)
{
    assert(fabs(ys) > 0.);

    int inc        = 1;
    const int ispt = gndofs_ + 2 * m_;
    const int iypt = ispt + gndofs_ * m_;
    int cp         = point_;
    if (point_ == 0) cp = m_;
    work_[gndofs_ + cp - 1] = 1. / ys;
    for (int i = 0; i < gndofs_; i++)
        work_[i] = -g_[i];
    cp = point_;
    for (int i = 0; i < bound; i++)
    {
        cp--;
        if (cp == -1) cp = m_ - 1;
        double sq = DDOT(
            &gndofs_, &work_[ispt + cp * gndofs_], &inc, &work_[0], &inc);
        int inmc     = gndofs_ + m_ + cp;
        int iycn     = iypt + cp * gndofs_;
        work_[inmc]  = work_[gndofs_ + cp] * sq;
        double alpha = -work_[inmc];
        DAXPY(&gndofs_, &alpha, &work_[iycn], &inc, &work_[0], &inc);
    }

    for (int i = 0; i < gndofs_; i++)
        work_[i] *= diag_[i];

    for (int i = 0; i < bound; i++)
    {
        double yr = DDOT(
            &gndofs_, &work_[iypt + cp * gndofs_], &inc, &work_[0], &inc);
        double beta = work_[gndofs_ + cp] * yr;
        int inmc    = gndofs_ + m_ + cp;
        beta        = work_[inmc] - beta;
        int iscn    = ispt + cp * gndofs_;
        DAXPY(&gndofs_, &beta, &work_[iscn], &inc, &work_[0], &inc);
        cp++;
        if (cp == m_) cp = 0;
    }
}

void LBFGS_IonicStepper::mcsrch(const double f, double* s, double& stp)
{
    //   Parameters for line search routine
    const int maxfev  = 5;
    const double gtol = 0.9;
    const double ftol = 1.e-4;
    // nonnegative input variables which specify lower and upper bounds
    // for the step
    const double stpmin = 1.e-5;
    const double stpmax = 1.e+6;

    // Adapted from the fortran77 subroutine csrch of More' and Thuente.
    //
    // The purpose of mcsrch is to find a step which satisfies
    // a sufficient decrease condition and a curvature condition.
    //
    // At each stage the subroutine updates an interval of
    // uncertainty with endpoints stx and sty. the interval of
    // uncertainty is initially chosen so that it contains a
    // minimizer of the modified function
    //
    //  f(x+stp*s) - f(x) - ftol*stp*(gradf(x)'s).
    //
    // if a step is obtained for which the modified function
    // has a nonpositive function value and nonnegative derivative,
    // then the interval of uncertainty is chosen so that it
    // contains a minimizer of f(x+stp*s).
    //
    // the algorithm is designed to find a step which satisfies
    // the sufficient decrease condition
    //
    //       f(x+stp*s) <= f(x) + ftol*stp*(gradf(x)'s),
    //
    // and the curvature condition
    //
    //  abs(gradf(x+stp*s)'s)) <= gtol*abs(gradf(x)'s).
    //
    // if ftol is less than gtol and if, for example, the function
    // is bounded below, then there is always a step which satisfies
    // both conditions. if no step can be found which satisfies both
    // conditions, then the algorithm usually stops when rounding
    // errors prevent further progress. in this case stp only
    // satisfies the sufficient decrease condition.
    //
    //   f is a variable. on input it must contain the value of f at x.
    //
    //   s specifies the search direction.
    //
    //   stp is a nonnegative variable.
    //   on input stp contains an initial estimate of a satisfactory step.
    //   on output stp contains the final estimate.
    //
    int inc = 1;

    if (info_ != -1)
    {
        infoc_ = 1;

        //  compute the initial gradient in the search direction
        //  and check that s is a descent direction.
        dginit_ = DDOT(&gndofs_, &g_[0], &inc, s, &inc);
        if (dginit_ >= 0.)
        {
            (*MPIdata::sout)
                << "LBFGS: the search direction is not a descent direction"
                << std::endl;
            return;
        }

        // initialize local variables.
        brackt_ = false;
        stage1_ = true;
        nfev_   = 0;
        finit_  = f;
        width_  = stpmax - stpmin;
        width1_ = 2. * width_;
        (*MPIdata::sout) << "LBFGS: update reference positions" << std::endl;
        DCOPY(&gndofs_, &xcurrent_[0], &inc, &xref_[0], &inc);
        last_step_accepted_ = 1;

        // the variables stx, fx, dgx contain the values of the step,
        // function, and directional derivative at the best step.
        // the variables sty, fy, dgy contain the value of the step,
        // function, and derivative at the other endpoint of
        // the interval of uncertainty.
        // the variables stp, f, dg contain the values of the step,
        // function, and derivative at the current step.
        stx_ = 0.;
        fx_  = finit_;
        dgx_ = dginit_;
        sty_ = 0.;
        fy_  = finit_;
        dgy_ = dginit_;
    }
    else
    {
        last_step_accepted_ = 0;
    }

    for (int k = 0; k < 10; k++)
    {
        if (info_ != -1)
        {
            // set the minimum and maximum steps to correspond
            // to the present interval of uncertainty.
            if (brackt_)
            {
                stmin_ = std::min(stx_, sty_);
                stmax_ = std::max(stx_, sty_);
            }
            else
            {
                stmin_ = stx_;
                stmax_ = stp + 4. * (stp - stx_);
            }

            // force the step to be within the bounds stpmax and stpmin.
            stp = std::max(stp, stpmin);
            stp = std::min(stp, stpmax);

            // if an unusual termination is to occur then let
            // stp be the lowest point obtained so far.

            if ((brackt_ && (stp <= stmin_ || stp >= stmax_))
                || nfev_ >= maxfev - 1 || infoc_ == 0
                || (brackt_ && stmax_ - stmin_ <= xtol_ * stmax_))
                stp = stx_;

            // evaluate the function and gradient at stp
            // and compute the directional derivative.
            // we return to main program to obtain f and g.
            (*MPIdata::sout) << std::setprecision(2) << std::scientific;
            (*MPIdata::sout)
                << "LBFGS: try new position with stp=" << stp << std::endl;
            for (int j = 0; j < gndofs_; j++)
            {
                assert(s[j] < 10.);
                xcurrent_[j] = xref_[j] + stp * s[j];
            }
            if (stp < stpmin) restart();
            info_ = -1;
            return;
        }
        info_ = 0;
        nfev_++;
        double dg     = DDOT(&gndofs_, &g_[0], &inc, s, &inc);
        double dgtest = ftol * dginit_;
        double ftest1 = finit_ + stp * dgtest;

        //(*MPIdata::sout)<<"ftest1="<<ftest1<<", f="<<f<<",
        // dginit_="<<dginit_<<", dg="<<dg
        //<<", stp="<<stp<<endl;

        // test for convergence.
        //
        //     info =-1  a return is made to compute the function and gradient.
        //     info = 1  the sufficient decrease condition and the
        //       directional derivative condition hold.
        //     info = 2  relative width of the interval of uncertainty
        //       is at most xtol.
        //     info = 3  number of calls to fcn has reached maxfev.
        //     info = 4  the step is at the lower bound stpmin.
        //     info = 5  the step is at the upper bound stpmax.
        //     info = 6  rounding errors prevent further progress.
        //       there may not be a step which satisfies the
        //       sufficient decrease and curvature conditions.
        //       tolerances may be too small.
        //
        if ((brackt_ && (stp <= stmin_ || stp >= stmax_)) || infoc_ == 0)
            info_ = 6;
        if (stp == stpmax && f <= ftest1 && dg <= dgtest) info_ = 5;
        if (stp == stpmin && (f > ftest1 || dg >= dgtest)) info_ = 4;
        if (nfev_ >= maxfev)
        {
            info_ = 3;
            if (f <= ftest1)
            {
                (*MPIdata::sout) << "Warning: 2nd Wolfe condition not satisfied"
                                 << std::endl;
                info_ = 1;
            }
        }
        if (brackt_ && stmax_ - stmin_ <= xtol_ * stmax_) info_ = 2;
        if (f <= ftest1 && fabs(dg) <= gtol * (-dginit_)) info_ = 1;

        // check for termination.
        if (info_ != 0) return;

        // in the first stage we seek a step for which the modified
        // function has a nonpositive value and nonnegative derivative.
        if (stage1_ && f <= ftest1 && dg >= std::min(ftol, gtol) * dginit_)
            stage1_ = false;

        // a modified function is used to predict the step only if
        // we have not obtained a step for which the modified
        // function has a nonpositive function value and nonnegative
        // derivative, and if a lower function value has been
        // obtained but the decrease is not sufficient.

        if (stage1_ && f <= fx_ && f > ftest1)
        {
            // define the modified function and derivative values.
            double fm   = f - stp * dgtest;
            double fxm  = fx_ - stx_ * dgtest;
            double fym  = fy_ - sty_ * dgtest;
            double dgm  = dg - dgtest;
            double dgxm = dgx_ - dgtest;
            double dgym = dgy_ - dgtest;

            // call mcstep to update the interval of uncertainty
            // and to compute the new step.
            infoc_ = mcstep(stx_, fxm, dgxm, sty_, fym, dgym, stp, fm, dgm,
                brackt_, stmin_, stmax_);

            // reset the function and gradient values for f.
            fx_  = fxm + stx_ * dgtest;
            fy_  = fym + sty_ * dgtest;
            dgx_ = dgxm + dgtest;
            dgy_ = dgym + dgtest;
        }
        else
        {
            // call mcstep to update the interval of uncertainty
            // and to compute the new step.
            infoc_ = mcstep(stx_, fx_, dgx_, sty_, fy_, dgy_, stp, f, dg,
                brackt_, stmin_, stmax_);
        }

        // force a sufficient decrease in the size of the
        // interval of uncertainty.
        if (brackt_)
        {
            if (fabs(sty_ - stx_) >= 0.66 * width1_)
                stp = stx_ + 0.5 * (sty_ - stx_);
            width1_ = width_;
            width_  = fabs(sty_ - stx_);
        }
    }
}
