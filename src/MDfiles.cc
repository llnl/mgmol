// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#include "MDfiles.h"
#include "Control.h"
#include "Mesh.h"
#include "Vector3D.h"
#include "mgmol_mpi_tools.h"

#include <iostream>
#include <sstream>
#include <sys/stat.h>
using namespace std;

#define GROUP_SIZE 128

Timer MDfiles::print_data_tm_("MDfiles::print_data");

MDfiles::MDfiles()
{
    Mesh* mymesh             = Mesh::instance();
    const pb::PEenv& myPEenv = mymesh->peenv();

    color_ = myPEenv.mytask() / GROUP_SIZE;
    key_   = myPEenv.mytask() % GROUP_SIZE;
    // cout<<"color="<<color_<<", key="<<key_<<endl;

    MPI_Comm_split(myPEenv.comm(), color_, key_, &sub_comm_);

    MPI_Comm_size(sub_comm_, &size_comm_);
}

void MDfiles::appendTaskNumberToName(string& name)
{
    int mytask = color_;
    if (mytask < 1000000)
    {
        name.append("0");
    }
    if (mytask < 100000)
    {
        name.append("0");
    }
    if (mytask < 10000)
    {
        name.append("0");
    }
    if (mytask < 1000)
    {
        name.append("0");
    }
    if (mytask < 100)
    {
        name.append("0");
    }
    if (mytask < 10)
    {
        name.append("0");
    }

    stringstream oss("");
    oss << mytask;

    name.append(oss.str());

    return;
}

void MDfiles::createSnapshotDir(const int mdstep, string& md_print_dir)
{
    Control& ct = *(Control::instance());

    stringstream extension("");
    extension << mdstep;

    // char   extension[10];
    // sprintf(extension, "%d", mdstep);

    md_print_dir = ct.md_print_filename;
    if (mdstep < 10000)
    {
        md_print_dir.append("0");
    }
    if (mdstep < 1000)
    {
        md_print_dir.append("0");
    }
    if (mdstep < 100)
    {
        md_print_dir.append("0");
    }
    if (mdstep < 10)
    {
        md_print_dir.append("0");
    }
    md_print_dir.append(extension.str());

    if (onpe0)
    {
        cout << "Print coordinates and forces in directory " << md_print_dir
             << endl;
        struct stat buf;
        int exists = stat(md_print_dir.c_str(), &buf);
        if (exists < 0)
        {
            mode_t mode = (S_IRWXU | S_IRWXG | S_IRWXO);
            mkdir(md_print_dir.c_str(), mode);
            (*MPIdata::sout) << "Create dir " << md_print_dir << endl;
        }
    }
}

void MDfiles::gatherStrings(vector<string>& ions_names, vector<char>& recvbufs)
{
    int size_s = 0;
    for (vector<string>::const_iterator it = ions_names.begin();
         it != ions_names.end(); ++it)
    {
        size_s += ((int)it->size() + 1); // +1 is for delimiter
    }
    vector<int> recvcounts(size_comm_);
    int mpi_err = MPI_Gather(
        &size_s, 1, MPI_INT, &recvcounts[0], 1, MPI_INT, 0, sub_comm_);
    if (mpi_err != MPI_SUCCESS)
    {
        (*MPIdata::serr) << "MDfiles::gatherStrings(), ERROR in MPI_gather!!!"
                         << endl;
    }
    vector<int> displs(size_comm_);
    if (key_ == 0)
    {
        displs[0] = 0;
        for (int i = 1; i < size_comm_; i++)
        {
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    // pack data
    vector<char> buffer(size_s);
    char* pbuf = &buffer[0];
    for (vector<string>::const_iterator it = ions_names.begin();
         it != ions_names.end(); ++it)
    {
        it->copy(pbuf, string::npos);
        pbuf += it->length();
        *pbuf = ' '; // add delimiter
        pbuf++;
    }

    // gather data
    if (key_ == 0)
        recvbufs.resize(displs[size_comm_ - 1] + recvcounts[size_comm_ - 1]);
    mpi_err = MPI_Gatherv(&buffer[0], size_s, MPI_CHAR, &recvbufs[0],
        &recvcounts[0], &displs[0], MPI_CHAR, 0, sub_comm_);

    if (mpi_err != MPI_SUCCESS)
    {
        (*MPIdata::serr) << "MDfiles::gatherStrings(), ERROR in MPI_Gatherv !!!"
                         << endl;
    }
}

void MDfiles::gatherVector3D(vector<Vector3D>& vectors, vector<double>& recvbuf)
{
    vector<double> tau;
    for (std::vector<Vector3D>::iterator iv = vectors.begin();
         iv != vectors.end(); ++iv)
    {
        tau.push_back((*iv)[0]);
        tau.push_back((*iv)[1]);
        tau.push_back((*iv)[2]);
    }

    mgmol_tools::gatherV(tau, recvbuf, 0, sub_comm_);
}

void MDfiles::printDataInFiles(vector<string>& ions_names, vector<double>& tau,
    vector<double>& forces, vector<double>& taum, vector<Vector3D>& centers,
    vector<float>& spreads, vector<int>& gids, const int mdstep,
    const double dt)
{
    print_data_tm_.start();

    // int tausize=tau.size();
    // int sumtausize;
    // MPI_Reduce(&tausize,&sumtausize,1,MPI_INT,MPI_SUM,0,sub_comm_);
    // if(key_==0)cout<<"key="<<key_<<", color="<<color_<<", sum of tau sizes:
    // "<<sumtausize<<endl;

    string dir_name = "";
    createSnapshotDir(mdstep, dir_name);

    // gather names
    vector<char> recvbufnames;
    gatherStrings(ions_names, recvbufnames);

    // gather tau
    vector<double> recvbuf;
    mgmol_tools::gatherV(tau, recvbuf, 0, sub_comm_);
    vector<double> recvbufm;
    mgmol_tools::gatherV(taum, recvbufm, 0, sub_comm_);
    vector<double> recvbuff;
    mgmol_tools::gatherV(forces, recvbuff, 0, sub_comm_);

    // gather spreads
    vector<float> recvbufspreads;
    mgmol_tools::gatherV(spreads, recvbufspreads, 0, sub_comm_);

    // gather centers
    vector<double> recvbufc;
    gatherVector3D(centers, recvbufc);

    // gather gids
    vector<int> recvbufi;
    mgmol_tools::gatherV(gids, recvbufi, 0, sub_comm_);

    // barrier seems to be needed here before writing files
    //(otherwise directory may not exists yet...)
    MPI_Barrier(MPI_COMM_WORLD);

    if (key_ == 0)
    {
        // atomic coordinates
        {
            string coord_dir_name(dir_name + "/coords");
            appendTaskNumberToName(coord_dir_name);
            // cout<<"filename="<<coord_dir_name<<endl;
            ofstream tfile(coord_dir_name.data(), ios::out);
            if (!tfile.is_open())
            {
                std::cerr << " Unable to open file " << coord_dir_name.data()
                          << endl;
                mgmol_abort();
            }
            // cout<<"recvbuf.size()="<<recvbuf.size()<<endl;
            // tfile<<"recvbuf.size()="<<recvbuf.size()<<endl;
            const int na = recvbuf.size() / 3;
            int j        = 0;
            for (int i = 0; i < na; i++)
            {
                // tfile<<"atom ";
                while (recvbufnames[j] != ' ')
                {
                    tfile << recvbufnames[j];
                    j++;
                }
                j++;
                tfile << setiosflags(ios::right) << setw(10) << setprecision(4)
                      << fixed << recvbuf[3 * i + 0] << setw(10)
                      << recvbuf[3 * i + 1] << setw(10) << recvbuf[3 * i + 2]
                      << setw(12) << setprecision(3) << scientific
                      << recvbuff[3 * i + 0] << setw(12) << recvbuff[3 * i + 1]
                      << setw(12) << recvbuff[3 * i + 2] << setw(12)
                      << setprecision(3) << scientific
                      << (recvbuf[3 * i + 0] - recvbufm[3 * i + 0]) / dt
                      << setw(12)
                      << (recvbuf[3 * i + 1] - recvbufm[3 * i + 1]) / dt
                      << setw(12)
                      << (recvbuf[3 * i + 2] - recvbufm[3 * i + 2]) / dt
                      << endl;
                // make sure buffers are written out
                tfile.flush();
            }
        }

        // now write localization centers and spreads
        if (recvbufspreads.size() > 0)
        {
            string wf_dir_name(dir_name + "/functions");
            appendTaskNumberToName(wf_dir_name);
            ofstream tfile2(wf_dir_name.data(), ios::out);
            if (!tfile2.is_open())
            {
                std::cerr << " Unable to open file " << wf_dir_name.data()
                          << std::endl;
                mgmol_abort();
            }
            const int na = recvbufspreads.size();
            for (int i = 0; i < na; i++)
            {
                tfile2 << setiosflags(ios::right) << setw(10) << setprecision(3)
                       << fixed << recvbufi[i] << setw(10)
                       << recvbufc[3 * i + 0] << setw(10) << recvbufc[3 * i + 1]
                       << setw(10) << recvbufc[3 * i + 2] << setw(10)
                       << recvbufspreads[i] << endl;
                // make sure buffers are written out
                tfile2.flush();
            }
        }
    }

    print_data_tm_.stop();
}
