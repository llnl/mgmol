// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_PROTOTYPES_H
#define MGMOL_PROTOTYPES_H

#include "global.h"
#include "mgmol_config.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

class Ions;
class KBPsiMatrixSparse;

void get_vnlpsi(const Ions& ions, const std::vector<std::vector<int>>&,
    const int, const KBPsiMatrixSparse* const kbpsi, ORBDTYPE* const);
double getLAeigen(const double tol, const int maxit, Ions& ions);
int read_config(int argc, char** argv,
    boost::program_options::variables_map& vm, std::string& input_file,
    std::string& lrs_filename, std::string& constraints_filename,
    float& total_spin, bool& with_spin);
#ifdef MGMOL_HAS_LIBROM
void setupROMConfigOption(po::options_description& rom_cfg);
#endif

#endif
