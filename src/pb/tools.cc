// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

// Some general tools
// $Id: tools.cc,v 1.6 2008/12/10 01:05:37 jeanluc Exp $
#include <ctime>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>

#include "MGmol_abort.h"
#include "tools.h"

namespace pb
{

FILE* open_file(const std::string& filename, const std::string& mode)
{
    FILE* file_ptr;

    if ((file_ptr = fopen(filename.data(), mode.data())) == nullptr)
    {
        std::cerr << "\n cannot open " << filename << std::endl;
        mgmol_abort("cannot open file");
    }
    else
    {
        std::cout << " open " << filename << std::endl;
    }

    return file_ptr;
}

double timer(void)
{
    struct timeval tt;

    gettimeofday(&tt, nullptr);
    double val1 = (double)tt.tv_usec;
    val1 /= 1000000.;
    double val = (double)tt.tv_sec;
    val += val1;
    return val;
}

} // namespace pb
