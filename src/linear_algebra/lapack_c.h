// Copyright (c) 2017, Lawrence Livermore National Security, LLC and
// UT-Battelle, LLC.
// Produced at the Lawrence Livermore National Laboratory and the Oak Ridge
// National Laboratory.
// LLNL-CODE-743438
// All rights reserved.
// This file is part of MGmol. For details, see https://github.com/llnl/mgmol.
// Please also read this link https://github.com/llnl/mgmol/LICENSE

#ifndef MGMOL_LAPACK_H
#define MGMOL_LAPACK_H

#include "fc_mangle.h"

typedef const char* const Pchar;
typedef const int* const Pint;
typedef const double* const Pdouble;

#define dsygst DSYGST
#define dtrtrs DTRTRS
#define dpocon DPOCON
#define dsygv DSYGV
#define dlange DLANGE

extern "C"
{
    void DSYEV(Pchar, Pchar, const int* const, double*, const int* const,
        double*, double*, const int* const, int*);
    void DSYGV(const int* const, Pchar, Pchar, const int* const, double*,
        const int* const, double*, const int* const, double*, double*,
        const int* const, int*);
    void DPOTRI(Pchar, const int* const, double*, const int* const, int*);
    void DPOTRF(Pchar, const int* const, double*, const int* const, int*);
    void DPOTRS(Pchar, const int* const, const int* const, double*,
        const int* const, double*, const int* const, int*);
    void DGETRF(int*, int*, double*, int*, int*, int*);
    void DGETRS(Pchar, int*, int*, double*, int*, int*, double*, int*, int*);
    void DGESVD(Pchar, Pchar, int*, int*, double*, int*, double*, double*, int*,
        double*, int*, double*, int*, int*);
    void dpocon(Pchar, const int* const, double*, const int* const, double*,
        double*, double*, const int* const, int*);
    void dtrtrs(Pchar, Pchar, Pchar, const int* const, const int* const,
        double*, const int* const, double*, const int* const, int*);
    void dtrtri(Pchar, Pchar, int*, double*, int*, int*);
    void dsygst(const int* const, Pchar, const int* const, double*,
        const int* const, double*, const int* const, int*);
    void dgesvd(Pchar, Pchar, int*, int*, double*, int*, double*, double*, int*,
        double*, int*, double*, int*, int*);
    double dlange(Pchar, int*, int*, double*, int*, double*);
    void DLACPY(Pchar, Pint, Pint, Pdouble, Pint, Pdouble, Pint);
}

#endif
