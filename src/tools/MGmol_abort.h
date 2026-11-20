// Single place to end a run after an unrecoverable error.
//
// Unlike exit(), MPI_Abort() stops every rank, so no process is left
// waiting in a collective call, and the process exit code is nonzero
// so schedulers and scripts can see that the run failed.
#ifndef MGMOL_ABORT_H
#define MGMOL_ABORT_H

#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <string>

[[noreturn]] inline void mgmol_abort(
    const std::string& msg = "", const int code = EXIT_FAILURE)
{
    if (!msg.empty()) std::cerr << "MGmol: fatal error: " << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, code);
    std::abort(); // not reached; keeps [[noreturn]] honest if MPI_Abort returns
}

#endif
