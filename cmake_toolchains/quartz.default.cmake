set(CMAKE_C_COMPILER mpicc)
set(CMAKE_CXX_COMPILER mpicxx)
set(CMAKE_Fortran_COMPILER mpif90)

set(SCALAPACK_ROOT $ENV{MKLROOT})
set(SCALAPACK_BLACS_LIBRARY $ENV{MKLROOT}/lib/intel64/libmkl_blacs_intelmpi_lp64.so)
