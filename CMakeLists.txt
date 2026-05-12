cmake_minimum_required (VERSION 3.9)

cmake_policy(SET CMP0074 NEW)

project (MGmol C CXX Fortran)

set (CMAKE_CXX_STANDARD 11)

# Specify the location of additional CMAKE modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Version number
set (PROJECT_VERSION_MAJOR 1)
set (PROJECT_VERSION_MINOR 0)
set(MGMOL_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")

# Prohibit in-source build
IF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  MESSAGE(FATAL_ERROR "In-source build prohibited.")
ENDIF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")

# Use RPATH in install, many mpicc scripts use RPATH so default
# behavior of CMAKE to remove RPATH from installed executables is not
# so good.
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Cmake build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "No user specified build type, using default...")
endif()
message(STATUS "Build type is ${CMAKE_BUILD_TYPE}")

# Additional linker flags
set(MGMOL_LINK_FLAGS "" CACHE STRING "Additional linker flags")
list(APPEND LINK_LIBRARIES ${MGMOL_LINK_FLAGS})

# Use MPI (required)
find_package(MPI REQUIRED)
message(STATUS "MPIEXEC :" ${MPIEXEC})

# Use openMP
set(MGMOL_WITH_OPENMP_OFFLOAD FALSE CACHE BOOL "Compile with OpenMP offload")
if(MGMOL_WITH_OPENMP_OFFLOAD)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_VERSION LESS "4.5") 
    message(SEND_ERROR "OpenMP 4.5 or later required")
  endif()
  message(STATUS "Build with OpenMP offload")
  add_definitions(-DHAVE_OPENMP_OFFLOAD)
else()
  find_package(OpenMP)
endif()

# Use mixed precision
set(MGMOL_WITH_MP FALSE CACHE BOOL "Compile with mixed precision")
if(MGMOL_WITH_MP)
  add_definitions(-DMGMOL_USE_MIXEDP)
endif()

# HDF5 (required)
set(MGMOL_USE_HDF5P TRUE CACHE BOOL "Use HDF5 parallel capability")
if(HDF5_LIBRARIES) #user sets libraries explicitly
  message(STATUS "HDF5_LIBRARIES: ${HDF5_LIBRARIES}")
  message(STATUS "HDF5_LIBRARIES include: ${HDF5_INCLUDE_DIRS}")
  message(STATUS "HDF5_HL_LIBRARIES: ${HDF5_HL_LIBRARIES}")
else() #search for HDF5
  if(MGMOL_USE_HDF5P)
    message(STATUS "Use HDF5 parallel capability")
    set(HDF5_PREFER_PARALLEL True)
  endif()
  message(STATUS "HDF5_ROOT: ${HDF5_ROOT}")
  set(HDF5_SHARED_LIBS TRUE CACHE BOOL "Use shared hdf5 lib")
  if(NOT HDF5_SHARED_LIBS)
    set(HDF5_USE_STATIC_LIBRARIES ON)
    message(STATUS "HDF5_USE_STATIC_LIBRARIES = " ${HDF5_USE_STATIC_LIBRARIES})
  endif()
  find_package(HDF5 REQUIRED COMPONENTS C HL)
  if (${HDF5_FOUND})
    message(STATUS "HDF5_LIBRARIES after search: ${HDF5_LIBRARIES}")
    message(STATUS "HDF5_HL_LIBRARIES after search: ${HDF5_HL_LIBRARIES}")
  else(${HDF5_FOUND})
    message(FATAL_ERROR "Required HDF5 package not found.")
  endif (${HDF5_FOUND})
endif()
if(MGMOL_USE_HDF5P)
  add_definitions(-DMGMOL_USE_HDF5P)
endif()


set(MGMOL_WITH_LIBXC FALSE CACHE BOOL "Compile with LIBXC")
if(${MGMOL_WITH_LIBXC})
  add_definitions(-DUSE_LIBXC)
endif (${MGMOL_WITH_LIBXC})

# Boost (required)
find_package(Boost COMPONENTS program_options random REQUIRED)
if(NOT ${Boost_FOUND})
  message(FATAL_ERROR "Boost library required")
endif(NOT ${Boost_FOUND})
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

# blas/ lapack (required)
# Use cmake option -DBLA_VENDOR=<vendor>
# to set/ find vendor specific libraries.
# (see findBLAS documentation for more info.)
if(NOT BLAS_LIBRARIES)
  find_package(BLAS REQUIRED)
  message(STATUS "BLAS_LIBARIES: ${BLAS_LIBRARIES}")
  if(NOT ${BLAS_FOUND})
    message(FATAL_ERROR "Required blas library not found.")
  endif(NOT ${BLAS_FOUND})  
else(NOT BLAS_LIBRARIES)
  message(STATUS "BLAS_LIBARIES set to ${BLAS_LIBRARIES}")
endif(NOT BLAS_LIBRARIES)  

set(MGMOL_WITH_BLIS FALSE CACHE BOOL "Use BLIS library")
if(MGMOL_WITH_BLIS)
  add_definitions(-DMGMOL_USE_BLIS)
  message(STATUS "BLIS_LIBRARIES include: ${BLIS_INCLUDE_DIRS}")
  include_directories("${BLIS_INCLUDE_DIRS}")
endif()

if(NOT LAPACK_LIBRARIES)  
  find_package(LAPACK REQUIRED)
  message(STATUS "LAPACK_LIBARIES: ${LAPACK_LIBRARIES}")
  if(NOT ${LAPACK_FOUND})
    message(FATAL_ERROR "Required LAPACK library not found.")
  endif(NOT ${LAPACK_FOUND}) 
else(NOT LAPACK_LIBRARIES)
  message(STATUS "LAPACK_LIBARIES: ${LAPACK_LIBRARIES}")
endif(NOT LAPACK_LIBRARIES)

# Scalapack (required)
set (MGMOL_WITH_SCALAPACK True CACHE BOOL "Build with SCALAPACK")
if (${MGMOL_WITH_SCALAPACK} OR DEFINED SCALAPACK_ROOT)
  find_package(SCALAPACK)
  if(${SCALAPACK_FOUND})
    add_definitions(-DMGMOL_USE_SCALAPACK)
    message(STATUS "SCALAPACK_INCLUDE_DIRS: ${SCALAPACK_INCLUDE_DIRS}")
    message(STATUS "SCALAPACK_LIBARIES: ${SCALAPACK_LIBRARIES}")
  else(${SCALAPACK_FOUND})
    message(FATAL_ERROR "Required SCALAPACK package not found.")
  endif(${SCALAPACK_FOUND})  
endif(${MGMOL_WITH_SCALAPACK} OR DEFINED SCALAPACK_ROOT)

# libROM (optional)
set(USE_LIBROM False CACHE BOOL "Build with libROM")
set(LIBROM_PATH "" CACHE STRING "Path of libROM")
if(USE_LIBROM)
  message(STATUS "LIBROM_PATH: ${LIBROM_PATH}")
  if(NOT LIBROM_PATH)
    message(FATAL_ERROR "Cmake is asked to use libROM, but LIBROM_PATH not specified.")
  endif(NOT LIBROM_PATH)

  find_package(libROM REQUIRED)

  if(libROM_FOUND)
    add_definitions(-DMGMOL_HAS_LIBROM)
  endif(libROM_FOUND)
endif(USE_LIBROM)

# ARPACK (optional)
set(MGMOL_WITH_ARPACK FALSE CACHE BOOL "Compile with ARPACK package")
if(${MGMOL_WITH_ARPACK} OR DEFINED ARPACK_ROOT)
  find_package(ARPACK)
  if(${ARPACK_FOUND})
    message(STATUS "Build with ARPACK package")
    add_definitions(-DHAVE_ARPACK)
  endif(${ARPACK_FOUND})
endif(${MGMOL_WITH_ARPACK} OR DEFINED ARPACK_ROOT)

# MAGMA (optional)
set(MGMOL_WITH_MAGMA FALSE CACHE BOOL "Compile with MAGMA")
if(${MGMOL_WITH_MAGMA})
  find_package(MAGMA)
  if(${MAGMA_FOUND})
    message(STATUS "Build with MAGMA")
    add_definitions(-DHAVE_MAGMA)
  endif(${MAGMA_FOUND})
endif(${MGMOL_WITH_MAGMA})

# tricubic (optional)
set(MGMOL_WITH_TRICUBIC FALSE CACHE BOOL "Compile with tricubic package")
if(${MGMOL_WITH_TRICUBIC} OR DEFINED TRICUBIC_ROOT)
  find_package(TRICUBIC)
  if(${TRICUBIC_FOUND})
    message(STATUS "Build with tricubic package")
    add_definitions(-DHAVE_TRICUBIC)
  endif(${TRICUBIC_FOUND})
endif(${MGMOL_WITH_TRICUBIC} OR DEFINED TRICUBIC_ROOT)

# clang-format (optional)
set(MGMOL_WITH_CLANG_FORMAT FALSE CACHE BOOL "Indent code with clang-format")
if(${MGMOL_WITH_CLANG_FORMAT})
  find_package(CLANG_FORMAT)
  if(${CLANG_FORMAT_FOUND})
    message(STATUS "Indent with clang-format")
    file(GLOB_RECURSE FORMAT_SOURCES src/*.cc src/*.h tests/*.cc tests/*.h drivers/*.cc)
    add_custom_target(format
      COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
      DEPENDS ${FORMAT_SOURCES})
  endif(${CLANG_FORMAT_FOUND})
endif()

# code coverage
if(${MGMOL_WITH_COVERAGE})
  find_package(COVERAGE)
  if(${COVERAGE_FOUND})
    set(COVERAGE_FILE ${CMAKE_BINARY_DIR}/lcov.info)
    set(COVERAGE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/html_coverage)
    add_custom_target(coverage
      COMMAND ${LCOV_EXECUTABLE}
      --capture
      --directory ${CMAKE_BINARY_DIR}
      --output-file=${COVERAGE_FILE}
      COMMAND ${GENHTML_EXECUTABLE}
      ${COVERAGE_FILE}
      --output-directory
      ${COVERAGE_OUTPUT_DIRECTORY}
      )
  endif(${COVERAGE_FOUND})
endif(${MGMOL_WITH_COVERAGE})

enable_testing()

# additional definitions

include(FortranCInterface)
FortranCInterface_HEADER(
    fc_mangle.h
    MACRO_NAMESPACE "FC_BLASLAPACK_"
    SYMBOLS
        DSCAL SSCAL DAXPY SAXPY DDOT SDOT DCOPY SCOPY
        DNRM2 SNRM2 IDAMAX ISAMAX DROT SROT
        DGEMV SGEMV DSYMV
        #blas3
        DGEMM SGEMM DSYMM DSYRK SSYRK DTRMM DTRSM STRSM
        #lapack
        DSYEV DPOTRF DPOTRS DPOTRI DSYGST DTRTRS DPOCON DSYGV DLANGE
        DGETRF DGETRS DLACPY
)

if(${SCALAPACK_FOUND})
  FortranCInterface_HEADER(
    scalapack_mangle.h
    MACRO_NAMESPACE "FC_SCALAPACK_"
    SYMBOLS
        NUMROC INDXL2G INDXG2L
        pdswap psswap pdlange pslange pdlacp3 pslacp3 pdtran pstran
        pdsymm pssymm pdgemm pdgemv pdsymv psgemm
        psgemv pssymv pdsyrk pssyrk pdgemr2d psgemr2d pigemr2d
        pdtrmm pstrmm pdtrsm pstrsm pdtrtrs pstrtrs pdpotrf pspotrf
        pdpotrs pspotrs pdgetrf psgetrf pdgetrs psgetrs pdpotri pspotri
        pdtrtri pstrtri pdpocon pspocon pdsygst pssygst pdsyev pssyev
        pdelset pselset pdelget pselget pdlatra pslatra pdlaset pslaset pdgesvd psgesvd
        pdamax psamax
  )
endif(${SCALAPACK_FOUND})

FortranCInterface_HEADER(
    arpack_mangle.h
    MACRO_NAMESPACE "FC_"
    SYMBOLS
        PDSAUPD PDSEUPD DSAUPD DSEUPD
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
add_definitions(-DGITHASH=${GIT_COMMIT_HASH})

# add include directories
include_directories("${PROJECT_SOURCE_DIR}/src/DistMatrix")
include_directories("${PROJECT_SOURCE_DIR}/src/linear_algebra")
include_directories("${PROJECT_SOURCE_DIR}/src/local_matrices")
include_directories("${PROJECT_SOURCE_DIR}/src/numerical_kernels")
include_directories("${PROJECT_SOURCE_DIR}/src/pb")
include_directories("${PROJECT_SOURCE_DIR}/src/radial")
include_directories("${PROJECT_SOURCE_DIR}/src/sparse_linear_algebra")
include_directories("${PROJECT_SOURCE_DIR}/src/tools")
include_directories("${PROJECT_SOURCE_DIR}/src")

include_directories("${LIBROM_PATH}/lib")
link_libraries(${LIBROM_LIB})

# add subdirectories for source files, tests
add_subdirectory(src)

add_subdirectory(drivers)

add_subdirectory(tests)

