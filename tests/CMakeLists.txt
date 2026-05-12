find_package(PythonInterp 3.2 REQUIRED)

if(MGMOL_USE_HDF5P)
  add_executable(testHDF5P
                 ${CMAKE_SOURCE_DIR}/tests/testHDF5P.cc
                 ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
endif()
add_executable(testMPI
               ${CMAKE_SOURCE_DIR}/tests/testMPI.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
if(${MGMOL_WITH_SCALAPACK})
add_executable(testBlacsContext
               ${CMAKE_SOURCE_DIR}/tests/testBlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testDistVector
               ${CMAKE_SOURCE_DIR}/tests/testDistVector.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testReplicated2DistMatrix
               ${CMAKE_SOURCE_DIR}/tests/testReplicated2DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/LocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/SquareLocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testDistMatrix
               ${CMAKE_SOURCE_DIR}/tests/testDistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testConditionDistMatrix
               ${CMAKE_SOURCE_DIR}/tests/testConditionDistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testConditionDistMatrixPower
               ${CMAKE_SOURCE_DIR}/tests/testConditionDistMatrixPower.cc
               ${CMAKE_SOURCE_DIR}/src/Power.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/LocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/SquareLocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testPower
               ${CMAKE_SOURCE_DIR}/tests/testPower.cc
               ${CMAKE_SOURCE_DIR}/src/Power.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/LocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/SquareLocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_serial_main.cc)
add_executable(testPowerDistMatrix
               ${CMAKE_SOURCE_DIR}/tests/testPowerDistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/Power.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/LocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/SquareLocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
endif(${MGMOL_WITH_SCALAPACK})
add_executable(testDirectionalReduce
               ${CMAKE_SOURCE_DIR}/tests/testDirectionalReduce.cc
               ${CMAKE_SOURCE_DIR}/src/sparse_linear_algebra/DirectionalReduce.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testAndersonMix
               ${CMAKE_SOURCE_DIR}/tests/Anderson/testAndersonMix.cc
               ${CMAKE_SOURCE_DIR}/tests/Anderson/Solution.cc
               ${CMAKE_SOURCE_DIR}/src/AndersonMix.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc)
add_executable(testSuperSampling
               ${CMAKE_SOURCE_DIR}/tests/testSuperSampling.cc
               ${CMAKE_SOURCE_DIR}/src/SuperSampling.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_serial_main.cc)
add_executable(testVariableSizeMatrix
               ${CMAKE_SOURCE_DIR}/tests/testVariableSizeMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/sparse_linear_algebra/VariableSizeMatrix.cc               
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/sparse_linear_algebra/SparseRow.cc               
               ${CMAKE_SOURCE_DIR}/src/sparse_linear_algebra/SparseRowAndTable.cc     
               ${CMAKE_SOURCE_DIR}/src/sparse_linear_algebra/Table.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testTradeGhostValues
               ${CMAKE_SOURCE_DIR}/tests/testTradeGhostValues.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFuncVector.cc
               ${CMAKE_SOURCE_DIR}/src/Map2Masks.cc
               ${CMAKE_SOURCE_DIR}/src/pb/MGkernels.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDkernels.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testSetGhostValues
               ${CMAKE_SOURCE_DIR}/tests/testSetGhostValues.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFuncVector.cc
               ${CMAKE_SOURCE_DIR}/src/Map2Masks.cc
               ${CMAKE_SOURCE_DIR}/src/pb/MGkernels.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDkernels.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testLaph4
               ${CMAKE_SOURCE_DIR}/tests/testLaph4.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph4.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Lap.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph2.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDoper.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFuncVector.cc
               ${CMAKE_SOURCE_DIR}/src/Map2Masks.cc
               ${CMAKE_SOURCE_DIR}/src/pb/MGkernels.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDkernels.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testBatchLaph4
               ${CMAKE_SOURCE_DIR}/tests/testBatchLaph4.cc
               ${CMAKE_SOURCE_DIR}/src/Map2Masks.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph4.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Lap.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph2.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDoper.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFuncVector.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/pb/MGkernels.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDkernels.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testPoisson
               ${CMAKE_SOURCE_DIR}/tests/testPoisson.cc
               ${CMAKE_SOURCE_DIR}/src/PCGSolver.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph4.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Lap.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph2.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph6.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph8.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Laph4M.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDoper.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/Map2Masks.cc
               ${CMAKE_SOURCE_DIR}/src/pb/MGkernels.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDkernels.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testMGkernels
               ${CMAKE_SOURCE_DIR}/tests/testMGkernels.cc
               ${CMAKE_SOURCE_DIR}/src/Map2Masks.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/pb/Grid.cc
               ${CMAKE_SOURCE_DIR}/src/pb/PEenv.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFunc.cc
               ${CMAKE_SOURCE_DIR}/src/pb/GridFuncVector.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/pb/MGkernels.cc
               ${CMAKE_SOURCE_DIR}/src/pb/FDkernels.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
add_executable(testIons
               ${CMAKE_SOURCE_DIR}/tests/testIons.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
if(${MGMOL_WITH_SCALAPACK})
add_executable(testGramMatrix
               ${CMAKE_SOURCE_DIR}/tests/testGramMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/GramMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/Power.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/LocalMatrices2ReplicatedMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/LocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/SquareLocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/ReplicatedMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/MatricesBlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrixTools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_magma_main.cc)
add_executable(testDensityMatrix
               ${CMAKE_SOURCE_DIR}/tests/testDensityMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DensityMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/GramMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/Power.cc
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/LocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/local_matrices/SquareLocalMatrices.cc
               ${CMAKE_SOURCE_DIR}/src/LocalMatrices2ReplicatedMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/ReplicatedMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrix.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/BlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/MatricesBlacsContext.cc
               ${CMAKE_SOURCE_DIR}/src/DistMatrix/DistMatrixTools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/syrk_impl.cc
               ${CMAKE_SOURCE_DIR}/src/linear_algebra/gemm_impl.cc
               ${CMAKE_SOURCE_DIR}/src/tools/MGmol_MPI.cc
               ${CMAKE_SOURCE_DIR}/src/tools/mgmol_mpi_tools.cc
               ${CMAKE_SOURCE_DIR}/src/tools/random.cc
               ${CMAKE_SOURCE_DIR}/src/ReplicatedWorkSpace.cc
               ${CMAKE_SOURCE_DIR}/src/hdf_tools.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_magma_main.cc)
endif(${MGMOL_WITH_SCALAPACK})
add_executable(testRhoVhRestart
               ${CMAKE_SOURCE_DIR}/tests/RhoVhRestart/testRhoVhRestart.cc)
add_executable(testEnergyAndForces
               ${CMAKE_SOURCE_DIR}/tests/EnergyAndForces/testEnergyAndForces.cc)
add_executable(testWFEnergyAndForces
               ${CMAKE_SOURCE_DIR}/tests/WFEnergyAndForces/testWFEnergyAndForces.cc)
add_executable(testDMandEnergyAndForces
               ${CMAKE_SOURCE_DIR}/tests/DMandEnergyAndForces/testDMandEnergyAndForces.cc)
add_executable(testRestartEnergyAndForces
               ${CMAKE_SOURCE_DIR}/tests/RestartEnergyAndForces/testRestartEnergyAndForces.cc)
add_executable(testPinnedH2O_3DOF
               ${CMAKE_SOURCE_DIR}/tests/PinnedH2O_3DOF/testPinnedH2O_3DOF.cc
               ${CMAKE_SOURCE_DIR}/src/tools/PinnedH2O.cc)

if(${MAGMA_FOUND})
  add_executable(testOpenmpOffload
               ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
               ${CMAKE_SOURCE_DIR}/tests/testOpenmpOffload.cc
               ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
  if(${MGMOL_WITH_OPENMP_OFFLOAD})
    add_executable(testMagmaOpenmp
                   ${CMAKE_SOURCE_DIR}/tests/testMagmaOpenmp.cc
                   ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
  endif()
  add_executable(testReplicatedMatrix
                 ${CMAKE_SOURCE_DIR}/tests/testReplicatedMatrix.cc
                 ${CMAKE_SOURCE_DIR}/src/ReplicatedMatrix.cc
                 ${CMAKE_SOURCE_DIR}/src/local_matrices/LocalMatrices.cc
                 ${CMAKE_SOURCE_DIR}/src/local_matrices/SquareLocalMatrices.cc
                 ${CMAKE_SOURCE_DIR}/src/magma_singleton.cc
                 ${CMAKE_SOURCE_DIR}/src/linear_algebra/mputils.cc
                 ${CMAKE_SOURCE_DIR}/src/tools/Timer.cc
                 ${CMAKE_SOURCE_DIR}/src/tools/random.cc
                 ${CMAKE_SOURCE_DIR}/tests/ut_main.cc)
endif()
               
target_compile_definitions(testAndersonMix PUBLIC TESTING)

# unit tests
if(MGMOL_USE_HDF5P)
  add_test(NAME testHDF5P
           COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   ${CMAKE_CURRENT_BINARY_DIR}/testHDF5P)
endif()
add_test(NAME testMPI
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testMPI)
if(${MGMOL_WITH_SCALAPACK})
add_test(NAME testBlacsContext
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testBlacsContext)
add_test(NAME testDistVector
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testDistVector)
add_test(NAME testReplicated2DistMatrix
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testReplicated2DistMatrix)
add_test(NAME testDistMatrix
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testDistMatrix)
add_test(NAME testConditionDistMatrix
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testConditionDistMatrix)
add_test(NAME testConditionDistMatrixPower
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testConditionDistMatrixPower)
add_test(NAME testPowerDistMatrix
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testPowerDistMatrix)
add_test(NAME testPower
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testPower)
endif(${MGMOL_WITH_SCALAPACK})
add_test(NAME testDirectionalReduce
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testDirectionalReduce)
add_test(NAME testAndersonMix
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testAndersonMix 20 2)
add_test(NAME testSuperSampling
         COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testSuperSampling)
add_test(NAME testVariableSizeMatrix
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testVariableSizeMatrix)
add_test(NAME testTradeGhostValues
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testTradeGhostValues)
add_test(NAME testSetGhostValues
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testSetGhostValues)
add_test(NAME testLaph4
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testLaph4)
add_test(NAME testBatchLaph4
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testBatchLaph4)
add_test(NAME testPoisson
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testPoisson)
add_test(NAME testtMGkernels
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testMGkernels)
add_test(NAME testIons
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testIons)
set_tests_properties(testIons PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/potentials)

if(${MGMOL_WITH_SCALAPACK})
add_test(NAME testGramMatrix
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testGramMatrix)
add_test(NAME testDensityMatrix
         COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                 ${CMAKE_CURRENT_BINARY_DIR}/testDensityMatrix)
endif(${MGMOL_WITH_SCALAPACK})
add_test(NAME testEnergyAndForces
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/EnergyAndForces/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testEnergyAndForces
         ${CMAKE_CURRENT_SOURCE_DIR}/EnergyAndForces/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/EnergyAndForces/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/EnergyAndForces/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testWFEnergyAndForces
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/WFEnergyAndForces/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testWFEnergyAndForces
         ${CMAKE_CURRENT_SOURCE_DIR}/WFEnergyAndForces/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/WFEnergyAndForces/sih4.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testDMandEnergyAndForces
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/DMandEnergyAndForces/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testDMandEnergyAndForces
         ${CMAKE_CURRENT_SOURCE_DIR}/DMandEnergyAndForces/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/DMandEnergyAndForces/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/DMandEnergyAndForces/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testRestartEnergyAndForces
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/RestartEnergyAndForces/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_BINARY_DIR}/testRestartEnergyAndForces
         ${CMAKE_CURRENT_SOURCE_DIR}/RestartEnergyAndForces/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/RestartEnergyAndForces/restart.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/RestartEnergyAndForces/h2o.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testPinnedH2O_3DOF
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/PinnedH2O_3DOF/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/testPinnedH2O_3DOF
         ${CMAKE_CURRENT_SOURCE_DIR}/PinnedH2O_3DOF/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/PinnedH2O_3DOF/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/PinnedH2O_3DOF/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testRhoVhRestart
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/RhoVhRestart/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_BINARY_DIR}/testRhoVhRestart
         ${CMAKE_CURRENT_SOURCE_DIR}/RhoVhRestart/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/RhoVhRestart/md.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/RhoVhRestart/restart.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/RhoVhRestart/h2o.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)


if(${MAGMA_FOUND}) 
  add_test(NAME testOpenmpOffload
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testOpenmpOffload)
  if(${MGMOL_WITH_OPENMP_OFFLOAD})
    add_test(NAME testMagmaOpenmp
             COMMAND ${CMAKE_CURRENT_BINARY_DIR}/testMagmaOpenmp)
  endif()
  add_test(NAME testReplicatedMatrix
           COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
                   ${CMAKE_CURRENT_BINARY_DIR}/testReplicatedMatrix)
else()
  add_test(NAME testProjectedMatrices
           COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ProjectedMatrices/test.py
           ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
           ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
           ${CMAKE_CURRENT_SOURCE_DIR}/ProjectedMatrices/quenchExact.cfg
           ${CMAKE_CURRENT_SOURCE_DIR}/ProjectedMatrices/quenchShortSighted.cfg
           ${CMAKE_CURRENT_SOURCE_DIR}/ProjectedMatrices/coords.in
           ${CMAKE_CURRENT_SOURCE_DIR}/ProjectedMatrices/lrs.in
           ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
endif()

#Regression tests
add_test(NAME testSiH4
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/SiH4/testSiH4.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/SiH4/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/SiH4/sih4.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testOverrideCoords
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/OverrideCoords/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/OverrideCoords/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/OverrideCoords/restart.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/OverrideCoords/sih4.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/OverrideCoords/shifted.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testCl2
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Cl2_ONCVPSP_LDA/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/Cl2_ONCVPSP_LDA/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/Cl2_ONCVPSP_LDA/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testForcesSymmetry
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ForcesSymmetry/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/ForcesSymmetry/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/ForcesSymmetry/p4.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testContinuumSolvent
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ContinuumSolvent/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/ContinuumSolvent/mgmol_diel.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/ContinuumSolvent/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/ContinuumSolvent/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testPinnedMolecule
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/PinnedMolecule/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/PinnedMolecule/quench.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/PinnedMolecule/md.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/PinnedMolecule/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testDipole
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Dipole/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/Dipole/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/Dipole/efield.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/Dipole/h2o.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME Davidson
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Davidson/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/Davidson/davidson.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/Davidson/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME DavidsonReplicated
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Davidson/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/DavidsonReplicated/davidson.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/Davidson/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME DavidsonMixing
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/DavidsonMixing/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/DavidsonMixing/davidson.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/DavidsonMixing/sih4.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME LocalPotentials
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/LocalPotentials/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/LocalPotentials/davidson.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/LocalPotentials/li2.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME PolakRibiere
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/PolakRibiere/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/PolakRibiere/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/PolakRibiere/sih4.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testSpinO2
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/SpinO2/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/SpinO2/mgmol_spin1.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/SpinO2/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testSpinO2LDA
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/SpinO2LDA/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/SpinO2LDA/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/SpinO2LDA/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
if(${MGMOL_WITH_SCALAPACK})
add_test(NAME testMVP
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MVP/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/MVP/mvp.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MVP/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
endif(${MGMOL_WITH_SCALAPACK})
add_test(NAME testMVPReplicated
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MVP/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/MVPReplicated/mvp.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MVP/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testMVPmix
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MVPmix/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/MVPmix/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MVPmix/h2o.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME BandGapN2
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/BandGapN2/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/BandGapN2/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/BandGapN2/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/BandGapN2/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testMLWF
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MLWF/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/MLWF/mlwf.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MLWF/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testSpreadPenalty
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/SpreadPenalty/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/SpreadPenalty/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/SpreadPenalty/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/SpreadPenalty/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME HMVP
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/HMVP/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/HMVP/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/HMVP/F.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME HMVPmix
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/HMVPmix/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/HMVPmix/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/HMVPmix/h2o.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME NonAufbau
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/NonAufbau/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/NonAufbau/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/NonAufbau/restart.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/NonAufbau/h2o.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME ReplicatedSP2
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ReplicatedSP2/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/ReplicatedSP2/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/ReplicatedSP2/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/ReplicatedSP2/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testMD_NoRestart
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MD_NoRestart/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_NoRestart/mgmol.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_NoRestart/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testMD_D72
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MD_D72/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 5 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_D72/mgmol_quench.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_D72/mgmol_md.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_D72/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_D72/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
if(MGMOL_USE_HDF5P)
  add_test(NAME testHDF5single
           COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/HDF5single/test.py
           ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
           ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
           ${CMAKE_CURRENT_SOURCE_DIR}/HDF5single/mgmol.cfg
           ${CMAKE_CURRENT_SOURCE_DIR}/HDF5single/md.cfg
           ${CMAKE_CURRENT_SOURCE_DIR}/HDF5single/h2o.xyz
           ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
endif()
add_test(NAME testMD_MVP
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/MD_MVP/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_MVP/quench.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_MVP/md.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/MD_MVP/li2.xyz
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testLBFGS
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/LBFGS/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/LBFGS/lbfgs1.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/LBFGS/lbfgs2.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/LBFGS/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/LBFGS/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
add_test(NAME testFIRE
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/FIRE/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/FIRE/fire1.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/FIRE/fire2.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/FIRE/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/FIRE/lrs.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
if(${MGMOL_WITH_SCALAPACK})
add_test(NAME ChebyshevMVP
         COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Chebyshev/test.py
         ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
         ${CMAKE_CURRENT_SOURCE_DIR}/Chebyshev/cheb.cfg
         ${CMAKE_CURRENT_SOURCE_DIR}/Chebyshev/coords.in
         ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)

endif(${MGMOL_WITH_SCALAPACK})
if(NOT ${MGMOL_WITH_MAGMA})
  add_test(NAME testShortSighted
           COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/ShortSighted/test.py
           ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 5 ${MPIEXEC_PREFLAGS}
           ${CMAKE_CURRENT_BINARY_DIR}/../src/mgmol-opt
           ${CMAKE_CURRENT_SOURCE_DIR}/ShortSighted/quench.cfg
           ${CMAKE_CURRENT_SOURCE_DIR}/ShortSighted/md.cfg
           ${CMAKE_CURRENT_SOURCE_DIR}/ShortSighted/coords.in
           ${CMAKE_CURRENT_SOURCE_DIR}/ShortSighted/lrs.in
           ${CMAKE_CURRENT_SOURCE_DIR}/../potentials)
endif()

if(MGMOL_USE_HDF5P)
  target_include_directories(testHDF5P PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries(testHDF5P PRIVATE ${HDF5_LIBRARIES} MPI::MPI_CXX)
endif()

target_include_directories(testAndersonMix PRIVATE
                           ${CMAKE_SOURCE_DIR}/tests/Anderson
                           ${HDF5_INCLUDE_DIRS})
if(${MGMOL_WITH_SCALAPACK})
target_include_directories(testDistVector PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testReplicated2DistMatrix PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testDistMatrix PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testConditionDistMatrix PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testConditionDistMatrixPower PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testPower PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testPowerDistMatrix PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testDensityMatrix PRIVATE ${Boost_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_include_directories(testGramMatrix PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(testBlacsContext PRIVATE ${SCALAPACK_LIBRARIES}
  ${BLAS_LIBRARIES} MPI::MPI_CXX)
target_link_libraries(testDensityMatrix PRIVATE ${HDF5_LIBRARIES})
endif(${MGMOL_WITH_SCALAPACK})

target_include_directories(testAndersonMix PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(testIons PRIVATE ${Boost_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
target_include_directories(testRhoVhRestart PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(testMPI PRIVATE MPI::MPI_CXX)
target_link_libraries(testSuperSampling PRIVATE MPI::MPI_CXX)
target_link_libraries(testDirectionalReduce PRIVATE MPI::MPI_CXX)
target_link_libraries(testEnergyAndForces PRIVATE mgmol_src)
target_link_libraries(testWFEnergyAndForces PRIVATE mgmol_src)
target_link_libraries(testDMandEnergyAndForces PRIVATE mgmol_src)
target_link_libraries(testRestartEnergyAndForces PRIVATE mgmol_src)
target_link_libraries(testPinnedH2O_3DOF PRIVATE mgmol_src)
target_link_libraries(testIons PRIVATE mgmol_src)
target_link_libraries(testRhoVhRestart mgmol_src)

if(${MAGMA_FOUND})
  target_link_libraries(testDistVector PRIVATE ${SCALAPACK_LIBRARIES}
    ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testReplicated2DistMatrix PRIVATE ${SCALAPACK_LIBRARIES}
    ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testDistMatrix PRIVATE ${SCALAPACK_LIBRARIES}
    ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testConditionDistMatrix PRIVATE ${SCALAPACK_LIBRARIES}
    ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testConditionDistMatrixPower PRIVATE
    ${SCALAPACK_LIBRARIES} ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testReplicatedMatrix PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testPower PRIVATE ${BLAS_LIBRARIES} ${SCALAPACK_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testPowerDistMatrix PRIVATE ${BLAS_LIBRARIES}
    ${SCALAPACK_LIBRARIES} MPI::MPI_CXX
    OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testAndersonMix PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} 
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testVariableSizeMatrix PRIVATE ${BLAS_LIBRARIES} 
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testTradeGhostValues PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testSetGhostValues PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testLaph4 PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testBatchLaph4 PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testMGkernels PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testGramMatrix PRIVATE ${SCALAPACK_LIBRARIES}
    ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testDensityMatrix PRIVATE ${SCALAPACK_LIBRARIES}
    ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  target_link_libraries(testOpenmpOffload PRIVATE 
    MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)

  if(${MGMOL_WITH_OPENMP_OFFLOAD})
    target_link_libraries(testMagmaOpenmp PRIVATE 
      MPI::MPI_CXX OpenMP::OpenMP_CXX PkgConfig::MAGMA)
  endif()
else()
  if(${MGMOL_WITH_SCALAPACK})
    target_link_libraries(testDistVector PRIVATE ${SCALAPACK_LIBRARIES}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_link_libraries(testReplicated2DistMatrix PRIVATE ${SCALAPACK_LIBRARIES}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_link_libraries(testDistMatrix PRIVATE ${SCALAPACK_LIBRARIES}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_link_libraries(testConditionDistMatrix PRIVATE ${SCALAPACK_LIBRARIES}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_link_libraries(testConditionDistMatrixPower PRIVATE
      ${SCALAPACK_LIBRARIES} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
      MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_link_libraries(testPowerDistMatrix PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
      ${SCALAPACK_LIBRARIES} MPI::MPI_CXX
      OpenMP::OpenMP_CXX)
    target_link_libraries(testPower PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
      ${SCALAPACK_LIBRARIES}
      MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_link_libraries(testGramMatrix PRIVATE ${SCALAPACK_LIBRARIES}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_link_libraries(testDensityMatrix PRIVATE ${SCALAPACK_LIBRARIES}
      ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} MPI::MPI_CXX OpenMP::OpenMP_CXX)
  endif(${MGMOL_WITH_SCALAPACK})
  target_link_libraries(testAndersonMix PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES} 
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(testVariableSizeMatrix PRIVATE ${BLAS_LIBRARIES} 
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(testTradeGhostValues PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(testSetGhostValues PRIVATE ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(testLaph4 PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(testBatchLaph4 PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(testPoisson PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_link_libraries(testMGkernels PRIVATE ${BLAS_LIBRARIES}
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()

set_tests_properties(testSiH4 PROPERTIES REQUIRED_FILES
                     ${CMAKE_SOURCE_DIR}/potentials/pseudo.Si)
