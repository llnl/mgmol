
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/mgmol_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/mgmol_config.h @ONLY)

add_subdirectory(linear_algebra)
add_subdirectory(local_matrices)
add_subdirectory(numerical_kernels)
add_subdirectory(pb)
add_subdirectory(radial)
add_subdirectory(sparse_linear_algebra)
add_subdirectory(tools)
if(${MGMOL_WITH_SCALAPACK})
  add_subdirectory(DistMatrix)
endif(${MGMOL_WITH_SCALAPACK})

if(${MGMOL_WITH_SCALAPACK})
set(link_libs mgmol_distmatrix 
   mgmol_linear_algebra 
   mgmol_local_matrices 
   mgmol_numerical_kernels 
   mgmol_pb 
   mgmol_radial 
   mgmol_sparse_linear_algebra 
   mgmol_tools
   )
else(${MGMOL_WITH_SCALAPACK})
set(link_libs
   mgmol_linear_algebra
   mgmol_local_matrices
   mgmol_numerical_kernels
   mgmol_pb
   mgmol_radial
   mgmol_sparse_linear_algebra
   mgmol_tools
   )
endif(${MGMOL_WITH_SCALAPACK})

set(SOURCES
 mgmol_run.cc
 read_config.cc
 jade.cc
 DotProductSimple.cc
 DotProductDiagonal.cc
 DotProductWithInvS.cc
 DotProductWithDM.cc
 LocalMatrices2ReplicatedMatrix.cc
 ReplicatedMatrix2SquareLocalMatrices.cc
 DielectricControl.cc
 ReplicatedMatrix.cc
 ReplicatedVector.cc
 DMStrategyFactory.cc
 manage_memory.cc
 SubCell.cc
 SinCosOps.cc
 XYZOps.cc
 DipoleMoment.cc
 PolakRibiereSolver.cc 
 EnergySpreadPenalty.cc 
 SpreadPenalty.cc 
 SpreadPenaltyVolume.cc 
 setup.cc 
 MVP_DMStrategy.cc 
 HamiltonianMVP_DMStrategy.cc 
 MVPSolver.cc 
 HamiltonianMVPSolver.cc 
 MGOrbitalsPreconditioning.cc
 DFTsolver.cc 
 NonOrthoDMStrategy.cc 
 FullyOccupiedNonOrthoDMStrategy.cc 
 EigenDMStrategy.cc 
 Masks4Orbitals.cc 
 AOMMprojector.cc 
 hdf_tools.cc 
 DistributedIonicData.cc 
 IonData.cc 
 SubspaceProjector.cc 
 MDfiles.cc 
 OrbitalsExtrapolation.cc 
 OrbitalsExtrapolationOrder2.cc 
 OrbitalsExtrapolationOrder3.cc 
 runfire.cc 
 FIRE.cc 
 IonicAlgorithm.cc 
 FIRE_IonicStepper.cc 
 tools.cc 
 MGmol.cc 
 MGmol_NEB.cc 
 ABPG.cc 
 LBFGS.cc 
 IonicStepper.cc 
 Energy.cc 
 GramMatrix.cc 
 DensityMatrix.cc 
 DensityMatrixSparse.cc 
 ConstraintSet.cc 
 DistanceConstraint.cc 
 MultiDistanceConstraint.cc 
 getLAeigen.cc 
 XConGrid.cc
 LDAFunctional.cc 
 LDAonGrid.cc 
 LDAonGridLibXC.cc
 LDAonGridSpin.cc
 LDAonGridSpinLibXC.cc
 PBEFunctional.cc 
 PBEonGrid.cc
 PBEonGridLibXC.cc
 PBEonGridSpin.cc
 PBEonGridSpinLibXC.cc
 Electrostatic.cc 
 ProjectedMatrices.cc 
 ProjectedMatricesSparse.cc 
 ProjectedMatrices2N.cc 
 ProjectedMatricesMehrstellen.cc 
 ReplicatedWorkSpace.cc 
 Rho.cc 
 Mesh.cc 
 Control.cc 
 Hamiltonian.cc 
 BlockVector.cc 
 KBPsiMatrixInterface.cc 
 KBPsiMatrixSparse.cc 
 KBprojectorSparse.cc 
 MultipoleExpansion.cc 
 SpreadsAndCenters.cc 
 Preconditioning.cc 
 LocalizationRegions.cc 
 Hartree.cc 
 ShiftedHartree.cc 
 PBdiel.cc 
 Species.cc 
 HDFrestart.cc 
 MasksSet.cc 
 Map2Masks.cc
 FunctionsPacking.cc 
 ColoredRegions.cc 
 LocGridOrbitals.cc 
 ExtendedGridOrbitals.cc
 mcstep.cc 
 MD_IonicStepper.cc 
 LBFGS_IonicStepper.cc 
 lbfgsrlx.cc 
 OrthoAndersonMix.cc
 AndersonMix.cc 
 Ion.cc 
 GridMask.cc 
 GridMaskMult.cc 
 GridMaskMax.cc 
 Ions.cc 
 restart.cc 
 md.cc 
 get_vnlpsi.cc 
 quench.cc 
 readInput.cc 
 Forces.cc 
 computeHij.cc 
 Potentials.cc 
 ShortSightedInverse.cc 
 PCGSolver_Diel.cc 
 PBdiel_CG.cc 
 PCGSolver.cc 
 Hartree_CG.cc 
 ClusterOrbitals.cc
 SP2.cc
 Power.cc
 PowerGen.cc
 SuperSampling.cc
 DavidsonSolver.cc
 magma_singleton.cc
 ChebyshevApproximation.cc
 ChebyshevApproximationInterface.cc
 mlwf.cc
 rom.cc
)

if(USE_LIBROM)
  list(APPEND SOURCES
    rom_workflows.cc)
endif(USE_LIBROM)

if(${MGMOL_WITH_SCALAPACK})
  set(SCALAPACK_SOURCES
    SquareSubMatrix2DistMatrix.cc
    LocalMatrices2DistMatrix.cc
    DistMatrix2SquareLocalMatrices.cc
    GrassmanCGFactory.cc
    GrassmanLineMinimization.cc
    GrassmanCG.cc
    GrassmanCGSparse.cc
    MLWFTransform.cc
    NOLMOTransform.cc
    OrbitalsTransform.cc
  )
  add_library(mgmol_src ${SOURCES} ${SCALAPACK_SOURCES})
else()
  add_library(mgmol_src ${SOURCES})
endif(${MGMOL_WITH_SCALAPACK})

target_include_directories(mgmol_src PRIVATE ${HDF5_INCLUDE_DIRS})
target_include_directories(mgmol_src PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(mgmol_src PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(USE_LIBROM)
  target_include_directories (mgmol_src PRIVATE ${LIBROM_INCLUDES})
  target_link_libraries(mgmol_src ${LIBROM_LIB})
endif(USE_LIBROM)

target_link_libraries(mgmol_src ${link_libs})
if(${MGMOL_WITH_MAGMA})
  target_link_libraries(mgmol_src PkgConfig::MAGMA)
endif()
if(${MGMOL_WITH_LIBXC})
  target_include_directories(mgmol_src PRIVATE ${LIBXC_DIR}/include)
endif (${MGMOL_WITH_LIBXC})

install(TARGETS mgmol_src DESTINATION lib)

# build executable
add_executable(mgmol-opt main.cc)
target_include_directories (mgmol-opt PRIVATE ${Boost_INCLUDE_DIRS})

target_link_libraries(mgmol_src ${link_libs})
target_link_libraries(mgmol_src ${SCALAPACK_LIBRARIES})
target_link_libraries(mgmol_src ${LAPACK_LIBRARIES})
target_link_libraries(mgmol_src ${BLAS_LIBRARIES})
target_link_libraries(mgmol_src ${HDF5_LIBRARIES})
target_link_libraries(mgmol_src ${HDF5_HL_LIBRARIES})
target_link_libraries(mgmol_src ${Boost_LIBRARIES})

target_link_libraries(mgmol-opt mgmol_src)
if (${OPENMP_CXX_FOUND})
  target_link_libraries(mgmol-opt OpenMP::OpenMP_CXX)
endif()
if(${MGMOL_WITH_LIBXC})
  target_link_libraries(mgmol-opt ${LIBXC_DIR}/lib/libxc.a)
endif (${MGMOL_WITH_LIBXC})

install(TARGETS mgmol-opt DESTINATION bin)

# build ROM executable
if(USE_LIBROM)
  add_executable(mgmol-rom rom_main.cc)
  target_include_directories (mgmol-rom PRIVATE ${Boost_INCLUDE_DIRS})

  target_link_libraries(mgmol-rom mgmol_src ${link_libs})
  target_link_libraries(mgmol-rom ${SCALAPACK_LIBRARIES})
  target_link_libraries(mgmol-rom ${HDF5_LIBRARIES})
  target_link_libraries(mgmol-rom ${HDF5_HL_LIBRARIES})
  target_link_libraries(mgmol-rom ${BLAS_LIBRARIES})
  target_link_libraries(mgmol-rom ${LAPACK_LIBRARIES})
  target_link_libraries(mgmol-rom ${Boost_LIBRARIES})
  target_link_libraries(mgmol-rom ${LIBROM_LIB})
  if (${OPENMP_CXX_FOUND})
    target_link_libraries(mgmol-rom OpenMP::OpenMP_CXX)
  endif()
  if(${MGMOL_WITH_LIBXC})
    target_link_libraries(mgmol-rom ${LIBXC_DIR}/lib/libxc.a)
  endif (${MGMOL_WITH_LIBXC})
  install(TARGETS mgmol-rom DESTINATION bin)
endif(USE_LIBROM)
