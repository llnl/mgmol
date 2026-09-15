FROM ubuntu:22.04

ENV ENVDIR=env

# install sudo
RUN apt-get -yq update && apt-get -yq install sudo

WORKDIR /$ENVDIR

# install packages
RUN sudo apt-get install -yq git
RUN sudo apt-get install --no-install-recommends -yq make gcc gfortran libssl-dev cmake
RUN sudo apt-get install -yq libopenblas-dev libmpich-dev libblas-dev liblapack-dev libscalapack-mpi-dev libhdf5-mpi-dev
RUN sudo apt-get install -yq libboost-all-dev
RUN sudo apt-get install -yq vim
RUN sudo apt-get install -yq git-lfs
RUN sudo apt-get install -yq valgrind hdf5-tools
RUN sudo apt-get install -yq wget
### clang-format seems to be updated to 14.0. Not using it for now.
# RUN sudo apt-get install -yq clang-format

# install lldb and gdb for debugging
RUN sudo apt-get install -yq lldb gdb

RUN sudo apt-get clean -q

ENV LIB_DIR=/$ENVDIR/dependencies
WORKDIR $LIB_DIR

# cmake toolchain file for librom
RUN echo 'set(CMAKE_C_COMPILER mpicc)\n\
set(CMAKE_CXX_COMPILER mpicxx)\n\
set(CMAKE_Fortran_COMPILER mpif90)' > ./librom_env.cmake
ENV TOOLCHAIN_FILE=$LIB_DIR/librom_env.cmake

# install libROM for scaleupROM
RUN sudo git clone https://github.com/LLNL/libROM.git
WORKDIR ./libROM/build
# libROM without MFEM.
RUN sudo cmake .. -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} -DCMAKE_BUILD_TYPE=Optimized -DUSE_MFEM=OFF
RUN sudo make -j 16

# create and switch to a user
ENV USERNAME=test
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN useradd --no-log-init -u 1001 --create-home --shell /bin/bash $USERNAME
RUN adduser $USERNAME sudo
USER $USERNAME
WORKDIR /home/$USERNAME

