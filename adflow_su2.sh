sudo apt install build-essential  gfortran valgrind cmake libblas-dev liblapack-dev swig (diffutils)
echo '#!/bin/bash' > $HOME/mdo/loadmdo.sh && \

echo '# miniconda3' >> $HOME/mdo/loadmdo.sh && \
bash /mnt/e/Downloads/WSL/Miniconda3-py39_4.9.2-Linux-x86_64.sh -b -p $HOME/mdo/packages/miniconda3 && \
cd ~/mdo/packages/miniconda3/condabin/ && ./conda init && . ~/.bashrc && \
conda create -n mdo --clone base && \
echo 'conda activate mdo' >> $HOME/mdo/loadmdo.sh && . ~/mdo/loadmdo.sh && \
cd ~/ && mkdir .pip && cd .pip && echo -e '[global]\nindex-url=https://mirrors.ustc.edu.cn/pypi/web/simple' > pip.conf && \
pip install scipy && \
pip install numpy==1.23


echo '# OpenMPI' >> $HOME/mdo/loadmdo.sh && \
echo 'export MPI_INSTALL_DIR=$HOME/mdo/packages/openmpi/opt-gfortran' >> $HOME/mdo/loadmdo.sh && \
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:}$MPI_INSTALL_DIR/lib' >> $HOME/mdo/loadmdo.sh && \
echo 'export PATH=$MPI_INSTALL_DIR/bin:$PATH' >> $HOME/mdo/loadmdo.sh && \
. ~/mdo/loadmdo.sh && \
cd ~/mdo/packages/ && mkdir openmpi && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/openmpi-4.1.5.tar.gz -C openmpi/ --strip-components=1 && \
cd openmpi && ./configure --prefix=$MPI_INSTALL_DIR && make -j60 all && make install && ldconfig && \
pip install mpi4py==3.1.4


echo '# PETSc' >> $HOME/mdo/loadmdo.sh && \
echo 'export PETSC_ARCH=real-opt' >> $HOME/mdo/loadmdo.sh && \
echo 'export PETSC_DIR=$HOME/mdo/packages/petsc' >> $HOME/mdo/loadmdo.sh && \
. ~/mdo/loadmdo.sh && \
cd ~/mdo/packages/ && mkdir -p petsc && cd petsc && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/petsc-3.14.6.tar.gz --strip-components=1 && \
./configure --PETSC_ARCH=real-opt --with-scalar-type=real --with-debugging=0 --download-metis=yes --download-parmetis=yes --download-superlu_dist=yes --download-fblaslapack=yes --with-shared-libraries=yes --with-fortran-bindings=1 --with-cxx-dialect=C++11 --COPTFLAGS=-O3 --CXXOPTFLAGS=-O3 --FOPTFLAGS=-O3  && \
make PETSC_DIR=$PETSC_DIR  PETSC_ARCH=$PETSC_ARCH all && \
cd $PETSC_DIR/src/binding/petsc4py && pip install . && \

echo '# CGNS' >> $HOME/mdo/loadmdo.sh && \
echo 'export CGNS_HOME=$HOME/mdo/packages/CGNS/opt-gfortran' >> $HOME/mdo/loadmdo.sh && \
echo 'export PATH=$PATH:$CGNS_HOME/bin' >> $HOME/mdo/loadmdo.sh && \
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:}$CGNS_HOME/lib' >> $HOME/mdo/loadmdo.sh && \
. ~/mdo/loadmdo.sh && \
cd ~/mdo/packages/ && mkdir -p CGNS && cd CGNS && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/CGNS-4.3.0.tar.gz --strip-components=1 && \
cmake -D CGNS_ENABLE_FORTRAN=ON -D CMAKE_INSTALL_PREFIX=$CGNS_HOME -D CGNS_ENABLE_64BIT=OFF -D CGNS_ENABLE_HDF5=OFF -D CGNS_BUILD_CGNSTOOLS=OFF -D CMAKE_C_FLAGS="-fPIC -O3 -march=native -mtune=native" -D CMAKE_Fortran_FLAGS="-fPIC -O3 -march=native -mtune=native" .&& \
make install && \

cd ~/mdo && mkdir repos && cd repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/baseclasses-1.7.0.tar.gz && \
cd baseclasses* && pip install . && \ 

cd ~/mdo/repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/pyspline-1.5.2.tar.gz && \
cd pyspline* && cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
make && pip install . && \

cd ~/mdo/repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/pygeo-1.12.3.tar.gz && \
cd pygeo* && pip install . && \

cd ~/mdo/repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/cgnsutilities-2.7.0.tar.gz && \
cd cgnsutilities* && cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
make && pip install . && \

cd ~/mdo/repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/idwarp-2.6.0.tar.gz && \
cd idwarp* && \
cp  config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
make && pip install .[multi] && \

cd ~/mdo && mkdir adflow && cd adflow && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/adflow.tar.gz --strip-components=1 && \
cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && \
make && pip install . && \


cd ~/mdo/repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/pyoptsparse-2.9.3.tar.gz && \
cd pyoptsparse* && \
pip install . && \


cd ~/mdo/repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/pyhyp-2.6.1.tar.gz && \
cd pyhyp* && cp config/defaults/config.LINUX_GFORTRAN_OPENMPI.mk config/config.mk && \
make && pip install . && \

cd ~/mdo/repos && \
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/multipoint-1.4.0.tar.gz && \
cd multipoint* && pip install .



echo '# SU2' >> $HOME/mdo/loadmdo.sh && \
echo 'export SU2_INSTALL_DIR=$HOME/su2/SU2' >> $HOME/mdo/loadmdo.sh && \
echo 'export SU2_RUN=$SU2_INSTALL_DIR/bin' >> $HOME/mdo/loadmdo.sh && \
echo 'export PATH=$SU2_RUN:$PATH' >> $HOME/mdo/loadmdo.sh && \
echo 'export PYTHONPATH=$SU2_RUN:$PYTHONPATH' >> $HOME/mdo/loadmdo.sh && \
. $HOME/mdo/loadmdo.sh && \
cd ~/su2/ && mkdir SU2_code && cd SU2_code && \
tar -xvf /mnt/e/Downloads/WSL/SU2/SU2_code.tar.gz --strip-components=1 && \
cd ~/su2/SU2_code/externals/codi && git clone https://ghproxy.com/https://github.com/SciCompKL/CoDiPack.git . && \
cd ~/su2/SU2_code/externals/medi && git clone https://ghproxy.com/https://github.com/SciCompKL/MeDiPack.git . && \
cd ~/su2/SU2_code/externals/mel && git clone https://ghproxy.com/https://github.com/pcarruscag/MEL.git . && \
cd ~/su2/SU2_code/externals/meson && git clone https://ghproxy.com/https://github.com/mesonbuild/meson.git . && \
cd ~/su2/SU2_code/externals/ninja && git clone https://ghproxy.com/https://github.com/ninja-build/ninja.git . && \
cd ~/su2/SU2_code/externals/opdi && git clone https://ghproxy.com/https://github.com/SciCompKL/OpDiLib.git . && \

export CXXFLAGS="-march=native -funroll-loops -O2" && \
export LIBRARY_PATH=/opt/mambaforge/envs/mdo/lib && \
export C_INCLUDE_PATH=/opt/mambaforge/envs/mdo/include && \
export CPLUS_INCLUDE_PATH=/opt/mambaforge/envs/mdo/include && \
export MPICC=/opt/mambaforge/envs/mdo/include
./meson.py build -Denable-autodiff=true -Denable-directdiff=true -Denable-pywrapper=true  --prefix=/opt/su2 && \
./ninja -C build install -j60 && \



conda create -n adfoam --clone mdo && \

echo '#! /bin/bash' > ~/mdo/loadadfoam.sh && \
echo 'source ~/mdo/loadmdo.sh' >> ~/mdo/loadadfoam.sh && \
echo 'conda activate adfoam' >> ~/mdo/loadadfoam.sh && \
echo '# OpenMPI-3.1.6' >> ~/mdo/loadadfoam.sh && \
echo 'export MPI_INSTALL_DIR=$HOME/mdo/packages/openmpi-3.1.6/opt-gfortran' >> ~/mdo/loadadfoam.sh && \
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPI_INSTALL_DIR/lib' >> ~/mdo/loadadfoam.sh && \
echo 'export PATH=$MPI_INSTALL_DIR/bin:$PATH' >> ~/mdo/loadadfoam.sh && \
chmod u+x ~/mdo/loadadfoam.sh && \
. ~/mdo/loadadfoam.sh

cd ~/mdo/packages && mkdir openmpi-3.1.6 && cd openmpi-3.1.6 && \
tar -xvf /mnt/e/Downloads/WSL/dafoam/openmpi-3.1.6.tar.gz --strip-components=1 && \
./configure --prefix=$MPI_INSTALL_DIR && \
make all install && \
pip install mpi4py==3.1.3




mkdir -p ~/mdo/OpenFOAM && cd $HOME/mdo/OpenFOAM && \
tar -xvf /mnt/e/Downloads/WSL/OpenFOAM/OpenFOAM-v1812.tgz && \
tar -xvf /mnt/e/Downloads/WSL/OpenFOAM/ThirdParty-v1812.tgz && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812 && \
sed -i 's/$HOME/$HOME\/mdo/g' etc/bashrc && \
wget https://github.com/DAFoam/files/releases/download/v1.0.0/UPstream.C && \
mv UPstream.C src/Pstream/mpi/UPstream.C && \
echo '# OpenFOAM-v1812' >> $HOME/mdo/loadmdo.sh && \
echo 'source $HOME/mdo/OpenFOAM/OpenFOAM-v1812/etc/bashrc' >> $HOME/mdo/loadmdo.sh && \
echo 'export LD_LIBRARY_PATH=$HOME/mdo/OpenFOAM/sharedLibs:$LD_LIBRARY_PATH' >> $HOME/mdo/loadmdo.sh && \
echo 'export PATH=$HOME/mdo/OpenFOAM/sharedBins:$PATH' >> $HOME/mdo/loadmdo.sh && \
. $HOME/mdo/loadmdo.sh && \
export WM_NCOMPPROCS=60 && \
./Allwmake && \


cd $HOME/mdo/OpenFOAM && \
tar -xvf /mnt/e/Downloads/WSL/OpenFOAM/OpenFOAM-v1812-AD-1.2.9.tar.gz && mv OpenFOAM-v1812-AD-* OpenFOAM-v1812-ADR && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812-ADR && \
sed -i 's/WM_PROJECT_VERSION=v1812-AD/WM_PROJECT_VERSION=v1812-ADR/g' etc/bashrc && \
sed -i 's/$HOME/$HOME\/mdo/g' etc/bashrc && \
sed -i 's/export WM_CODI_AD_MODE=CODI_AD_FORWARD/export WM_CODI_AD_MODE=CODI_AD_REVERSE/g' etc/bashrc && \
. $HOME/mdo/loadadfoam.sh && \
source etc/bashrc && \
export WM_NCOMPPROCS=60 && \
./Allwmake 2> warningLog.txt && \
rm warningLog.txt && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812/platforms/*/lib && \
ln -s ../../../../OpenFOAM-v1812-ADR/platforms/*/lib/*.so . && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/dummy && \
ln -s ../../../../../OpenFOAM-v1812-ADR/platforms/*/lib/dummy/*.so . && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/openmpi-system && \
ln -s ../../../../../OpenFOAM-v1812-ADR/platforms/*/lib/openmpi-system/*.so .

cd $HOME/mdo/OpenFOAM && \
tar -xvf /mnt/e/Downloads/WSL/OpenFOAM/OpenFOAM-v1812-AD-1.2.9.tar.gz && mv OpenFOAM-v1812-AD-* OpenFOAM-v1812-ADF && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812-ADF && \
sed -i 's/WM_PROJECT_VERSION=v1812-AD/WM_PROJECT_VERSION=v1812-ADF/g' etc/bashrc && \
sed -i 's/$HOME/$HOME\/mdo/g' etc/bashrc && \
. $HOME/mdo/loadmdo.sh && \
source etc/bashrc && \
export WM_NCOMPPROCS=60 && \
./Allwmake 2> warningLog.txt
rm warningLog.txt && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812/platforms/*/lib && \
ln -s ../../../../OpenFOAM-v1812-ADF/platforms/*/lib/*.so . && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/dummy && \
ln -s ../../../../../OpenFOAM-v1812-ADF/platforms/*/lib/dummy/*.so . && \
cd $HOME/mdo/OpenFOAM/OpenFOAM-v1812/platforms/*/lib/openmpi-system && \
ln -s ../../../../../OpenFOAM-v1812-ADF/platforms/*/lib/openmpi-system/*.so .
unset WM_CODI_AD_MODE && \
. $HOME/mdo/loadmdo.sh && \

cd $HOME/mdo/repos &&  mkdir pyofm && cd pyofm
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/pyofm-1.2.1.tar.gz --strip-components=1 && \
. $HOME/mdo/loadmdo.sh && \
make && pip install . && \

cd $HOME/mdo/repos && mkdir dafoam && cd dafoam
tar -xvf /mnt/e/Downloads/WSL/new\ dafoam/dafoam-3.0.4.tar.gz --strip-components=1 && \
. $HOME/mdo/loadmdo.sh && \
export DAFOAM_NO_WARNINGS=1 && \
./Allmake && \
source $HOME/mdo/OpenFOAM/OpenFOAM-v1812-ADR/etc/bashrc && \
./Allclean && ./Allmake && \
source $HOME/mdo/OpenFOAM/OpenFOAM-v1812-ADF/etc/bashrc && \
./Allclean && ./Allmake && \
pip install .






