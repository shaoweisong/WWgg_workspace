#!/bin/bash
echo "Starting job on " `date`
echo "Running on: `uname -a`"
echo "System software: `cat /etc/redhat-release`"
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
#install parquet_to_root
cp /eos/user/s/shsong/pkgs_condor/*.tar.gz .
cp ${3}create_inputfiles_flashggfinalfit_cat12_pytorch.py . 
mkdir PBDT_HH_FHSL_combine_${5}
mkdir PBDT_HH_FHSL_combine_${5}/pkgs_condor
tar -xzvf bayesian-optimization-1.4.3.tar.gz -C ./PBDT_HH_FHSL_combine_${5}/pkgs_condor/
tar -xzvf parquet_to_root-0.3.0.tar.gz -C ./PBDT_HH_FHSL_combine_${5}/pkgs_condor/
cd PBDT_HH_FHSL_combine_${5}/pkgs_condor/bayesian-optimization-1.4.3
python setup.py install 
cd -
cd PBDT_HH_FHSL_combine_${5}/pkgs_condor/parquet_to_root-0.3.0
python setup.py install
cd -
mkdir PBDT_HH_FHSL_combine_${5}/flashgginput
rm *.root
echo "========================================="
echo "cat create_inputfiles_flashggfinalfit_cat12_pytorch.py"
echo "..."
cat create_inputfiles_flashggfinalfit_cat12_pytorch.py
echo "..."
echo "========================================="
python create_inputfiles_flashggfinalfit_cat12_pytorch.py --inputFHFiles ${1} --inputBKGFiles ${2} --year ${5}
echo "====> List root files : " 
ls ./PBDT_HH_FHSL_combine_${5}/flashgginput/MX*/*.root
echo "====> copying workspace input directory to stores area..." 
if ls ./PBDT_HH_FHSL_combine_${5}/flashgginput/MX*/*.root 1> /dev/null 2>&1; then
    echo "File *.root exists. Copy this."
    echo "cp *.root ${4}"
    cp -r ./PBDT_HH_FHSL_combine_${5}/flashgginput/MX*/ ${4}
fi
cd ${_CONDOR_SCRATCH_DIR}
rm create_inputfiles_flashggfinalfit_cat12_pytorch.py
rm -rf PBDT_HH_FHSL_combine_${5}
