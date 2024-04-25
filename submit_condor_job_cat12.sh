#!/bin/bash
echo "Starting job on " `date`
echo "Running on: `uname -a`"
echo "System software: `cat /etc/redhat-release`"
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
#install parquet_to_root

cp ${3}create_inputfiles_flashggfinalfit_cat12_loop.py . 
mkdir PBDT_HH_FHSL_combine_2017
mkdir PBDT_HH_FHSL_combine_2017/flashgginput
rm *.root
echo "========================================="
echo "cat create_inputfiles_flashggfinalfit_cat12_loop.py"
echo "..."
cat create_inputfiles_flashggfinalfit_cat12_loop.py
echo "..."
echo "========================================="
python create_inputfiles_flashggfinalfit_cat12_loop.py --inputFHFiles ${1} --inputBKGFiles ${2}
echo "====> List root files : " 
ls ./PBDT_HH_FHSL_combine_2017/flashgginput/MX*/*.root
echo "====> copying workspace input directory to stores area..." 
if ls ./PBDT_HH_FHSL_combine_2017/flashgginput/MX*/*.root 1> /dev/null 2>&1; then
    echo "File *.root exists. Copy this."
    echo "cp *.root ${4}"
    cp -r ./PBDT_HH_FHSL_combine_2017/flashgginput/MX*/ ${4}
fi
cd ${_CONDOR_SCRATCH_DIR}
rm create_inputfiles_flashggfinalfit_cat12_loop.py
rm -rf PBDT_HH_FHSL_combine_2017
