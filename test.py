#/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
import sys
sys.path.append("/eos/user/s/shsong/pkgs_condor/parquet_to_root-0.3.0")
from parquet_to_root import parquet_to_root
# parquet_to_root("/afs/cern.ch/user/s/shsong/WWggDNN/bdt/Data_2017_combineFHSL_cat12test_MX3000_MH12520.parquet","/afs/cern.ch/user/s/shsong/WWggDNN/bdt/M3000_20data/Data_2017_combineFHSL_cat12highpurity_MX3000_MH125.root",treename="Data_13TeV_combineFHSL_cat12highpurity",verbose=False)
parquet_to_root("/eos/user/s/shsong/MX3000_MH125/Data_2017_combineFHSL_cat12highpurity_MX3000_MH125.parquet","/eos/user/s/shsong/MX3000_MH125/Data_2017_combineFHSL_cat12highpurity_MX3000_MH125.root",treename="Data_13TeV_combineFHSL_cat12highpurity",verbose=False)
parquet_to_root('/eos/user/s/shsong/MX3000_MH125/CombineFHSL_MX3000_MH125_2017_combineFHSL_cat12highpurity.parquet','/eos/user/s/shsong/MX3000_MH125/CombineFHSL_MX3000_MH125_2017_combineFHSL_cat12highpurity.root',treename="gghh_125_13TeV_combineFHSL_cat12highpurity",verbose=False)