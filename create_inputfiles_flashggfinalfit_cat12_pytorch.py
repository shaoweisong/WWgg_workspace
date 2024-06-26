#/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
import awkward as ak
import sys
debug=False
local=False
import numpy as np
import os
import vector
vector.register_awkward()
import mplhep as hep
hep.style.use(hep.style.CMS)
import re
import pandas as pd
import glob
from scipy.optimize import minimize
import argparse
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
# 自定义函数，将参数字符串转换为列表
def str_to_list(arg):
    return ast.literal_eval(arg)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--is_HH', type=str, default="False", help='is HH')
parser.add_argument('--inputFHFiles',type=str_to_list, help='inputFHFiles List')
parser.add_argument('--inputBKGFiles',type=str_to_list, help='input pp and dd Files List')
parser.add_argument('--data',type=str,default="/eos/user/s/shsong/HiggsDNA/UL18data/merged_nominal.parquet", help='data file')
parser.add_argument('--model',type=str,default="/eos/user/z/zhenxuan/PNN_wwgg/PNN_YH_combined/boosted_FHSL/data/simple_DNN_real_epoch_30_mx500_3000_random_mass_reweightforsignal_lesslayers_bkgcp2times/model.pth", help='model file') #old model for 3TeV 220fb limit
# parser.add_argument('--model',type=str,default="/eos/user/z/zhenxuan/PNN_wwgg/PNN_YH_combined/boosted_FHSL/data/simple_DNN_real_epoch_200_mx500_3000_random_mass_reweightforsignal_lesslayers_bkgcp2times_fixedsoftmax_fixedreweight_reweightsignalwithhighmasshighweight_nobbgg_adddiphotonptreweight_largebatchsize_addweight_decay_addbatchnorm_highlr/model.pth", help='model file') old model with not good performance
# parser.add_argument('--model',type=str,default="/eos/user/z/zhenxuan/PNN_wwgg/PNN_YH_combined/boosted_FHSL/data/simple_DNN_train_boosted_1/model.pth", help='model file') new model zhenxuan change weight 3TeV 175 
parser.add_argument('--scalar',type=str,default="/eos/user/z/zhenxuan/PNN_wwgg/PNN_YH_combined/boosted_FHSL/data/simple_DNN_real_epoch_30_mx500_3000_random_mass_reweightforsignal_lesslayers_bkgcp2times/scaler_params.json", help='scalar file')
parser.add_argument('--year',type=str,default="2017", help='year')
args = parser.parse_args()
datapath = args.data
model_path = args.model
scalar_path = args.scalar
year = args.year
print(year)
if local:
    sys.path.append("/eos/user/s/shsong/pkgs_condor/parquet_to_root-0.3.0")
    sys.path.append("/eos/user/s/shsong/pkgs_condor/bayesian-optimization-1.4.3")
else:
    sys.path.append("./PBDT_HH_FHSL_combine_"+year+"/pkgs_condor/parquet_to_root-0.3.0")
    sys.path.append("./PBDT_HH_FHSL_combine_"+year+"/pkgs_condor/bayesian-optimization-1.4.3")
if debug:
    print(sys.path)
from parquet_to_root import parquet_to_root
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
# class MultiClassDNN_model(nn.Module): new model zhenxuan change weight 3TeV 175 
#     def __init__(self, input_size, output_size):
#         super(MultiClassDNN_model, self).__init__()
#         self.fc1 = nn.Sequential(nn.Linear(input_size, 128), nn.LeakyReLU(0.01))
#         self.fc2 = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU(0.01))
#         self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.01))
#         self.dropout = nn.Dropout(0.5)
#         self.fc4 = nn.Linear(64, output_size)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.dropout(out)
#         out = self.fc2(out)
#         out = self.dropout(out)
#         out = self.fc3(out)
#         out = self.dropout(out)
#         out = self.fc4(out)
#         return out
class MultiClassDNN_model(nn.Module): 
    def __init__(self, input_size, output_size):
        super(MultiClassDNN_model, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, 128), nn.LeakyReLU(0.01))
        self.fc2 = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(0.01))
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.dropout(out)

        out = self.fc2(out)

        out = self.dropout(out)

        out = self.fc3(out)

        return F.softmax(out, dim=1)
# class MultiClassDNN_model(nn.Module): performance not good
#     def __init__(self, input_size, output_size):
#         super(MultiClassDNN_model, self).__init__()
#         self.fc1 = nn.Sequential(
#             nn.Linear(input_size, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(0.5)
#         )
#         self.fc2 = nn.Sequential(
#             nn.Linear(128, 128),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(0.5)
#         )
#         self.fc3 = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(0.01),
#             nn.Dropout(0.5)
#         )
#         self.fc4 = nn.Linear(64, output_size)

    # def forward(self, x):
    #     out = self.fc1(x)
    #     out = self.fc2(out)
    #     out = self.fc3(out)
    #     out = self.fc4(out)
    #     return out

def get_fwhm(hist_sig, bin_sig):
    peak_index = np.argmax(hist_sig)
    peak_mass = 0.5 * (bin_sig[peak_index] + bin_sig[peak_index + 1])
    half_max_height = 0.5 * hist_sig[peak_index]
    left_index = np.where(hist_sig[:peak_index] <= half_max_height)[0][-1]
    right_index = np.where(hist_sig[peak_index:] <= half_max_height)[0][0] + peak_index
    fwhm = bin_sig[right_index] - bin_sig[left_index]
    return fwhm
def target_function(bo1, bo2,s, b, bkg):
    if debug:
        print("bo1: ", bo1)
        print("bo2: ", bo2)
    events_sig1 = s[(s['PNN_score'] >= 1-bo1) & (s['PNN_score'] <=1)]
    events_datasideband_1 = b[(b['PNN_score'] >= 1-bo1) & (b['PNN_score'] <=1)]
    events_bkg1 = bkg[(bkg['PNN_score'] >= 1-bo1) & (bkg['PNN_score'] <=1)]
    events_sig2 = s[(s['PNN_score'] >= 1-bo2-bo1) & (s['PNN_score'] <1-bo1)]
    events_datasideband_2 = b[(b['PNN_score'] >= 1-bo2-bo1) & (b['PNN_score'] <1-bo1)]
    events_bkg2 = bkg[(bkg['PNN_score'] >= 1-bo2-bo1) & (bkg['PNN_score'] <1-bo1)]

    if len(events_datasideband_1[((events_datasideband_1.Diphoton_mass>135)|(events_datasideband_1.Diphoton_mass<115))]) < 5:
        significance=0
        return 0
    elif len(events_datasideband_2[((events_datasideband_2.Diphoton_mass>135)|(events_datasideband_2.Diphoton_mass<115))]) < 5:
        significance=0
        return 0
    if debug:
        print("len(events_datasideband_1): ", len(events_datasideband_1))
        print("len(events_datasideband_2): ", len(events_datasideband_2))
    # 1
    # check signal mass distribution
    hist_sig1, bin_sig1 = np.histogram(np.array(events_sig1['Diphoton_mass']), bins=100, range=(110,150), weights=np.array(events_sig1['weight_central']))
    if debug:
        print("hist_sig1: ", hist_sig1)
        print("bin_sig1: ", bin_sig1)
    # get signal mass FWHM
    fwhm1 = get_fwhm(hist_sig1, bin_sig1)
    # get significance s/sqrt(s+b)
    s1 = np.sum(events_sig1['weight_central'][(events_sig1['Diphoton_mass'] > 115) & (events_sig1['Diphoton_mass'] < 135)])
    bkg1_weight = (events_bkg1['weight_central'])[(events_bkg1['Diphoton_mass'] > 115) & (events_bkg1['Diphoton_mass'] < 135)]
    b1 = np.sum(bkg1_weight) 
    if b1==0:
        return 0
    d1 = np.sum(events_datasideband_1['weight_central'][(events_datasideband_1['Diphoton_mass'] > 135) | (events_datasideband_1['Diphoton_mass'] < 115)])
    if d1 >10:
        return 0
    significance1 = s1 / np.sqrt(b1)
    # significance1 = s1 / np.sqrt(b1)
    # 2
    # check signal mass distribution
    hist_sig2, bin_sig2 = np.histogram(np.array(events_sig2['Diphoton_mass']), bins=100, range=(110,150), weights=np.array(events_sig2['weight_central']))
    # check signal mass FWHM
    fwhm2 = get_fwhm(hist_sig2, bin_sig2)
    # get significance s/sqrt(s+b)
    s2 = np.sum(events_sig2['weight_central'][(events_sig2['Diphoton_mass'] > 115) & (events_sig2['Diphoton_mass'] < 135)])
    bkg2_weight = (events_bkg2['weight_central'])[(events_bkg2['Diphoton_mass'] > 115) & (events_bkg2['Diphoton_mass'] < 135)]
    b2 = np.sum(bkg2_weight)
    if b2==0:
        return 0
    d2 = np.sum(events_datasideband_2['weight_central'][(events_datasideband_2['Diphoton_mass'] > 135) | (events_datasideband_2['Diphoton_mass'] < 115)])
    # significance2 = s2 / np.sqrt(b2)
    significance2 = s2 / np.sqrt(b2)

    significance = np.sqrt(significance1**2 + significance2**2 )
    fwhm = np.sqrt(fwhm1**2 + fwhm2**2)
    return significance

#/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
def get_jet_variables(events):
    events['fatjet_1_pt'] = events['fatjet_1_pt']
    events['fatjet_2_pt'] = events['fatjet_2_pt']
    events['fatjet_1_eta'] = events['fatjet_1_eta']
    events['fatjet_2_eta'] = events['fatjet_2_eta']
    events['fatjet_1_phi'] = events['fatjet_1_phi']
    events['fatjet_2_phi'] = events['fatjet_2_phi']
    events['fatjet_1_msoftdrop'] = events['fatjet_1_msoftdrop']
    events['fatjet_2_msoftdrop'] = events['fatjet_2_msoftdrop']
    # get fatjet and diphoton dR
    fatjet_1_4D = vector.obj(pt=events.fatjet_1_pt, eta=events.fatjet_1_eta, phi=events.fatjet_1_phi, mass=events.fatjet_1_msoftdrop)
    fatjet_2_4D = vector.obj(pt=events.fatjet_2_pt, eta=events.fatjet_2_eta, phi=events.fatjet_2_phi, mass=events.fatjet_2_msoftdrop)
    fatjet_3_4D = vector.obj(pt=events.fatjet_3_pt, eta=events.fatjet_3_eta, phi=events.fatjet_3_phi, mass=events.fatjet_3_msoftdrop)
    diphoton_4D = vector.obj(pt=events.Diphoton_pt, eta=events.Diphoton_eta, phi=events.Diphoton_phi, mass=events.Diphoton_mass)
    fatjet_1_diphoton_dR = fatjet_1_4D.deltaR(diphoton_4D)
    fatjet_2_diphoton_dR = fatjet_2_4D.deltaR(diphoton_4D)
    events['fatjet_1_diphoton_dR'] = np.where(events.fatjet_1_pt>0, fatjet_1_diphoton_dR, -999)
    events['fatjet_2_diphoton_dR'] = np.where(events.fatjet_2_pt>0, fatjet_2_diphoton_dR, -999)
    leadphoton=vector.obj(pt=events.LeadPhoton_pt,eta=events.LeadPhoton_eta,phi=events.LeadPhoton_phi,mass=events.LeadPhoton_mass)
    subleadphoton=vector.obj(pt=events.SubleadPhoton_pt,eta=events.SubleadPhoton_eta,phi=events.SubleadPhoton_phi,mass=events.SubleadPhoton_mass)
    # dR with photon and fatjet
    fatjet_1_leadphoton_dR = fatjet_1_4D.deltaR(leadphoton)
    events['fatjet_1_leadphoton_dR'] = np.where(events.fatjet_1_pt>0, fatjet_1_leadphoton_dR, -999)
    fatjet_1_subleadphoton_dR = fatjet_1_4D.deltaR(subleadphoton)
    events['fatjet_1_subleadphoton_dR'] = np.where(events.fatjet_1_pt>0, fatjet_1_subleadphoton_dR, -999)
    fatjet_2_leadphoton_dR = fatjet_2_4D.deltaR(leadphoton)
    events['fatjet_2_leadphoton_dR'] = np.where(events.fatjet_2_pt>0, fatjet_2_leadphoton_dR, -999)
    fatjet_2_subleadphoton_dR = fatjet_2_4D.deltaR(subleadphoton)
    events['fatjet_2_subleadphoton_dR'] = np.where(events.fatjet_2_pt>0, fatjet_2_subleadphoton_dR, -999)
    # get fatjet1 and fatjet2 dR
    fatjet_1_2_dR = fatjet_1_4D.deltaR(fatjet_2_4D)
    events['fatjet_1_2_dR'] = np.where(np.logical_and(events.fatjet_1_pt>0, events.fatjet_2_pt>0), fatjet_1_2_dR, -999)
    # get the maximum fatjets mass with the combination of 2 fatjets in 3 fatjets
    fatjet_12_4D = fatjet_1_4D+fatjet_2_4D
    fatjet_13_4D = fatjet_1_4D+fatjet_3_4D
    fatjet_23_4D = fatjet_2_4D+fatjet_3_4D
    fatjet_12_msoftdrop = np.where(((fatjet_1_4D.pt >0) & (fatjet_2_4D.pt >0)), fatjet_12_4D.mass, -999)
    fatjet_13_msoftdrop = np.where(((fatjet_1_4D.pt >0) & (fatjet_3_4D.pt >0)), fatjet_13_4D.mass, -999)
    fatjet_23_msoftdrop = np.where(((fatjet_2_4D.pt >0) & (fatjet_3_4D.pt >0)), fatjet_23_4D.mass, -999)
    max_fatjets_mass = np.maximum(fatjet_12_msoftdrop, np.maximum(fatjet_13_msoftdrop, fatjet_23_msoftdrop))
    events['max_fatjets_mass'] = max_fatjets_mass
    # get max WvsQCD score with three fatjets
    events['fatjet_1_WvsQCDMD'] = events['fatjet_1_WvsQCDMD']
    events['fatjet_2_WvsQCDMD'] = events['fatjet_2_WvsQCDMD']
    # get max H4qvsQCD score with three fatjets
    events['fatjet_1_Hqqqq_vsQCDTop'] = events['fatjet_1_Hqqqq_vsQCDTop']
    events['fatjet_2_Hqqqq_vsQCDTop'] = events['fatjet_2_Hqqqq_vsQCDTop']
    Hscore = ak.concatenate([ak.unflatten(events['fatjet_1_Hqqqq_vsQCDTop'], counts=1), ak.unflatten(events['fatjet_2_Hqqqq_vsQCDTop'], counts=1)], axis=1)
    events['max_fatjet_Hqqqq_vsQCDTop'] = Hscore[ak.argsort(Hscore, axis=-1, ascending=False)][:,0]
    # get XbbvsQCD
    events['fatjet_1_XbbvsQCDMD'] = (events['fatjet_1_particleNetMD_Xbb']) / (events['fatjet_1_particleNetMD_Xbb'] + events['fatjet_1_particleNetMD_QCD'])
    events['fatjet_2_XbbvsQCDMD'] = (events['fatjet_2_particleNetMD_Xbb']) / (events['fatjet_2_particleNetMD_Xbb'] + events['fatjet_2_particleNetMD_QCD'])
    # add number of good AK4 jets
    events['nGoodAK4jets'] = events['nGoodAK4jets']
    # add number of good AK8 jets
    events['nGoodAK8jets'] = events['nGoodAK8jets']
    # get 4 ak4 jets 4D info
    events['jet_1_pt'] = events['jet_1_pt']
    events['jet_2_pt'] = events['jet_2_pt']
    events['jet_3_pt'] = events['jet_3_pt']
    events['jet_1_eta'] = events['jet_1_eta']
    events['jet_2_eta'] = events['jet_2_eta']
    events['jet_3_eta'] = events['jet_3_eta']
    events['jet_1_phi'] = events['jet_1_phi']
    events['jet_2_phi'] = events['jet_2_phi']
    events['jet_3_phi'] = events['jet_3_phi']
    events['jet_1_mass'] = events['jet_1_mass']
    events['jet_2_mass'] = events['jet_2_mass']
    events['jet_3_mass'] = events['jet_3_mass']
    

    return events
def get_leptons_variables(events):
    events['nGoodisoleptons'] = events['nGoodisoleptons']
    events['nGoodnonisoleptons'] = events['nGoodnonisoleptons']
    events['electrons_all_1_pt'] = events['electrons_all_1_pt']
    events['electrons_all_2_pt'] = events['electrons_all_2_pt']
    events['electrons_all_1_eta'] = events['electrons_all_1_eta']
    events['electrons_all_2_eta'] = events['electrons_all_2_eta']
    events['electrons_all_1_phi'] = events['electrons_all_1_phi']
    events['electrons_all_2_phi'] = events['electrons_all_2_phi']
    events['electrons_all_1_mass'] = events['electrons_all_1_mass']
    events['electrons_all_2_mass'] = events['electrons_all_2_mass']
    events['muons_all_1_pt'] = events['muons_all_1_pt']
    events['muons_all_2_pt'] = events['muons_all_2_pt']
    events['muons_all_1_eta'] = events['muons_all_1_eta']
    events['muons_all_2_eta'] = events['muons_all_2_eta']
    events['muons_all_1_phi'] = events['muons_all_1_phi']
    events['muons_all_2_phi'] = events['muons_all_2_phi']
    events['muons_all_1_mass'] = events['muons_all_1_mass']
    events['muons_all_2_mass'] = events['muons_all_2_mass']
    
    return events
def get_met_variables(events):
    events['PuppiMET_pt'] = events['PuppiMET_pt']
    events['PuppiMET_sumEt'] = events['PuppiMET_sumEt']
    return events
input_features = ['Diphoton_pt','Diphoton_eta','Diphoton_phi','Diphoton_dR','fatjet_1_pt','fatjet_2_pt','fatjet_1_eta','fatjet_2_eta','fatjet_1_phi','fatjet_2_phi','fatjet_1_msoftdrop','fatjet_2_msoftdrop','fatjet_1_diphoton_dR','fatjet_2_diphoton_dR','fatjet_1_leadphoton_dR','fatjet_1_subleadphoton_dR','fatjet_2_leadphoton_dR','fatjet_2_subleadphoton_dR','fatjet_1_2_dR','max_fatjets_mass','fatjet_1_WvsQCDMD','fatjet_2_WvsQCDMD','fatjet_1_Hqqqq_vsQCDTop','fatjet_2_Hqqqq_vsQCDTop','nGoodAK4jets','nGoodAK8jets','jet_1_pt','jet_2_pt','jet_3_pt','jet_1_eta','jet_2_eta','jet_3_eta','jet_1_phi','jet_2_phi','jet_3_phi','jet_1_mass','jet_2_mass','jet_3_mass','nGoodisoleptons','nGoodnonisoleptons','electrons_all_1_pt','electrons_all_2_pt','electrons_all_1_eta','electrons_all_2_eta','electrons_all_1_phi','electrons_all_2_phi','electrons_all_1_mass','electrons_all_2_mass','muons_all_1_pt','muons_all_2_pt','muons_all_1_eta','muons_all_2_eta','muons_all_1_phi','muons_all_2_phi','muons_all_1_mass','muons_all_2_mass','PuppiMET_pt','PuppiMET_sumEt','fatjet_1_XbbvsQCDMD','fatjet_2_XbbvsQCDMD','mx','my']
# input_features =['Diphoton_pt','Diphoton_eta','Diphoton_phi','Diphoton_dR','fatjet_1_pt','fatjet_2_pt','fatjet_1_eta','fatjet_2_eta','fatjet_1_phi','fatjet_2_phi','fatjet_1_diphoton_dR','fatjet_2_diphoton_dR','fatjet_1_leadphoton_dR','fatjet_1_subleadphoton_dR','fatjet_2_leadphoton_dR','fatjet_2_subleadphoton_dR','fatjet_1_2_dR','max_fatjets_mass','fatjet_1_WvsQCDMD','fatjet_2_WvsQCDMD','max_fatjet_Hqqqq_vsQCDTop','nGoodAK4jets','nGoodAK8jets','jet_1_pt','jet_2_pt','jet_3_pt','jet_1_eta','jet_2_eta','jet_3_eta','jet_1_phi','jet_2_phi','jet_3_phi','jet_1_mass','jet_2_mass','jet_3_mass','nGoodisoleptons','nGoodnonisoleptons','electrons_all_1_pt','electrons_all_2_pt','electrons_all_1_eta','electrons_all_2_eta','electrons_all_1_phi','electrons_all_2_phi','electrons_all_1_mass','electrons_all_2_mass','muons_all_1_pt','muons_all_2_pt','muons_all_1_eta','muons_all_2_eta','muons_all_1_phi','muons_all_2_phi','muons_all_1_mass','muons_all_2_mass','PuppiMET_pt','PuppiMET_sumEt','fatjet_1_XbbvsQCDMD','mx'] new model zhenxuan change weight 3TeV 175
other_vars  = ['weight_central','Diphoton_mass','Diphoton_minID_modified','Diphoton_maxID_modified']

def get_sig_events_forApply(filename,mx,my):
    events = ak.from_parquet(filename)
    category_cut = ((events["category"]==2) | (events["category"]==1))
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    events = get_jet_variables(events)
    events = get_leptons_variables(events)
    events = get_met_variables(events)
    events['type'] = "sig"
    return events
def get_bkg_events_forApply(filename, mx, my):
    events = ak.from_parquet(filename)
    category_cut = ((events["category"]==2) | (events["category"]==1))
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    # add type
    if "DatadrivenQCD" in filename:
        events["type"] = "dd"
    elif "DiPhotonJetsBox" in filename:
        events["type"] = "pp"
    events = get_jet_variables(events)
    events = get_leptons_variables(events)
    events = get_met_variables(events)
    return events
def get_data_events_forApply(filename,mx, my):
    events = ak.from_parquet(filename)
    category_cut = ((events["category"]==2) | (events["category"]==1))
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    # specify mx and my for each data for each signal mass point
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    events['year'] = np.ones(len(events))*int(year)
    events = get_jet_variables(events)
    events = get_leptons_variables(events)
    events = get_met_variables(events)
    return events     
def get_dd_events_forApply(filename,mx, my):
    events = ak.from_parquet(filename)
    category_cut = ((events["category"]==2) | (events["category"]==1))
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    # specify mx and my for each data for each signal mass point
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    events = get_jet_variables(events)
    events = get_leptons_variables(events)
    events = get_met_variables(events)
    return events     
print('start to get input features')
# get all all_files directory name inside the folder_path
FHsignal_path_list = []
SLsignal_path_list = []
ZZggsignal_path_list = []
BBGGsignal_path_list = []
signal_output_name=[]
bbgg_output_name=[]
zzgg_output_name=[]
list_of_files = args.inputFHFiles #merged_nominal.parquet should be the first one
for FHfile in list_of_files:
    FHsignal_path_list.append(FHfile)
    SLfile=(FHfile.replace("2G4Q","2G2Q1L1Nu")).replace("HHFH","HHSL")
    bbggfile=(FHfile.replace("2G2WTo2G4Q","2B2G")).replace("HHFH","HHbbgg")
    zzggfile=(FHfile.replace("2G2W","2G2Z")).replace("HHFH","HHZZgg")
    SLsignal_path_list.append(SLfile)
    ZZggsignal_path_list.append(zzggfile)
    BBGGsignal_path_list.append(bbggfile)
    dir_name="CombineFHSL_MX" + FHfile.split("M-")[1].split("_")[0] + "_MH125_cat12_"+(FHfile.split("/")[-1]).split(".")[0]
    # dir_name = "CombineFHSL_MX1100_MH125_cat12_merged_FJER_down"
    signal_output_name.append(dir_name)
    bbgg_output_name.append(dir_name.replace("CombineFHSL","BBGG"))
    zzgg_output_name.append(dir_name.replace("CombineFHSL","ZZGG"))
    
signal_samples = {'FHpath':FHsignal_path_list, 'sig_output_name':signal_output_name, 'SLpath':SLsignal_path_list, 'ZZggpath':ZZggsignal_path_list, 'BBGGpath':BBGGsignal_path_list, 'bbgg_output_name':bbgg_output_name, 'zzgg_output_name':zzgg_output_name}
boundary1=[]
boundary2=[]
bkgfiles = args.inputBKGFiles

#-------------------------------------Get the boundary-------------------------------------#
#firstly use the merged_nominal.parquet to get the boundary
mx = signal_samples['sig_output_name'][0].split('_')[1].split('X')[1]
my = signal_samples['sig_output_name'][0].split('_')[2].split('H')[1]
events_sigFH = get_sig_events_forApply(signal_samples['FHpath'][0],mx,my)
events_sigSL = get_sig_events_forApply(signal_samples['SLpath'][0],mx,my)
events_bbgg = get_sig_events_forApply(signal_samples['BBGGpath'][0],mx,my)
events_zzgg = get_sig_events_forApply(signal_samples['ZZggpath'][0],mx,my)
events_sig = ak.concatenate([events_sigFH,events_sigSL])
events_data = get_data_events_forApply(datapath,mx,my)
events_pp_cat1= get_data_events_forApply(bkgfiles[0],mx,my)
events_pp_cat2= get_data_events_forApply(bkgfiles[1],mx,my)
events_dd_cat1= get_dd_events_forApply(bkgfiles[2],mx,my)
events_dd_cat2= get_dd_events_forApply(bkgfiles[3],mx,my)
event_bkgmc=ak.concatenate([events_pp_cat1,events_pp_cat2,events_dd_cat1,events_dd_cat2])
import json
#load model to get dnnscore
print('load the model')

# model = MultiClassDNN_model(len(input_features), 5) new model zhenxuan change weight 3TeV 175
model = MultiClassDNN_model(len(input_features), 4) #old model 3TeV 220
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()
with open(scalar_path, 'r') as f:
    loaded_params = json.load(f)
    loaded_scaler = StandardScaler()
    loaded_scaler.mean_ = np.array(loaded_params['mean'])
    loaded_scaler.scale_ = np.array(loaded_params['scale'])
print('successfully load the model and scalar')
def model_predict(event, model, loaded_scaler, input_features):
    df = ak.to_pandas(event[input_features + ['weight_central','Diphoton_mass','Diphoton_minID_modified','Diphoton_maxID_modified']])
    X_test = loaded_scaler.transform(df[input_features])
    X_test = torch.tensor(X_test).float()
    proba = model(X_test)
    # proba = F.softmax(proba, dim=1) only for new models
    proba = proba.detach().numpy()
    pnn_score = (proba[:,1] + proba[:,2] + proba[:,3]) / (proba[:,0] + proba[:,1] + proba[:,2] + proba[:,3])# old model 3TeV 220
    # pnn_score = (proba[:,4] + proba[:,2] + proba[:,3]) / (proba[:,0] + proba[:,1] + proba[:,2] + proba[:,3] + proba[:,4]) new model zhenxuan change weight 3TeV 175
    return pnn_score
# for sig, data, bkgmc, bbgg and zzgg
# evaluate the model
PNN_score = model_predict(events_sig, model, loaded_scaler, input_features)
events_sig['PNN_score'] = PNN_score

PNN_score = model_predict(events_data, model, loaded_scaler, input_features)
events_data['PNN_score'] = PNN_score

PNN_score = model_predict(event_bkgmc, model, loaded_scaler, input_features)
event_bkgmc['PNN_score'] = PNN_score

PNN_score = model_predict(events_bbgg, model, loaded_scaler, input_features)
events_bbgg['PNN_score'] = PNN_score

PNN_score = model_predict(events_zzgg, model, loaded_scaler, input_features)
events_zzgg['PNN_score'] = PNN_score
print('get all the PBDT score for merged nominal.parquet')
# #minmaxscalar
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(-10, 10))
# sigpnn_transformed = scaler.fit_transform(events_sig['PNN_score'].reshape(-1, 1))
# bkgpnn_transformed = scaler.fit_transform(event_bkgmc['PNN_score'].reshape(-1, 1))
# data_pnn_transformed = scaler.fit_transform(events_data['PNN_score'].reshape(-1, 1))
# plot the PNN score
import mplhep as hep
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
hep.style.use("CMS")
plt.hist(events_sig['PNN_score'],weights = 30*events_sig.weight_central ,range=(0,1),bins=100, histtype='step', label='30*signal',color='#FF0000')
plt.hist(event_bkgmc['PNN_score'],weights = event_bkgmc.weight_central ,range=(0,1),bins=100, histtype='stepfilled', label='pp+dd',color='#00BFFF')
hist, bins = np.histogram(events_data[(events_data.Diphoton_mass>135)|(events_data.Diphoton_mass<115)]['PNN_score'],bins=100)
non_zero_bins = hist > 0
bin_centers = 0.5 * (bins[:-1] + bins[1:])
plt.scatter(bin_centers[non_zero_bins], hist[non_zero_bins], marker='o', color='black', label='data')
plt.xlabel('PNN Score')
plt.ylabel('Events')
plt.yscale('log')
#y axis unit small
plt.legend()
print('start to get the best threshold by using bayesian optimization')
# 贝叶斯优化
# 输出最佳的PNN score阈值

# pbounds = {
#         'bo1': (0.00000001, 0.05), 
#         'bo2': (0.00000001, 0.45), 
#         } old model 3TeV 220

pbounds = {
        'bo1': (0.0001, 0.1), 
        'bo2': (0.0001, 0.7), 
        }
debug=False
optimizer = BayesianOptimization(
    f=target_function,
    pbounds=pbounds,
    random_state=1,
    verbose=4
)
utility = UtilityFunction(kind="ei", kappa=2.576, xi=0.0)
next_point=optimizer.suggest(utility)
target=target_function(next_point['bo1'],next_point['bo2'],events_sig,events_data,event_bkgmc)
print("looping BayesianOptimization")
for _ in range(300):
    next_point = optimizer.suggest(utility)
    target = target_function(next_point['bo1'],next_point['bo2'],events_sig,events_data,event_bkgmc)
    optimizer.register(params=next_point, target=target)
boundaries=optimizer.max['params']
cut1=1-boundaries['bo1']
cut2=1-boundaries['bo1']-boundaries['bo2']
boundary1.append(cut1)
boundary2.append(cut2)
print("best significance",optimizer.max['target'])
print('high purity cut:',cut1)
print('low purity cut:',cut2)
highpurity_sigeff=ak.sum(events_sig["weight_central"][(events_sig.PNN_score>cut1)&(events_sig.PNN_score<=1)])/ak.sum(events_sig["weight_central"])
lowpurity_sigeff=ak.sum(events_sig["weight_central"][(events_sig.PNN_score>cut2)&(events_sig.PNN_score<=cut1)])/ak.sum(events_sig["weight_central"])
highpurity_sidebandnum=len(events_data[((events_data.Diphoton_mass>135)|(events_data.Diphoton_mass<115))&((events_data.PNN_score > cut1) & (events_data.PNN_score <= 1))])
lowpurity_sidebandnum=len(events_data[((events_data.Diphoton_mass>135)|(events_data.Diphoton_mass<115))&((events_data.PNN_score > cut2) & (events_data.PNN_score <= cut1))])
print('high purity signal efficiency:',highpurity_sigeff)
print('low purity signal efficiency:',lowpurity_sigeff)
print('high purity sideband number:',highpurity_sidebandnum)
print('low purity sideband number:',lowpurity_sidebandnum)
ak.to_parquet(events_data, './Data_'+year+'_combineFHSL_cat12test_MX3000_MH125.parquet')
ak.to_parquet(events_sig, './Signal_'+year+'_combineFHSL_cat12test_MX3000_MH125.parquet')
# #add PNSF to merged nominal.parquet
def add_HtaggerSF(event,mass):
    mass_list=[500,550,600,650,700,750,800,850,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000]
    # mass_list=[1000]
    index=mass_list.index(mass)
    HvsQCD_SF=[0.993610,1.083977,0.923107,1.134969,1.097200,0.973850,0.970535,1.039317,1.008370,0.993610,1.083977,0.923107,1.134969,1.097200,0.973850,0.970535,1.039317,1.008370,0.962904,0.969104,1.002570,0.941431,1.025998,1.004931,0.967810]
    HvsQCD_SFup=np.array([0.993610,1.083977,0.923107,1.134969,1.097200,0.973850,0.970535,1.039317,1.008370,0.993610,1.083977,0.923107,1.134969,1.097200,0.973850,0.970535,1.039317,1.008370,0.962904,0.969104,1.002570,0.941431,1.025998,1.004931,0.967810])*1.22
    HvsQCD_SFdown=np.array([0.993610,1.083977,0.923107,1.134969,1.097200,0.973850,0.970535,1.039317,1.008370,0.993610,1.083977,0.923107,1.134969,1.097200,0.973850,0.970535,1.039317,1.008370,0.962904,0.969104,1.002570,0.941431,1.025998,1.004931,0.967810])*0.75
    weight_PTransformer_up = ak.ones_like(event.category)
    weight_PTransformer_down = ak.ones_like(event.category)
    weight_PTransformer_central = ak.ones_like(event.category)
    Htagger_SF=event.category==2
    weight_PTransformer_central = ak.where(Htagger_SF, ak.ones_like(weight_PTransformer_central)*HvsQCD_SF[index], weight_PTransformer_central)
    weight_PTransformer_up = ak.where(Htagger_SF, ak.ones_like(weight_PTransformer_up)*HvsQCD_SFup[index], weight_PTransformer_up)
    weight_PTransformer_down = ak.where(Htagger_SF, ak.ones_like(weight_PTransformer_down)*HvsQCD_SFdown[index], weight_PTransformer_down)
    event['weight_PTransformer_up']=weight_PTransformer_up
    event['weight_PTransformer_down']=weight_PTransformer_down
    event['weight_PTransformer_central']=weight_PTransformer_central
    return event
events_sig = add_HtaggerSF(events_sig, int(mx))
events_bbgg = add_HtaggerSF(events_bbgg, int(mx))
events_zzgg = add_HtaggerSF(events_zzgg, int(mx))

#save systematics branches to root file
def add_sf_branches(events):
    events["CMS_hgg_mass"]=events["Diphoton_mass"]
    events["weight"]=events["weight_central"]
    events["dZ"]=np.ones(len(events['CMS_hgg_mass']))
    events["muon_highptreco_sf_Down01sigma"]=events["weight_nonisomuon_highptreco_sf_SelectedMuon_noiso_down"]/events["weight_nonisomuon_highptreco_sf_SelectedMuon_noiso_central"]
    events["muon_highptreco_sf_Up01sigma"]=events["weight_nonisomuon_highptreco_sf_SelectedMuon_noiso_up"]/events["weight_nonisomuon_highptreco_sf_SelectedMuon_noiso_central"]
    events["muon_highptid_sf_Down01sigma"]=events["weight_nonisomuon_highptid_sf_SelectedMuon_noiso_down"]/events["weight_nonisomuon_highptid_sf_SelectedMuon_noiso_central"]
    events["muon_highptid_sf_Up01sigma"]=events["weight_nonisomuon_highptid_sf_SelectedMuon_noiso_up"]/events["weight_nonisomuon_highptid_sf_SelectedMuon_noiso_central"]
    events["L1_prefiring_sf_Down01sigma"]=events["weight_L1_prefiring_sf_down"]/events["weight_L1_prefiring_sf_central"]
    events["L1_prefiring_sf_Up01sigma"]=events["weight_L1_prefiring_sf_up"]/events["weight_L1_prefiring_sf_central"]
    events["puWeight_Up01sigma"]=events["weight_pu_reweight_sf_up"]/events["weight_pu_reweight_sf_central"]
    events["puWeight_Down01sigma"]=events["weight_pu_reweight_sf_down"]/events["weight_pu_reweight_sf_central"]
    events["electron_veto_sf_Diphoton_Photon_Up01sigma"]=events["weight_electron_veto_sf_Diphoton_Photon_up"]/events["weight_electron_veto_sf_Diphoton_Photon_central"]
    events["electron_veto_sf_Diphoton_Photon_Down01sigma"]=events["weight_electron_veto_sf_Diphoton_Photon_down"]/events["weight_electron_veto_sf_Diphoton_Photon_central"]
    events["isoelectron_id_sf_SelectedElectron_iso_Up01sigma"]=events["weight_isoelectron_id_sf_SelectedElectron_iso_up"]/events["weight_isoelectron_id_sf_SelectedElectron_iso_central"]
    events["isoelectron_id_sf_SelectedElectron_iso_Down01sigma"]=events["weight_isoelectron_id_sf_SelectedElectron_iso_down"]/events["weight_isoelectron_id_sf_SelectedElectron_iso_central"]
    events["isoelectron_id_sf_SelectedElectron_noiso_Up01sigma"]= events["weight_isoelectron_id_sf_SelectedElectron_noiso_up"]/events["weight_isoelectron_id_sf_SelectedElectron_noiso_central"]
    events["isoelectron_id_sf_SelectedElectron_noiso_Down01sigma"]= events["weight_isoelectron_id_sf_SelectedElectron_noiso_down"]/events["weight_isoelectron_id_sf_SelectedElectron_noiso_central"] 
    events["isomuon_id_sf_SelectedMuon_iso_Up01sigma"]=events["weight_isomuon_id_sf_SelectedMuon_iso_up"]/events["weight_isomuon_id_sf_SelectedMuon_iso_central"]
    events["isomuon_id_sf_SelectedMuon_iso_Down01sigma"]=events["weight_isomuon_id_sf_SelectedMuon_iso_down"]/events["weight_isomuon_id_sf_SelectedMuon_iso_central"]        
    events["nonisoelectron_id_sf_SelectedElectron_noiso_Up01sigma"]=events["weight_nonisoelectron_id_sf_SelectedElectron_noiso_up"]/events["weight_nonisoelectron_id_sf_SelectedElectron_noiso_central"]
    events["nonisoelectron_id_sf_SelectedElectron_noiso_Down01sigma"]=events["weight_nonisoelectron_id_sf_SelectedElectron_noiso_down"]/events["weight_nonisoelectron_id_sf_SelectedElectron_noiso_central"]
    events["photon_id_sf_Diphoton_Photon_Up01sigma"]=events["weight_photon_id_sf_Diphoton_Photon_up"]/events["weight_photon_id_sf_Diphoton_Photon_central"]
    events["photon_id_sf_Diphoton_Photon_Down01sigma"]=events["weight_photon_id_sf_Diphoton_Photon_down"]/events["weight_photon_id_sf_Diphoton_Photon_central"]
    events["photon_presel_sf_Diphoton_Photon_Up01sigma"]=events["weight_photon_presel_sf_Diphoton_Photon_up"]/events["weight_photon_presel_sf_Diphoton_Photon_central"]
    events["photon_presel_sf_Diphoton_Photon_Down01sigma"]=events["weight_photon_presel_sf_Diphoton_Photon_down"]/events["weight_photon_presel_sf_Diphoton_Photon_central"]        
    events["trigger_sf_Up01sigma"]=events["weight_trigger_sf_up"]/events["weight_trigger_sf_central"]
    events["trigger_sf_Down01sigma"]=events["weight_trigger_sf_down"]/events["weight_trigger_sf_central"]
    events["PNetWvsQCDW1_sf_Up01sigma"]=events["weight_PNet_WvsQCDW1_up"]/events["weight_PNet_WvsQCDW1_central"]
    events["PNetWvsQCDW1_sf_Down01sigma"]=events["weight_PNet_WvsQCDW1_down"]/events["weight_PNet_WvsQCDW1_central"]
    events["PTransformerHtagger_sf_Up01sigma"]=events["weight_PTransformer_up"]/events["weight_PTransformer_central"]
    events["PTransformerHtagger_sf_Down01sigma"]=events["weight_PTransformer_down"]/events["weight_PTransformer_central"]
    events=events[['dZ','PNN_score','category','CMS_hgg_mass','weight','muon_highptid_sf_Down01sigma','muon_highptid_sf_Up01sigma','L1_prefiring_sf_Down01sigma','L1_prefiring_sf_Up01sigma','puWeight_Up01sigma','puWeight_Down01sigma','electron_veto_sf_Diphoton_Photon_Up01sigma','electron_veto_sf_Diphoton_Photon_Down01sigma','isoelectron_id_sf_SelectedElectron_iso_Up01sigma','isoelectron_id_sf_SelectedElectron_iso_Down01sigma','isoelectron_id_sf_SelectedElectron_noiso_Up01sigma','isoelectron_id_sf_SelectedElectron_noiso_Down01sigma','isomuon_id_sf_SelectedMuon_iso_Up01sigma','isomuon_id_sf_SelectedMuon_iso_Down01sigma','nonisoelectron_id_sf_SelectedElectron_noiso_Up01sigma','nonisoelectron_id_sf_SelectedElectron_noiso_Down01sigma','photon_id_sf_Diphoton_Photon_Up01sigma','photon_id_sf_Diphoton_Photon_Down01sigma','photon_presel_sf_Diphoton_Photon_Up01sigma','photon_presel_sf_Diphoton_Photon_Down01sigma','trigger_sf_Up01sigma','trigger_sf_Down01sigma','PTransformerHtagger_sf_Up01sigma','PTransformerHtagger_sf_Down01sigma','PNetWvsQCDW1_sf_Up01sigma','PNetWvsQCDW1_sf_Down01sigma','weight_PNet_WvsQCDW1_central']]
    return events
events_sig = add_sf_branches(events_sig)
events_bbgg = add_sf_branches(events_bbgg)
events_zzgg = add_sf_branches(events_zzgg)
events_sig_highpurity = events_sig[(events_sig['PNN_score'] > cut1) & (events_sig['PNN_score'] <= 1)]
events_sig_lowpurity = events_sig[(events_sig['PNN_score'] > cut2) & (events_sig['PNN_score'] <= cut1)]
ak.to_parquet(events_sig_highpurity,"./PBDT_HH_FHSL_combine_"+year+"/"+signal_samples['sig_output_name'][0]+"_highpurity.parquet")
ak.to_parquet(events_sig_lowpurity,"./PBDT_HH_FHSL_combine_"+year+"/"+signal_samples['sig_output_name'][0]+"_lowpurity.parquet")
events_bbgg_highpurity = events_bbgg[(events_bbgg['PNN_score'] > cut1) & (events_bbgg['PNN_score'] <= 1)]
events_bbgg_lowpurity = events_bbgg[(events_bbgg['PNN_score'] > cut2) & (events_bbgg['PNN_score'] <= cut1)]
ak.to_parquet(events_bbgg_highpurity,"./PBDT_HH_FHSL_combine_"+year+"/"+signal_samples['bbgg_output_name'][0]+"_highpurity.parquet")
ak.to_parquet(events_bbgg_lowpurity,"./PBDT_HH_FHSL_combine_"+year+"/"+signal_samples['bbgg_output_name'][0]+"_lowpurity.parquet")
events_zzgg_highpurity = events_zzgg[(events_zzgg['PNN_score'] > cut1) & (events_zzgg['PNN_score'] <= 1)]
events_zzgg_lowpurity = events_zzgg[(events_zzgg['PNN_score'] > cut2) & (events_zzgg['PNN_score'] <= cut1)]
ak.to_parquet(events_zzgg_highpurity,"./PBDT_HH_FHSL_combine_"+year+"/"+signal_samples['zzgg_output_name'][0]+"_highpurity.parquet")
ak.to_parquet(events_zzgg_lowpurity,"./PBDT_HH_FHSL_combine_"+year+"/"+signal_samples['zzgg_output_name'][0]+"_lowpurity.parquet")
#get data once only for the nominal parquet
events_data["CMS_hgg_mass"]=events_data["Diphoton_mass"]
events_data['weight']=events_data.weight_central
events_data=events_data[['CMS_hgg_mass','weight','PNN_score']]
events_data_highpurity = events_data[(events_data['PNN_score'] > cut1) & (events_data['PNN_score'] <= 1)]
events_data_lowpurity = events_data[(events_data['PNN_score'] > cut2) & (events_data['PNN_score'] <= cut1)]
massname="MX"+(signal_samples['sig_output_name'][0].split("MX"))[1].split("_cat12")[0]
sigcatAname="combineFHSL_cat12highpurity"
bbggcatAname="bbgg_cat12highpurity"
zzggcatAname="zzgg_cat12highpurity"
sigA_rootname="CombineFHSL_"+massname+"_"+year+"_"+sigcatAname+".root"
bbggA_rootname="CombineFHSL_"+massname+"_"+year+"_"+bbggcatAname+".root"
zzggA_rootname="CombineFHSL_"+massname+"_"+year+"_"+zzggcatAname+".root"
dataA_rootname="Data_"+year+"_"+sigcatAname+"_"+massname+".root"
dataA_treename="Data_13TeV_"+sigcatAname
sigA_treename="gghh_125_13TeV_"+sigcatAname
bbggA_treename="gghh_125_13TeV_"+bbggcatAname
zzggA_treename="gghh_125_13TeV_"+zzggcatAname
sigcatBname="combineFHSL_cat12lowpurity"
sigB_rootname="CombineFHSL_"+massname+"_"+year+"_"+sigcatBname+".root"
sigB_treename="gghh_125_13TeV_"+sigcatBname
dataB_rootname="Data_"+year+"_"+sigcatBname+"_"+massname+".root"
dataB_treename="Data_13TeV_"+sigcatBname
data_Acat_output_path="./PBDT_HH_FHSL_combine_"+year+"/"+dataA_rootname.replace(".root",".parquet")
data_Bcat_output_path="./PBDT_HH_FHSL_combine_"+year+"/"+dataB_rootname.replace(".root",".parquet")
ak.to_parquet(events_data_highpurity, data_Acat_output_path)
ak.to_parquet(events_data_lowpurity, data_Bcat_output_path)
bbgg_highpurity=ak.sum(events_bbgg_highpurity['weight'])
bbgg_lowpurity=ak.sum(events_bbgg_lowpurity['weight'])
zzgg_highpurity=ak.sum(events_zzgg_highpurity['weight'])
zzgg_lowpurity=ak.sum(events_zzgg_lowpurity['weight'])
FHSL_highpurity=ak.sum(events_sig_highpurity['weight'])
FHSL_lowpurity=ak.sum(events_sig_lowpurity['weight'])
print('start to process all the samples')
def process_sig_samples(FHfile,SLfile,bbggfile,zzggfile,sigoutput,bbggoutput,zzggoutput,input_features,cut1,cut2):
    def get_jet_variables(events):
        events['fatjet_1_pt'] = events['fatjet_1_pt']
        events['fatjet_2_pt'] = events['fatjet_2_pt']
        events['fatjet_1_eta'] = events['fatjet_1_eta']
        events['fatjet_2_eta'] = events['fatjet_2_eta']
        events['fatjet_1_phi'] = events['fatjet_1_phi']
        events['fatjet_2_phi'] = events['fatjet_2_phi']
        events['fatjet_1_msoftdrop'] = events['fatjet_1_msoftdrop']
        events['fatjet_2_msoftdrop'] = events['fatjet_2_msoftdrop']
        # get fatjet and diphoton dR
        fatjet_1_4D = vector.obj(pt=events.fatjet_1_pt, eta=events.fatjet_1_eta, phi=events.fatjet_1_phi, mass=events.fatjet_1_msoftdrop)
        fatjet_2_4D = vector.obj(pt=events.fatjet_2_pt, eta=events.fatjet_2_eta, phi=events.fatjet_2_phi, mass=events.fatjet_2_msoftdrop)
        fatjet_3_4D = vector.obj(pt=events.fatjet_3_pt, eta=events.fatjet_3_eta, phi=events.fatjet_3_phi, mass=events.fatjet_3_msoftdrop)
        diphoton_4D = vector.obj(pt=events.Diphoton_pt, eta=events.Diphoton_eta, phi=events.Diphoton_phi, mass=events.Diphoton_mass)
        fatjet_1_diphoton_dR = fatjet_1_4D.deltaR(diphoton_4D)
        fatjet_2_diphoton_dR = fatjet_2_4D.deltaR(diphoton_4D)
        events['fatjet_1_diphoton_dR'] = np.where(events.fatjet_1_pt>0, fatjet_1_diphoton_dR, -999)
        events['fatjet_2_diphoton_dR'] = np.where(events.fatjet_2_pt>0, fatjet_2_diphoton_dR, -999)
        leadphoton=vector.obj(pt=events.LeadPhoton_pt,eta=events.LeadPhoton_eta,phi=events.LeadPhoton_phi,mass=events.LeadPhoton_mass)
        subleadphoton=vector.obj(pt=events.SubleadPhoton_pt,eta=events.SubleadPhoton_eta,phi=events.SubleadPhoton_phi,mass=events.SubleadPhoton_mass)
        # dR with photon and fatjet
        fatjet_1_leadphoton_dR = fatjet_1_4D.deltaR(leadphoton)
        events['fatjet_1_leadphoton_dR'] = np.where(events.fatjet_1_pt>0, fatjet_1_leadphoton_dR, -999)
        fatjet_1_subleadphoton_dR = fatjet_1_4D.deltaR(subleadphoton)
        events['fatjet_1_subleadphoton_dR'] = np.where(events.fatjet_1_pt>0, fatjet_1_subleadphoton_dR, -999)
        fatjet_2_leadphoton_dR = fatjet_2_4D.deltaR(leadphoton)
        events['fatjet_2_leadphoton_dR'] = np.where(events.fatjet_2_pt>0, fatjet_2_leadphoton_dR, -999)
        fatjet_2_subleadphoton_dR = fatjet_2_4D.deltaR(subleadphoton)
        events['fatjet_2_subleadphoton_dR'] = np.where(events.fatjet_2_pt>0, fatjet_2_subleadphoton_dR, -999)
        # get fatjet1 and fatjet2 dR
        fatjet_1_2_dR = fatjet_1_4D.deltaR(fatjet_2_4D)
        events['fatjet_1_2_dR'] = np.where(np.logical_and(events.fatjet_1_pt>0, events.fatjet_2_pt>0), fatjet_1_2_dR, -999)
        # get the maximum fatjets mass with the combination of 2 fatjets in 3 fatjets
        fatjet_12_4D = fatjet_1_4D+fatjet_2_4D
        fatjet_13_4D = fatjet_1_4D+fatjet_3_4D
        fatjet_23_4D = fatjet_2_4D+fatjet_3_4D
        fatjet_12_msoftdrop = np.where(((fatjet_1_4D.pt >0) & (fatjet_2_4D.pt >0)), fatjet_12_4D.mass, -999)
        fatjet_13_msoftdrop = np.where(((fatjet_1_4D.pt >0) & (fatjet_3_4D.pt >0)), fatjet_13_4D.mass, -999)
        fatjet_23_msoftdrop = np.where(((fatjet_2_4D.pt >0) & (fatjet_3_4D.pt >0)), fatjet_23_4D.mass, -999)
        max_fatjets_mass = np.maximum(fatjet_12_msoftdrop, np.maximum(fatjet_13_msoftdrop, fatjet_23_msoftdrop))
        events['max_fatjets_mass'] = max_fatjets_mass
        # get max WvsQCD score with three fatjets
        events['fatjet_1_WvsQCDMD'] = events['fatjet_1_WvsQCDMD']
        events['fatjet_2_WvsQCDMD'] = events['fatjet_2_WvsQCDMD']
        # get max H4qvsQCD score with three fatjets
        events['fatjet_1_Hqqqq_vsQCDTop'] = events['fatjet_1_Hqqqq_vsQCDTop']
        events['fatjet_2_Hqqqq_vsQCDTop'] = events['fatjet_2_Hqqqq_vsQCDTop']
        # get XbbvsQCD
        events['fatjet_1_XbbvsQCDMD'] = (events['fatjet_1_particleNetMD_Xbb']) / (events['fatjet_1_particleNetMD_Xbb'] + events['fatjet_1_particleNetMD_QCD'])
        events['fatjet_2_XbbvsQCDMD'] = (events['fatjet_2_particleNetMD_Xbb']) / (events['fatjet_2_particleNetMD_Xbb'] + events['fatjet_2_particleNetMD_QCD'])
        Hscore = ak.concatenate([ak.unflatten(events['fatjet_1_Hqqqq_vsQCDTop'], counts=1), ak.unflatten(events['fatjet_2_Hqqqq_vsQCDTop'], counts=1)], axis=1)
        events['max_fatjet_Hqqqq_vsQCDTop'] = Hscore[ak.argsort(Hscore, axis=-1, ascending=False)][:,0]
        events['nGoodAK4jets'] = events['nGoodAK4jets']
        # add number of good AK8 jets
        events['nGoodAK8jets'] = events['nGoodAK8jets']
        # get 4 ak4 jets 4D info
        events['jet_1_pt'] = events['jet_1_pt']
        events['jet_2_pt'] = events['jet_2_pt']
        events['jet_3_pt'] = events['jet_3_pt']
        events['jet_1_eta'] = events['jet_1_eta']
        events['jet_2_eta'] = events['jet_2_eta']
        events['jet_3_eta'] = events['jet_3_eta']
        events['jet_1_phi'] = events['jet_1_phi']
        events['jet_2_phi'] = events['jet_2_phi']
        events['jet_3_phi'] = events['jet_3_phi']
        events['jet_1_mass'] = events['jet_1_mass']
        events['jet_2_mass'] = events['jet_2_mass']
        events['jet_3_mass'] = events['jet_3_mass']
        return events
    def get_leptons_variables(events):
        events['nGoodisoleptons'] = events['nGoodisoleptons']
        events['nGoodnonisoleptons'] = events['nGoodnonisoleptons']
        events['electrons_all_1_pt'] = events['electrons_all_1_pt']
        events['electrons_all_2_pt'] = events['electrons_all_2_pt']
        events['electrons_all_1_eta'] = events['electrons_all_1_eta']
        events['electrons_all_2_eta'] = events['electrons_all_2_eta']
        events['electrons_all_1_phi'] = events['electrons_all_1_phi']
        events['electrons_all_2_phi'] = events['electrons_all_2_phi']
        events['electrons_all_1_mass'] = events['electrons_all_1_mass']
        events['electrons_all_2_mass'] = events['electrons_all_2_mass']
        events['muons_all_1_pt'] = events['muons_all_1_pt']
        events['muons_all_2_pt'] = events['muons_all_2_pt']
        events['muons_all_1_eta'] = events['muons_all_1_eta']
        events['muons_all_2_eta'] = events['muons_all_2_eta']
        events['muons_all_1_phi'] = events['muons_all_1_phi']
        events['muons_all_2_phi'] = events['muons_all_2_phi']
        events['muons_all_1_mass'] = events['muons_all_1_mass']
        events['muons_all_2_mass'] = events['muons_all_2_mass']
        
        return events
    def get_met_variables(events):
        events['PuppiMET_pt'] = events['PuppiMET_pt']
        events['PuppiMET_sumEt'] = events['PuppiMET_sumEt']
        return events
    def get_sig_events_forApply(filename,mx,my):
        events = ak.from_parquet(filename)
        category_cut = ((events["category"]==2) | (events["category"]==1))
        photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
        events = events[ category_cut & photonID_cut]
        events['mx'] = np.ones(len(events))*int(mx)
        events['my'] = np.ones(len(events))*int(my)
        events = get_jet_variables(events)
        events = get_leptons_variables(events)
        events = get_met_variables(events)
        return events
    def model_predict(event, model, loaded_scaler, input_features):
        df = ak.to_pandas(event[input_features + ['weight_central','Diphoton_mass','Diphoton_minID','Diphoton_maxID']])
        X_test = loaded_scaler.transform(df[input_features])
        X_test = torch.tensor(X_test).float()
        proba = model(X_test)
        # proba = F.softmax(proba, dim=1) only for new models
        proba = proba.detach().numpy()
        pnn_score = (proba[:,1] + proba[:,2] + proba[:,3]) / (proba[:,0] + proba[:,1] + proba[:,2] + proba[:,3]) #old model 3TeV 220
        # pnn_score = (proba[:,4] + proba[:,2] + proba[:,3]) / (proba[:,0] + proba[:,1] + proba[:,2] + proba[:,3] + proba[:,4]) new model zhenxuan changed weight 3TeV 170
        event['PNN_score'] = pnn_score
        return event
    def add_shape_uncertainty_br(events):
        events["CMS_hgg_mass"] = events["Diphoton_mass"]
        events["weight"] = events["weight_central"]
        events["dZ"] = np.ones(len(events['CMS_hgg_mass']))
        events = events[['CMS_hgg_mass', 'weight', 'dZ', 'PNN_score']]
        return events
    mx = sigoutput.split('_')[1].split('X')[1]
    my = sigoutput.split('_')[2].split('H')[1]
    events_sigFH = get_sig_events_forApply(FHfile, mx, my)
    events_sigSL = get_sig_events_forApply(SLfile, mx, my)
    events_bbgg = get_sig_events_forApply(bbggfile, mx, my)
    events_zzgg = get_sig_events_forApply(zzggfile, mx, my)
    events_sig = ak.concatenate([events_sigFH, events_sigSL])
    del events_sigFH
    del events_sigSL
    events_sig = model_predict(events_sig, model, loaded_scaler, input_features)
    events_bbgg = model_predict(events_bbgg, model, loaded_scaler, input_features)
    events_zzgg = model_predict(events_zzgg, model, loaded_scaler, input_features)
    events_sig=add_shape_uncertainty_br(events_sig)
    events_bbgg=add_shape_uncertainty_br(events_bbgg)
    events_zzgg=add_shape_uncertainty_br(events_zzgg)
    events_sig_highpurity = events_sig[(events_sig['PNN_score'] > cut1) & (events_sig['PNN_score'] <= 1)]
    events_sig_lowpurity = events_sig[(events_sig['PNN_score'] > cut2) & (events_sig['PNN_score'] <= cut1)]
    ak.to_parquet(events_sig_highpurity, "./PBDT_HH_FHSL_combine_"+year+"/" + sigoutput + "_highpurity.parquet")
    ak.to_parquet(events_sig_lowpurity, "./PBDT_HH_FHSL_combine_"+year+"/" + sigoutput + "_lowpurity.parquet")
    events_bbgg_highpurity = events_bbgg[(events_bbgg['PNN_score'] > cut1) & (events_bbgg['PNN_score'] <= 1)]
    events_bbgg_lowpurity = events_bbgg[(events_bbgg['PNN_score'] > cut2) & (events_bbgg['PNN_score'] <= cut1)]
    ak.to_parquet(events_bbgg_highpurity, "./PBDT_HH_FHSL_combine_"+year+"/" + bbggoutput + "_highpurity.parquet")
    ak.to_parquet(events_bbgg_lowpurity, "./PBDT_HH_FHSL_combine_"+year+"/" + bbggoutput + "_lowpurity.parquet")
    events_zzgg_highpurity = events_zzgg[(events_zzgg['PNN_score'] > cut1) & (events_zzgg['PNN_score'] <= 1)]
    events_zzgg_lowpurity = events_zzgg[(events_zzgg['PNN_score'] > cut2) & (events_zzgg['PNN_score'] <= cut1)]
    ak.to_parquet(events_zzgg_highpurity, "./PBDT_HH_FHSL_combine_"+year+"/" + zzggoutput + "_highpurity.parquet")
    ak.to_parquet(events_zzgg_lowpurity, "./PBDT_HH_FHSL_combine_"+year+"/" + zzggoutput + "_lowpurity.parquet")
    highpuritylen=len(events_sig_highpurity)

    return highpuritylen
for i in range(1, len(signal_samples['FHpath'])):
    result=process_sig_samples(signal_samples['FHpath'][i], signal_samples['SLpath'][i], signal_samples['BBGGpath'][i], signal_samples['ZZggpath'][i], signal_samples['sig_output_name'][i], signal_samples['bbgg_output_name'][i], signal_samples['zzgg_output_name'][i], input_features, cut1, cut2)
def process_highpurity_sigfile(file):
    signal = file.split("/")[-1].split("_highpurity")[0].split("merge")[0]
    signaltype = "combineFHSL"
    name=file.split("/")[-1].split('merged_')[-1]
    if "down" in name:
        sys=name.split("_down")[0]
    elif "up" in name:
        sys=name.split("_up")[0]

    if "down" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity"
    rootname="./PBDT_HH_FHSL_combine_"+year+"/flashgginput/"+file.split("/")[-1].replace("parquet", "root")
    parquet_to_root(
        file,
        "./PBDT_HH_FHSL_combine_"+year+"/flashgginput/"
        + file.split("/")[-1].replace("parquet", "root"),
        treename=tree_name,
    )
    print("done")

    return tree_name
def process_highpurity_zzggfile(file):
    signaltype = "zzgg"
    name=file.split("/")[-1].split('merged_')[-1]
    if "down" in name:
        sys=name.split("_down")[0]
    elif "up" in name:
        sys=name.split("_up")[0]

    if "down" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity"
    parquet_to_root(
        file,
        "./PBDT_HH_FHSL_combine_"+year+"/flashgginput/"
        + file.split("/")[-1].replace("parquet", "root"),
        treename=tree_name,
    )
    print("done")

    return tree_name
def process_highpurity_bbggfile(file):
    signaltype = "bbgg"
    name=file.split("/")[-1].split('merged_')[-1]
    if "down" in name:
        sys=name.split("_down")[0]
    elif "up" in name:
        sys=name.split("_up")[0]

    if "down" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12highpurity"
    parquet_to_root(
        file,
        "./PBDT_HH_FHSL_combine_"+year+"/flashgginput/"
        + file.split("/")[-1].replace("parquet", "root"),
        treename=tree_name,
    )
    print("done")

    return tree_name
def process_lowpurity_sigfile(file):
    signal = file.split("/")[-1].split("_lowpurity")[0].split("merge")[0]
    signaltype = "combineFHSL"
    name=file.split("/")[-1].split('merged_')[-1]
    if "down" in name:
        sys=name.split("_down")[0]
    elif "up" in name:
        sys=name.split("_up")[0]

    if "down" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity"
    parquet_to_root(
        file,
        "./PBDT_HH_FHSL_combine_"+year+"/flashgginput/"
        + file.split("/")[-1].replace("parquet", "root"),
        treename=tree_name,
    )
    print("done")

    return tree_name
def process_lowpurity_zzggfile(file):
    signal = file.split("/")[-1].split("_lowpurity")[0].split("merge")[0]
    signaltype = "zzgg"
    name=file.split("/")[-1].split('merged_')[-1]
    if "down" in name:
        sys=name.split("_down")[0]
    elif "up" in name:
        sys=name.split("_up")[0]
    if "down" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity"
    parquet_to_root(
        file,
        "./PBDT_HH_FHSL_combine_"+year+"/flashgginput/"
        + file.split("/")[-1].replace("parquet", "root"),
        treename=tree_name,
    )
    print("done")
    return tree_name
def process_lowpurity_bbggfile(file):
    signal = file.split("/")[-1].split("_lowpurity")[0].split("merge")[0]
    signaltype = "bbgg"
    name=file.split("/")[-1].split('merged_')[-1]
    if "down" in name:
        sys=name.split("_down")[0]
    elif "up" in name:
        sys=name.split("_up")[0]
    if "down" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat12lowpurity"
    parquet_to_root(
        file,
        "./PBDT_HH_FHSL_combine_"+year+"/flashgginput/"
        + file.split("/")[-1].replace("parquet", "root"),
        treename=tree_name,
    )
    print("done")
    return tree_name
import multiprocessing

if __name__ == "__main__":
    print("starting to get root")
    Xmass=FHfile.split("M-")[1].split("_")[0]
    directory_path = "./PBDT_HH_FHSL_combine_"+year+"/flashgginput/MX"+Xmass+"_MH125"
    if os.path.exists(directory_path):
        rmcommand="rm -rf "+directory_path
        os.system(rmcommand)
        print(f"The directory {directory_path} exists.")
        os.mkdir(directory_path)
    else:
        os.mkdir(directory_path)
    highpurity_sigfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/CombineFHSL_MX"+Xmass+"_MH125_cat12_m*_highpurity.parquet")
    lowpurity_sigfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/CombineFHSL_MX"+Xmass+"_MH125_cat12_m*_lowpurity.parquet")
    lowpurity_bbggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/BBGG_MX"+Xmass+"_MH125_cat12_m*_lowpurity.parquet")
    highpurity_bbggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/BBGG_MX"+Xmass+"_MH125_cat12_m*_highpurity.parquet")
    highpurity_zzggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/ZZGG_MX"+Xmass+"_MH125_cat12_m*_highpurity.parquet")
    lowpurity_zzggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/ZZGG_MX"+Xmass+"_MH125_cat12_m*_lowpurity.parquet")
    
    #multiprocessing to convert parquet to root, and hadd the root files for highpurity and lowpurity WWggsiganl
    pool = multiprocessing.Pool(processes=10)
    results = []
    #highpurity signal
    for file in tqdm(highpurity_sigfiles):
        result = pool.apply_async(process_highpurity_sigfile, args=(file,))
        results.append(result)
    pool.close()
    pool.join()
    tree_names = [result.get() for result in tqdm(results)]
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_combineFHSL_cat12highpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/CombineFHSL_MX' + Xmass + '*highpurity.root'
    print(command)
    os.system(command)
    command = 'rm ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/CombineFHSL_MX'+Xmass+'*highpurity.root'
    os.system(command)
    pool = multiprocessing.Pool(processes=10)
    results = []
    #lowpurity signal
    pool = multiprocessing.Pool(processes=10)
    results = []
    for file in tqdm(lowpurity_sigfiles):
        result = pool.apply_async(process_lowpurity_sigfile, args=(file,))
        results.append(result)
    pool.close()
    pool.join()
    tree_names = [result.get() for result in tqdm(results)]
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_combineFHSL_cat12lowpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/CombineFHSL_MX' + Xmass + '*lowpurity.root'
    os.system(command)
    command = 'rm ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/CombineFHSL_MX'+Xmass+'*lowpurity.root'
    os.system(command)
    parquet_to_root(data_Acat_output_path,"./PBDT_HH_FHSL_combine_"+year+"/"+"flashgginput/MX"+Xmass+"_MH125/"+dataA_rootname,treename=dataA_treename,verbose=False)
    parquet_to_root(data_Bcat_output_path,"./PBDT_HH_FHSL_combine_"+year+"/"+"flashgginput/MX"+Xmass+"_MH125/"+dataB_rootname,treename=dataB_treename,verbose=False)
    parquet_to_root(data_Acat_output_path,"./PBDT_HH_FHSL_combine_"+year+"/"+"flashgginput/MX"+Xmass+"_MH125/"+dataA_rootname.replace("combineFHSL","bbgg"),treename=dataA_treename.replace("combineFHSL","bbgg"),verbose=False)
    parquet_to_root(data_Bcat_output_path,"./PBDT_HH_FHSL_combine_"+year+"/"+"flashgginput/MX"+Xmass+"_MH125/"+dataB_rootname.replace("combineFHSL","bbgg"),treename=dataB_treename.replace("combineFHSL","bbgg"),verbose=False)
    parquet_to_root(data_Acat_output_path,"./PBDT_HH_FHSL_combine_"+year+"/"+"flashgginput/MX"+Xmass+"_MH125/"+dataA_rootname.replace("combineFHSL","zzgg"),treename=dataA_treename.replace("combineFHSL","zzgg"),verbose=False)
    parquet_to_root(data_Bcat_output_path,"./PBDT_HH_FHSL_combine_"+year+"/"+"flashgginput/MX"+Xmass+"_MH125/"+dataB_rootname.replace("combineFHSL","zzgg"),treename=dataB_treename.replace("combineFHSL","zzgg"),verbose=False)
    #multiprocessing to convert parquet to root, and hadd the root files for highpurity and lowpurity bbggsignal
    #highpurity bbgg
    pool = multiprocessing.Pool(processes=10)
    results = []
    for file in tqdm(highpurity_bbggfiles):
        result = pool.apply_async(process_highpurity_bbggfile, args=(file,))
        results.append(result)
    pool.close()
    pool.join()
    tree_names = [result.get() for result in tqdm(results)]
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_bbgg_cat12highpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/BBGG_MX' + Xmass + '*highpurity.root'
    print(command)
    os.system(command)
    command = 'rm ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/BBGG_MX'+Xmass+'*highpurity.root'
    os.system(command)
    #lowpurity bbgg
    #lowpurity signal
    pool = multiprocessing.Pool(processes=10)
    results = []
    for file in tqdm(lowpurity_bbggfiles):
        result = pool.apply_async(process_lowpurity_bbggfile, args=(file,))
        results.append(result)
    pool.close()
    pool.join()
    tree_names = [result.get() for result in tqdm(results)]
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_bbgg_cat12lowpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/BBGG_MX' + Xmass + '*lowpurity.root'
    os.system(command)
    command = 'rm ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/BBGG_MX'+Xmass+'*lowpurity.root'
    os.system(command)
    #multiprocessing to convert parquet to root, and hadd the root files for highpurity and lowpurity zzggsignal
    #highpurity zzgg
    pool = multiprocessing.Pool(processes=10)
    results = []
    for file in tqdm(highpurity_zzggfiles):
        result = pool.apply_async(process_highpurity_zzggfile, args=(file,))
        results.append(result)
    pool.close()
    pool.join()
    tree_names = [result.get() for result in tqdm(results)]
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_zzgg_cat12highpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/ZZGG_MX' + Xmass + '*highpurity.root'
    print(command)
    os.system(command)
    command = 'rm ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/ZZGG_MX'+Xmass+'*highpurity.root'
    os.system(command)
    #lowpurity zzgg
    pool = multiprocessing.Pool(processes=10)
    results = []
    for file in tqdm(lowpurity_zzggfiles):
        result = pool.apply_async(process_lowpurity_zzggfile, args=(file,))
        results.append(result)
    pool.close()
    pool.join()
    tree_names = [result.get() for result in tqdm(results)]
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_zzgg_cat12lowpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/ZZGG_MX' + Xmass + '*lowpurity.root'
    print(command)
    os.system(command)
    command = 'rm ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/ZZGG_MX'+Xmass+'*lowpurity.root'
    os.system(command)
    boundary={str(Xmass):{"cut1":cut1,"cut2":cut2,"FHSL_highpurity":FHSL_highpurity,"FHSL_lowpurity":FHSL_lowpurity,"BBGG_highpurity":bbgg_highpurity,"BBGG_lowpurity":bbgg_lowpurity,"ZZGG_highpurity":zzgg_highpurity,"ZZGG_lowpurity":zzgg_lowpurity,"highpurity_sigeff":highpurity_sigeff,"lowpurity_sigeff":lowpurity_sigeff,"highpurity_sidebandnum":highpurity_sidebandnum,"lowpurity_sidebandnum":lowpurity_sidebandnum}}
    with open("./PBDT_HH_FHSL_combine_"+year+"/flashgginput/MX"+Xmass+"_MH125/boundaries.json", "w") as f:
        json.dump(boundary, f, indent=4)
    plt.savefig("./PBDT_HH_FHSL_combine_"+year+"/flashgginput/MX"+Xmass+"_MH125/dnnscore.png", dpi=140)
