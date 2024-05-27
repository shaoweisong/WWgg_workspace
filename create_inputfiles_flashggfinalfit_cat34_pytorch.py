#/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
import awkward as ak
import sys
debug=False
local=True
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
parser.add_argument('--model',type=str,default="/eos/user/z/zhenxuan/PNN_wwgg/PNN_YH_combined/resolved_FHSL/data/simple_DNN_real_epoch_300_mx500_850_random_mass_reweight_forsignal_moremorelayer_model_nobbggweight_noweightdecay_fixedbug4outputsize_fixedsignaltarget_withmorebkg_noscheduler_fixedbkgclassto2_nohighmasslowweight_adddiphotonptreweight_withsilu/model.pth", help='model file')
parser.add_argument('--scalar',type=str,default="/eos/user/z/zhenxuan/PNN_wwgg/PNN_YH_combined/resolved_FHSL/data/simple_DNN_real_epoch_300_mx500_850_random_mass_reweight_forsignal_moremorelayer_model_nobbggweight_noweightdecay_fixedbug4outputsize_fixedsignaltarget_withmorebkg_noscheduler_fixedbkgclassto2_nohighmasslowweight_adddiphotonptreweight_withsilu/scaler_params.json", help='scalar file')
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
class MultiClassDNN_model(nn.Module):
    def __init__(self, input_size, output_size):
        super(MultiClassDNN_model, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 528),
            nn.BatchNorm1d(528),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(528, 528),
            nn.BatchNorm1d(528),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(528, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        self.fc5 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.5)
        )
        self.fc6 = nn.Linear(64, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        return out
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
    d1 = np.sum(events_datasideband_1['weight_central'][(events_datasideband_1['Diphoton_mass'] > 135) | (events_datasideband_1['Diphoton_mass'] < 115)])
    if d1>10:
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
    d2 = np.sum(events_datasideband_2['weight_central'][(events_datasideband_2['Diphoton_mass'] > 135) | (events_datasideband_2['Diphoton_mass'] < 115)])
    # significance2 = s2 / np.sqrt(b2)
    significance2 = s2 / np.sqrt(b2)

    significance = np.sqrt(significance1**2 + significance2**2 )
    fwhm = np.sqrt(fwhm1**2 + fwhm2**2)
    return significance
# 自定义函数，将参数字符串转换为列表
def costheta1(event):
    W1=ak.zip({
    "pt":event['W1_pt'],
    "eta":event['W1_eta'],
    "phi":event['W1_phi'],
    "mass":event['W1_mass']},with_name="Momentum4D")
    H1=ak.zip({
    "pt":event['W1_W2_pt'],
    "eta":event['W1_W2_eta'],
    "phi":event['W1_W2_phi'],
    "mass":event['W1_W2_mass']},with_name="Momentum4D")
    H2=ak.zip({
    "pt":event['Diphoton_pt'],
    "eta":event['Diphoton_eta'],
    "phi":event['Diphoton_phi'],
    "mass":event['Diphoton_mass']},with_name="Momentum4D")
    HH=H1+H2
    # H1 rest frame
    boost_vec=vector.obj(px=H1.px / H1.E,py=H1.py / H1.E,pz=H1.pz / H1.E)
    #     boost_vec=vector.obj(px=HH.px / (HH.E*HH.mass),py=HH.py / (HH.E*HH.mass),pz=HH.pz / (HH.E*HH.mass))
    W1_rest=W1.boost(boost_vec)
    HH_rest=HH.boost(boost_vec)
    # calculate W1 HH momentum magnitude
    p1 = np.sqrt(W1_rest.px**2 + W1_rest.py**2 + W1_rest.pz**2)
    p2 = np.sqrt(HH_rest.px**2 + HH_rest.py**2 + HH_rest.pz**2)
    # calculate  W1 unit vector and HH unit vector 
    ux1 = W1_rest.px / p1
    uy1 = W1_rest.py / p1
    uz1 = W1_rest.pz / p1
    ux2 = HH_rest.px / p2
    uy2 = HH_rest.py / p2
    uz2 = HH_rest.pz / p2
    # The dot product of two unit vectors is equal to cos theta
    cos_theta = ux1 * ux2 + uy1 * uy2 + uz1 * uz2
    dummy = ak.zeros_like(W1.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    costheta1 = ak.where(((H1.pt<0)|(H2.pt<0)|(W1.pt<0)),(ak.ones_like(dummy)*(-999)),cos_theta)
    return costheta1
def costheta2(event):
    W2=ak.zip({
    "pt":event['W2_pt'],
    "eta":event['W2_eta'],
    "phi":event['W2_phi'],
    "mass":event['W2_mass']},with_name="Momentum4D")
    H1=ak.zip({
    "pt":event['W1_W2_pt'],
    "eta":event['W1_W2_eta'],
    "phi":event['W1_W2_phi'],
    "mass":event['W1_W2_mass']},with_name="Momentum4D")
    H2=ak.zip({
    "pt":event['Diphoton_pt'],
    "eta":event['Diphoton_eta'],
    "phi":event['Diphoton_phi'],
    "mass":event['Diphoton_mass']},with_name="Momentum4D")
    HH=H1+H2
    # H1 rest frame
    boost_vec=vector.obj(px=H2.px / H2.E,py=H2.py / H2.E,pz=H2.pz / H2.E)
#     boost_vec=vector.obj(px=HH.px / (HH.E*HH.mass),py=HH.py / (HH.E*HH.mass),pz=HH.pz / (HH.E*HH.mass))
    W2_rest=W2.boost(boost_vec)
    HH_rest=HH.boost(boost_vec)
    # calculate W1 HH momentum magnitude
    p1 = np.sqrt(W2_rest.px**2 + W2_rest.py**2 + W2_rest.pz**2)
    p2 = np.sqrt(HH_rest.px**2 + HH_rest.py**2 + HH_rest.pz**2)
    # calculate  W1 unit vector and HH unit vector 
    ux1 = W2_rest.px / p1
    uy1 = W2_rest.py / p1
    uz1 = W2_rest.pz / p1
    ux2 = HH_rest.px / p2
    uy2 = HH_rest.py / p2
    uz2 = HH_rest.pz / p2
    # The dot product of two unit vectors is equal to cos theta
    cos_theta = ux1 * ux2 + uy1 * uy2 + uz1 * uz2
    dummy = ak.zeros_like(W2.pt)
    dummy = ak.ones_like(ak.fill_none(dummy, 0))
    
    return ak.where(((H1.pt<0)|(H2.pt<0)|(W2.pt<0)),(ak.ones_like(dummy)*(-999)),cos_theta)
def get_photon_variables(events):
    # get Diphoton pt 
    events['Diphoton_pt'] = events['Diphoton_pt']
    # get Diphoton eta
    events['Diphoton_eta'] = events['Diphoton_eta']
    # get Diphoton phi
    events['Diphoton_phi'] = events['Diphoton_phi']
    # get Diphoton dR
    events['Diphoton_dR'] = events['Diphoton_dR']

    # get min and max ID photon info
    pho_pt=ak.concatenate([ak.unflatten(events.LeadPhoton_pt,counts=1),ak.unflatten(events.SubleadPhoton_pt,counts=1)],axis=1)
    pho_eta=ak.concatenate([ak.unflatten(events.LeadPhoton_eta,counts=1),ak.unflatten(events.SubleadPhoton_eta,counts=1)],axis=1)
    pho_phi=ak.concatenate([ak.unflatten(events.LeadPhoton_phi,counts=1),ak.unflatten(events.SubleadPhoton_phi,counts=1)],axis=1)
    pho_mass=ak.concatenate([ak.unflatten(events.LeadPhoton_mass,counts=1),ak.unflatten(events.SubleadPhoton_mass,counts=1)],axis=1)
    photonID=ak.concatenate([ak.unflatten(events.Diphoton_minID_modified,counts=1),ak.unflatten(events.Diphoton_minID_modified,counts=1)],axis=1)
    #sort photonID by max to min
    photonID_sorted = photonID[ak.argsort(photonID,axis=1,ascending=False)]
    events["Diphoton_minID_modified"] = photonID_sorted[:,1]
    events["Diphoton_maxID_modified"] = photonID_sorted[:,0]
    photon = ak.zip({"pt":pho_pt,"eta":pho_eta,"phi":pho_phi,"mass":pho_mass})
    events['LeadPhoton_pt'] = photon.pt[:,0]
    events['LeadPhoton_eta'] = photon.eta[:,0]
    events['LeadPhoton_phi'] = photon.phi[:,0]
    events['SubleadPhoton_pt'] = photon.pt[:,1]
    events['SubleadPhoton_eta'] = photon.eta[:,1]
    events['SubleadPhoton_phi'] = photon.phi[:,1]
    # get the minPhotonID and maxPhotonID
    events['Diphoton_minID'] = events['Diphoton_minID_modified']
    
    events['Diphoton_maxID'] = events['Diphoton_maxID_modified']
    return events
def get_FHSL_jet_W_variables(events, is_FH_SL):
    # get 4 jets 4D info
    events['jet_1_pt'] = events['jet_1_pt']
    events['jet_2_pt'] = events['jet_2_pt']
    events['jet_3_pt'] = events['jet_3_pt']
    events['jet_4_pt'] = events['jet_4_pt']
    events['jet_1_eta'] = events['jet_1_eta']
    events['jet_2_eta'] = events['jet_2_eta']
    events['jet_3_eta'] = events['jet_3_eta']
    events['jet_4_eta'] = events['jet_4_eta']
    events['jet_1_phi'] = events['jet_1_phi']
    events['jet_2_phi'] = events['jet_2_phi']
    events['jet_3_phi'] = events['jet_3_phi']
    events['jet_4_phi'] = events['jet_4_phi']
    events['jet_1_mass'] = events['jet_1_mass']
    events['jet_2_mass'] = events['jet_2_mass']
    events['jet_3_mass'] = events['jet_3_mass']
    events['jet_4_mass'] = events['jet_4_mass']
    # get jet multiplicity
    events['nGoodAK4jets'] = events['nGoodAK4jets']
    events['nGoodAK8jets'] = events['nGoodAK8jets']
    # get 4 jets distance: distance between 1st and 2nd, 1st and 3rd, 1st and 4th, 2nd and 3rd, 2nd and 4th, 3rd and 4th mean min(pt1, pt2) * deltaR**2
    jet_1_pt = np.where(events.jet_1_pt>0, events.jet_1_pt, -999)
    jet_2_pt = np.where(events.jet_2_pt>0, events.jet_2_pt, -999)
    jet_3_pt = np.where(events.jet_3_pt>0, events.jet_3_pt, -999)
    jet_4_pt = np.where(events.jet_4_pt>0, events.jet_4_pt, -999)
    jet_1_eta = np.where(events.jet_1_pt>0, events.jet_1_eta, -999)
    jet_2_eta = np.where(events.jet_2_pt>0, events.jet_2_eta, -999)
    jet_3_eta = np.where(events.jet_3_pt>0, events.jet_3_eta, -999)
    jet_4_eta = np.where(events.jet_4_pt>0, events.jet_4_eta, -999)
    jet_1_phi = np.where(events.jet_1_pt>0, events.jet_1_phi, -999)
    jet_2_phi = np.where(events.jet_2_pt>0, events.jet_2_phi, -999)
    jet_3_phi = np.where(events.jet_3_pt>0, events.jet_3_phi, -999)
    jet_4_phi = np.where(events.jet_4_pt>0, events.jet_4_phi, -999)
    jet_1_mass = np.where(events.jet_1_pt>0, events.jet_1_mass, -999)
    jet_2_mass = np.where(events.jet_2_pt>0, events.jet_2_mass, -999)
    jet_3_mass = np.where(events.jet_3_pt>0, events.jet_3_mass, -999)
    jet_4_mass = np.where(events.jet_4_pt>0, events.jet_4_mass, -999)
    
    
    jet_1_4D = vector.obj(pt=jet_1_pt, eta=jet_1_eta, phi=jet_1_phi, mass=jet_1_mass)
    jet_2_4D = vector.obj(pt=jet_2_pt, eta=jet_2_eta, phi=jet_2_phi, mass=jet_2_mass)
    jet_3_4D = vector.obj(pt=jet_3_pt, eta=jet_3_eta, phi=jet_3_phi, mass=jet_3_mass)
    jet_4_4D = vector.obj(pt=jet_4_pt, eta=jet_4_eta, phi=jet_4_phi, mass=jet_4_mass)
    
    jet_1_2_dR = jet_1_4D.deltaR(jet_2_4D)
    jet_1_2_dR = np.where(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), jet_1_2_dR, -999)
    jet_1_3_dR = jet_1_4D.deltaR(jet_3_4D)
    jet_1_3_dR = np.where(np.logical_and(events.jet_1_pt>0, events.jet_3_pt>0), jet_1_3_dR, -999)
    jet_1_4_dR = jet_1_4D.deltaR(jet_4_4D)
    jet_1_4_dR = np.where(np.logical_and(events.jet_1_pt>0, events.jet_4_pt>0), jet_1_4_dR, -999)
    jet_2_3_dR = jet_2_4D.deltaR(jet_3_4D)
    jet_2_3_dR = np.where(np.logical_and(events.jet_2_pt>0, events.jet_3_pt>0), jet_2_3_dR, -999)
    jet_2_4_dR = jet_2_4D.deltaR(jet_4_4D)
    jet_2_4_dR = np.where(np.logical_and(events.jet_2_pt>0, events.jet_4_pt>0), jet_2_4_dR, -999)
    jet_3_4_dR = jet_3_4D.deltaR(jet_4_4D)
    jet_3_4_dR = np.where(np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0), jet_3_4_dR, -999)

    
    
    distance_12 = np.where(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), np.minimum(events.jet_1_pt, events.jet_2_pt)*jet_1_2_dR**2, -999)
    distance_13 = np.where(np.logical_and(events.jet_1_pt>0, events.jet_3_pt>0), np.minimum(events.jet_1_pt, events.jet_3_pt)*jet_1_3_dR**2, -999)
    distance_14 = np.where(np.logical_and(events.jet_1_pt>0, events.jet_4_pt>0), np.minimum(events.jet_1_pt, events.jet_4_pt)*jet_1_4_dR**2, -999)
    distance_23 = np.where(np.logical_and(events.jet_2_pt>0, events.jet_3_pt>0), np.minimum(events.jet_2_pt, events.jet_3_pt)*jet_2_3_dR**2, -999)
    distance_24 = np.where(np.logical_and(events.jet_2_pt>0, events.jet_4_pt>0), np.minimum(events.jet_2_pt, events.jet_4_pt)*jet_2_4_dR**2, -999)
    distance_34 = np.where(np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0), np.minimum(events.jet_3_pt, events.jet_4_pt)*jet_3_4_dR**2, -999)
    events['distance_12'] = distance_12
    events['distance_13'] = distance_13
    events['distance_14'] = distance_14
    events['distance_23'] = distance_23
    events['distance_24'] = distance_24
    events['distance_34'] = distance_34


    # lepton is electron or muons depend on electron_iso_pt >0 or muons_all_1_pt_pt >0
    # TOFIX: need to change muons_all_1_pt back to muons_iso_pt
    lepton = vector.obj(pt=np.where(events.electron_iso_pt>0, events.electron_iso_pt, events.muons_all_1_pt), eta=np.where(events.electron_iso_pt>0, events.electron_iso_eta, events.muons_all_1_eta), phi=np.where(events.electron_iso_pt>0, events.electron_iso_phi, events.muons_all_1_phi), mass=np.where(events.electron_iso_pt>0, events.electron_iso_mass, events.muons_all_1_mass))
    
    # get W1 and W2 4D info
    if is_FH_SL == 'FH':
        W1 = jet_1_4D + jet_2_4D
        W2 = jet_3_4D + jet_4_4D
    if is_FH_SL == 'SL':
        events['lepton_pt'] = lepton.pt
        events['lepton_eta'] = lepton.eta
        events['lepton_phi'] = lepton.phi
        events['lepton_iso_MET_mt'] = np.sqrt(lepton.mass*lepton.mass+2*(lepton.pt*events.PuppiMET_pt-lepton.pt*events.PuppiMET_pt))
        events['leptonpt_METpt'] = lepton.pt*events.PuppiMET_pt
        events['lepton_E'] = lepton.E
        events['lepton_Et'] = np.sqrt(lepton.pt**2 + lepton.mass**2)
        
        W1 = jet_1_4D + lepton
        W2 = W1 # attention: for SL, W2 will all set to -999 in deal_with_none_object
        
    events['W1_pt'] = W1.pt
    events['W1_eta'] = W1.eta
    events['W1_phi'] = W1.phi
    events['W1_mass'] = W1.mass
    events['W2_pt'] = W2.pt
    events['W2_eta'] = W2.eta
    events['W2_phi'] = W2.phi
    events['W2_mass'] = W2.mass
    # get W1 and W2 dR
    W1_W2_dR = W1.deltaR(W2)
    # jet_1_4D.pt > 0 and jet_2_4D.pt > 0 and jet_3_4D.pt > 0 and jet_4_4D.pt > 0
    condition_4jets = np.logical_and(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0))
    condition_12_jets = np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0)
    condition_13_jets = np.logical_and(events.jet_1_pt>0, events.jet_3_pt>0)
    condition_14_jets = np.logical_and(events.jet_1_pt>0, events.jet_4_pt>0)
    condition_23_jets = np.logical_and(events.jet_2_pt>0, events.jet_3_pt>0)
    condition_24_jets = np.logical_and(events.jet_2_pt>0, events.jet_4_pt>0)    
    condition_34_jets = np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0)
    events['W1_W2_dR'] = W1_W2_dR
    
    # get W1+W2 4D info
    W1_W2_mass = (W1+W2).mass
    W1_W2_mass = W1_W2_mass
    W1_W2_pt = (W1+W2).pt
    W1_W2_pt = W1_W2_pt
    W1_W2_eta = (W1+W2).eta
    W1_W2_eta = W1_W2_eta
    W1_W2_phi = (W1+W2).phi
    W1_W2_phi = W1_W2_phi
    events['W1_W2_mass'] = W1_W2_mass
    events['W1_W2_pt'] = W1_W2_pt
    events['W1_W2_eta'] = W1_W2_eta
    events['W1_W2_phi'] = W1_W2_phi
    
    # get W1 and W2 dR with Diphoton
    diphoton_4D = vector.obj(pt=events.Diphoton_pt, eta=events.Diphoton_eta, phi=events.Diphoton_phi, mass=events.Diphoton_mass)
    
    W1_diphoton_dR = W1.deltaR(diphoton_4D)
    W2_diphoton_dR = W2.deltaR(diphoton_4D)
    # W_pts[arg_min] > 0 
    events['W1_diphoton_dR'] = W1_diphoton_dR
    events['W2_diphoton_dR'] = W2_diphoton_dR
    
    # get maximum dR between 4 jets and diphoton
    jet_1_diphoton_dR = np.where(jet_1_pt>0, jet_1_4D.deltaR(diphoton_4D), -999)
    jet_2_diphoton_dR = np.where(jet_2_pt>0, jet_2_4D.deltaR(diphoton_4D), -999)
    jet_3_diphoton_dR = np.where(jet_3_pt>0, jet_3_4D.deltaR(diphoton_4D), -999)
    jet_4_diphoton_dR = np.where(jet_4_pt>0, jet_4_4D.deltaR(diphoton_4D), -999)
    max_dR_jet_diphoton = np.maximum(jet_1_diphoton_dR, np.maximum(jet_2_diphoton_dR, np.maximum(jet_3_diphoton_dR, jet_4_diphoton_dR)))
    events['max_dR_jet_diphoton'] = max_dR_jet_diphoton
    
    # get of two max bscores from 7 jets
    bscore_list = [events.jet_1_btagDeepFlavB, events.jet_2_btagDeepFlavB, events.jet_3_btagDeepFlavB, events.jet_4_btagDeepFlavB, events.jet_5_btagDeepFlavB, events.jet_6_btagDeepFlavB]
    # sum each two bscore and get the maximum
    bscore_sum_list = [bscore_list[i]+bscore_list[j] for i in range(6) for j in range(i+1, 6)]
    max_bscore_sum = np.max(bscore_sum_list, axis=0)
    events['jet_1_btagDeepFlavB'] = events['jet_1_btagDeepFlavB']
    events['jet_2_btagDeepFlavB'] = events['jet_2_btagDeepFlavB']
    events['jet_3_btagDeepFlavB'] = events['jet_3_btagDeepFlavB']
    events['jet_4_btagDeepFlavB'] = events['jet_4_btagDeepFlavB']
    events['sum_two_max_bscores'] = max_bscore_sum
    
    # get costhetastar
    objects1 = diphoton_4D
    W1_W2_4D = vector.obj(pt=events.W1_W2_pt, eta=events.W1_W2_eta, phi=events.W1_W2_phi, mass=events.W1_W2_mass)
    objects2= W1_W2_4D
    HH=objects1+objects2 #obj could be HH condidate    
    
    p= np.sqrt(HH.px**2 + HH.py**2 + HH.pz**2)
    
    events['costhetastar'] = ak.where((objects2.pt<0),-999,HH.pz/p)
    # get costheta1 and costheta2
    events['costheta1'] = costheta1(events)
    events['costheta2'] = costheta2(events)

    return events
def deal_with_none_object(events, is_FH_SL):
    
    if is_FH_SL == "FH":
        # for lepton
        events['lepton_pt'] = np.ones_like(events.W2_pt) * -999
        events['lepton_eta'] = np.ones_like(events.W2_pt) * -999
        events['lepton_phi'] = np.ones_like(events.W2_pt) * -999
        events['lepton_iso_MET_mt'] = np.ones_like(events.W2_pt) * -999
        events['leptonpt_METpt'] = np.ones_like(events.W2_pt) * -999
        events['lepton_E'] = np.ones_like(events.W2_pt) * -999
        events['lepton_Et'] = np.ones_like(events.W2_pt) * -999
        
        # for W1 and W2
        # condition is jet_1_pt > 0 and jet_2_pt > 0
        condition_12_jets = np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0)
        events['W1_pt'] = np.where(condition_12_jets, events.W1_pt, -999)
        events['W1_eta'] = np.where(condition_12_jets, events.W1_eta, -999)
        events['W1_phi'] = np.where(condition_12_jets, events.W1_phi, -999)
        events['W1_mass'] = np.where(condition_12_jets, events.W1_mass, -999)
        # condition is jet_3_pt > 0 and jet_4_pt > 0
        condition_34_jets = np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0)
        events['W2_pt'] = np.where(condition_34_jets, events.W2_pt, -999)
        events['W2_eta'] = np.where(condition_34_jets, events.W2_eta, -999)
        events['W2_phi'] = np.where(condition_34_jets, events.W2_phi, -999)
        events['W2_mass'] = np.where(condition_34_jets, events.W2_mass, -999)
        # for W1 and W2 dR
        events['W1_W2_dR'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_dR, -999)
        # for W1+W2
        events['W1_W2_mass'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_mass, -999)
        events['W1_W2_pt'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_pt, -999)
        events['W1_W2_eta'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_eta, -999)
        events['W1_W2_phi'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_phi, -999)
        # for W1 and W2 dR with Diphoton
        events['W1_diphoton_dR'] = np.where(condition_12_jets, events.W1_diphoton_dR, -999)
        events['W2_diphoton_dR'] = np.where(condition_34_jets, events.W2_diphoton_dR, -999)
        # for sum of two max bscore
        events['sum_two_max_bscores'] = np.where(np.logical_and(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0)), events.sum_two_max_bscores, -999)
        # for costhetastar
        events['costhetastar'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.costhetastar, -999)
        # for costheta1 and costheta2
        events['costheta1'] = np.where(condition_12_jets, events.costheta1, -999)
        events['costheta2'] = np.where(condition_34_jets, events.costheta2, -999)
        events['electrons_all_1_pt'] = np.ones_like(events.W2_pt) * -999
        events['electrons_all_2_pt'] = np.ones_like(events.W2_pt) * -999
        events['electrons_all_1_eta'] = np.ones_like(events.W2_pt) * -999
        events['electrons_all_2_eta'] = np.ones_like(events.W2_pt) * -999
        events['electrons_all_1_phi'] = np.ones_like(events.W2_pt) * -999
        events['electrons_all_2_phi'] = np.ones_like(events.W2_pt) * -999
        events['electrons_all_1_mass'] = np.ones_like(events.W2_pt) * -999
        events['electrons_all_2_mass'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_1_pt'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_2_pt'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_1_eta'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_2_eta'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_1_phi'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_2_phi'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_1_mass'] = np.ones_like(events.W2_pt) * -999
        events['muons_all_2_mass'] = np.ones_like(events.W2_pt) * -999
    if is_FH_SL == "SL":
        # for lepton
        events['lepton_pt'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_pt'], -999)
        events['lepton_eta'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_eta'], -999)
        events['lepton_phi'] =  np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_phi'], -999)
        events['lepton_iso_MET_mt'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_iso_MET_mt'], -999)
        events['leptonpt_METpt'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['leptonpt_METpt'], -999)
        events['lepton_E'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_E'], -999)
        events['lepton_Et'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_Et'], -999)
        # for W1
        # condition is jet_1_pt > 0 and jet_2_pt > 0
        condition_12_jets = np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0)
        events['W1_pt'] = np.where(condition_12_jets, events.W1_pt, -999)
        events['W1_eta'] = np.where(condition_12_jets, events.W1_eta, -999)
        events['W1_phi'] = np.where(condition_12_jets, events.W1_phi, -999)
        events['W1_mass'] = np.where(condition_12_jets, events.W1_mass, -999)
        # for W2 is all -999
        events['W2_pt'] = np.ones_like(events.W1_pt)*-999
        events['W2_eta'] = np.ones_like(events.W1_pt)*-999
        events['W2_phi'] = np.ones_like(events.W1_pt)*-999
        events['W2_mass'] = np.ones_like(events.W1_pt)*-999
        # for W1 and W2 dR
        events['W1_W2_dR'] = np.ones_like(events.W1_pt)*-999
        # for W1+W2
        events['W1_W2_mass'] = np.ones_like(events.W1_pt)*-999
        events['W1_W2_pt'] = np.ones_like(events.W1_pt)*-999
        events['W1_W2_eta'] = np.ones_like(events.W1_pt)*-999
        events['W1_W2_phi'] = np.ones_like(events.W1_pt)*-999
        # for W1 and W2 dR with Diphoton
        events['W1_diphoton_dR'] = np.where(condition_12_jets, events.W1_diphoton_dR, -999)
        events['W2_diphoton_dR'] = np.ones_like(events.W1_pt)*-999
        # for sum of two max bscore
        events['sum_two_max_bscores'] = np.ones_like(events.W1_pt)*-999
        # for costhetastar is all -999
        events['costhetastar'] = np.ones_like(events.W1_pt)*-999
        # for costheta1
        events['costheta1'] = np.where(condition_12_jets, events.costheta1, -999)
        # for costheta2 is all -999
        events['costheta2'] = np.ones_like(events.W1_pt)*-999
        events['jet_1_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
        events['jet_2_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
        events['jet_3_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
        events['jet_4_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
        
    return events
def get_leptons_variables(events):
    events['nGoodisoleptons'] = events['nGoodisoleptons']
    events['nGoodnonisoleptons'] = events['nGoodnonisoleptons']
    events['nGoodisoelectrons'] = events['nGoodisoelectrons']
    events['nGoodnonisoelectrons'] = events['nGoodnonisoelectrons']
    events['nGoodisomuons'] = events['nGoodisomuons']
    events['nGoodnonisomuons'] = events['nGoodnonisomuons']
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
    category_cut = ((events["category"]==3) | (events["category"]==4))
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    events = get_photon_variables(events)
    events = get_leptons_variables(events)
    events = get_met_variables(events)
    eventsFH=events[events["category"]==4]
    eventsSL=events[events["category"]==3]
    eventsFH = get_FHSL_jet_W_variables(eventsFH, "FH")
    eventsFH = deal_with_none_object(eventsFH, "FH")
    eventsSL = get_FHSL_jet_W_variables(eventsSL, "SL")
    eventsSL = deal_with_none_object(eventsSL, "SL")
    events = ak.concatenate([eventsFH, eventsSL])
    return events
def get_bkgcat3_events_forApply(filename,mx, my):
    events = ak.from_parquet(filename)
    category_cut = (events["category"]==3)
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    events = get_photon_variables(events)
    events = get_leptons_variables(events)
    events = get_met_variables(events)
    events = get_FHSL_jet_W_variables(events, "SL")
    events = deal_with_none_object(events, "SL")
    return events
def get_bkgcat4_events_forApply(filename,mx, my):
    events = ak.from_parquet(filename)
    category_cut = events["category"]==4
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    events = get_FHSL_jet_W_variables(events, "FH")
    events = deal_with_none_object(events, "FH")
    return events

    return events
def get_data_events_forApply(filename,mx, my):
    events = ak.from_parquet(filename)
    category_cut = ((events["category"]==3) | (events["category"]==4))
    photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
    events = events[ category_cut & photonID_cut]
    events['mx'] = np.ones(len(events))*int(mx)
    events['my'] = np.ones(len(events))*int(my)
    events = get_photon_variables(events)
    events = get_leptons_variables(events)
    events = get_met_variables(events)
    eventsFH=events[events["category"]==4]
    eventsSL=events[events["category"]==3]
    eventsFH = get_FHSL_jet_W_variables(eventsFH, "FH")
    eventsFH = deal_with_none_object(eventsFH, "FH")
    eventsSL = get_FHSL_jet_W_variables(eventsSL, "SL")
    eventsSL = deal_with_none_object(eventsSL, "SL")
    events = ak.concatenate([eventsFH, eventsSL])

    return events
print('start to get input features')

input_features = ['Diphoton_pt','Diphoton_eta','Diphoton_phi','Diphoton_dR','LeadPhoton_pt','LeadPhoton_eta','LeadPhoton_phi','SubleadPhoton_pt','SubleadPhoton_eta','SubleadPhoton_phi',"Diphoton_minID_modified","Diphoton_maxID_modified",'jet_1_pt','jet_2_pt','jet_3_pt','jet_4_pt','jet_1_eta','jet_2_eta','jet_3_eta','jet_4_eta','jet_1_phi','jet_2_phi','jet_3_phi','jet_4_phi','jet_1_mass','jet_2_mass','jet_3_mass','jet_4_mass','nGoodAK4jets','nGoodAK8jets','distance_12','distance_13','distance_14','distance_23','distance_24','distance_34','W1_pt','W1_eta','W1_phi','W1_mass','W2_pt','W2_eta','W2_phi','W2_mass','W1_W2_dR','W1_W2_mass','W1_W2_pt','W1_W2_eta','W1_W2_phi','W1_diphoton_dR','W2_diphoton_dR','max_dR_jet_diphoton','jet_1_btagDeepFlavB','jet_2_btagDeepFlavB','jet_3_btagDeepFlavB','jet_4_btagDeepFlavB','sum_two_max_bscores','costhetastar','PuppiMET_pt','PuppiMET_sumEt','nGoodisoleptons','nGoodnonisoleptons', 'nGoodisoelectrons','nGoodnonisoelectrons','nGoodisomuons','nGoodnonisomuons','lepton_pt','lepton_eta','lepton_phi','lepton_iso_MET_mt','lepton_E','lepton_Et','electrons_all_1_pt','electrons_all_1_eta','electrons_all_1_phi','electrons_all_1_mass','muons_all_1_pt','muons_all_1_eta','muons_all_1_phi','muons_all_1_mass','mx','leptonpt_METpt']
other_vars  = ['weight_central','Diphoton_mass']
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
    dir_name="CombineFHSL_MX" + FHfile.split("M-")[1].split("_")[0] + "_MH125_cat34_"+(FHfile.split("/")[-1]).split(".")[0]
    # dir_name = "CombineFHSL_MX1100_MH125_cat34_merged_FJER_down"
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
del events_sigFH
del events_sigSL
events_data = get_data_events_forApply(datapath,mx,my)
events_pp_cat3= get_bkgcat3_events_forApply(bkgfiles[0],mx,my)
events_pp_cat4= get_bkgcat4_events_forApply(bkgfiles[1],mx,my)
events_dd_cat3= get_bkgcat3_events_forApply(bkgfiles[2],mx,my)
events_dd_cat4= get_bkgcat4_events_forApply(bkgfiles[3],mx,my)
import json
model = MultiClassDNN_model(len(input_features), 4)
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
    df = ak.to_pandas(event[input_features + ['weight_central','Diphoton_mass']])
    X_test = loaded_scaler.transform(df[input_features])
    X_test = torch.tensor(X_test).float()
    proba = model(X_test)
    proba = F.softmax(proba, dim=1)
    proba = proba.detach().numpy()
    pnn_score = (proba[:,2] + proba[:,3]) / (proba[:,0] + proba[:,1] + proba[:,2] + proba[:,3])
    return pnn_score
# for sig, data, bkgmc, bbgg and zzgg
# evaluate the model
event_pp_cat4_1=events_pp_cat4[:int(len(events_pp_cat4)/4)]
event_pp_cat4_2=events_pp_cat4[int(len(events_pp_cat4)/4):int(len(events_pp_cat4)/2)]
event_pp_cat4_3=events_pp_cat4[int(len(events_pp_cat4)/2):int(len(events_pp_cat4)*3/4)]
event_pp_cat4_4=events_pp_cat4[int(len(events_pp_cat4)*3/4):]
PNN_score = model_predict(event_pp_cat4_1, model, loaded_scaler, input_features)
event_pp_cat4_1['PNN_score'] = PNN_score
PNN_score = model_predict(event_pp_cat4_2, model, loaded_scaler, input_features)
event_pp_cat4_2['PNN_score'] = PNN_score
PNN_score = model_predict(event_pp_cat4_3, model, loaded_scaler, input_features)
event_pp_cat4_3['PNN_score'] = PNN_score
PNN_score = model_predict(event_pp_cat4_4, model, loaded_scaler, input_features)
event_pp_cat4_4['PNN_score'] = PNN_score
PNN_score = model_predict(events_pp_cat3, model, loaded_scaler, input_features)
events_pp_cat3['PNN_score'] = PNN_score
events_pp_cat4 = ak.concatenate([event_pp_cat4_1,event_pp_cat4_2,event_pp_cat4_3,event_pp_cat4_4])
del event_pp_cat4_1
del event_pp_cat4_2
del event_pp_cat4_3
del event_pp_cat4_4
events_pp=ak.concatenate([events_pp_cat3,events_pp_cat4])
del events_pp_cat3
del events_pp_cat4
PNN_score = model_predict(events_dd_cat3, model, loaded_scaler, input_features)
events_dd_cat3['PNN_score'] = PNN_score
PNN_score = model_predict(events_dd_cat4, model, loaded_scaler, input_features)
events_dd_cat4['PNN_score'] = PNN_score
events_dd = ak.concatenate([events_dd_cat3,events_dd_cat4])
del events_dd_cat3
del events_dd_cat4
event_bkgmc=ak.concatenate([events_pp,events_dd])
print("get the PNN score for bkgmc")
PNN_score = model_predict(events_bbgg, model, loaded_scaler, input_features)
events_bbgg['PNN_score'] = PNN_score
print("get the PNN score for bbgg")
PNN_score = model_predict(events_sig, model, loaded_scaler, input_features)
events_sig['PNN_score'] = PNN_score
print("get the PNN score for signal")
PNN_score = model_predict(events_data, model, loaded_scaler, input_features)
events_data['PNN_score'] = PNN_score
print("get the PNN score for data")

PNN_score = model_predict(events_zzgg, model, loaded_scaler, input_features)
events_zzgg['PNN_score'] = PNN_score
print('get all the PBDT score for merged nominal.parquet')
ak.to_parquet(events_data, './Data_'+year+'_combineFHSL_cat34test_MX3000_MH125.parquet')
ak.to_parquet(events_sig, './Signal_'+year+'_combineFHSL_cat34test_MX3000_MH125.parquet')
ak.to_parquet(events_bbgg, './BBGG_'+year+'_combineFHSL_cat34test_MX3000_MH125.parquet')

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
pbounds = {
        'bo1': (0.00001, 0.2), 
        'bo2': (0.01, 0.6), 
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
# for _ in range(300):
for _ in range(3):
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
highpurity_sigeff=ak.sum(events_sig.weight_central[(events_sig.PNN_score>cut1)&(events_sig.PNN_score<=1)])/ak.sum(events_sig.weight_central)
lowpurity_sigeff=ak.sum(events_sig.weight_central[(events_sig.PNN_score>cut2)&(events_sig.PNN_score<=cut1)])/ak.sum(events_sig.weight_central)
highpurity_sidebandnum=len(events_data[((events_data.Diphoton_mass>135)|(events_data.Diphoton_mass<115))&((events_data.PNN_score > cut1) & (events_data.PNN_score <= 1))])
lowpurity_sidebandnum=len(events_data[((events_data.Diphoton_mass>135)|(events_data.Diphoton_mass<115))&((events_data.PNN_score > cut2) & (events_data.PNN_score <= cut1))])
print('high purity signal efficiency:',highpurity_sigeff)
print('low purity signal efficiency:',lowpurity_sigeff)
print('high purity sideband number:',highpurity_sidebandnum)
print('low purity sideband number:',lowpurity_sidebandnum)


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

    events=events[['dZ','PNN_score','category','CMS_hgg_mass','weight','muon_highptid_sf_Down01sigma','muon_highptid_sf_Up01sigma','L1_prefiring_sf_Down01sigma','L1_prefiring_sf_Up01sigma','puWeight_Up01sigma','puWeight_Down01sigma','electron_veto_sf_Diphoton_Photon_Up01sigma','electron_veto_sf_Diphoton_Photon_Down01sigma','isoelectron_id_sf_SelectedElectron_iso_Up01sigma','isoelectron_id_sf_SelectedElectron_iso_Down01sigma','isoelectron_id_sf_SelectedElectron_noiso_Up01sigma','isoelectron_id_sf_SelectedElectron_noiso_Down01sigma','isomuon_id_sf_SelectedMuon_iso_Up01sigma','isomuon_id_sf_SelectedMuon_iso_Down01sigma','nonisoelectron_id_sf_SelectedElectron_noiso_Up01sigma','nonisoelectron_id_sf_SelectedElectron_noiso_Down01sigma','photon_id_sf_Diphoton_Photon_Up01sigma','photon_id_sf_Diphoton_Photon_Down01sigma','photon_presel_sf_Diphoton_Photon_Up01sigma','photon_presel_sf_Diphoton_Photon_Down01sigma','trigger_sf_Up01sigma','trigger_sf_Down01sigma']]
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
massname="MX"+(signal_samples['sig_output_name'][0].split("MX"))[1].split("_cat34")[0]
sigcatAname="combineFHSL_cat34highpurity"
bbggcatAname="bbgg_cat34highpurity"
zzggcatAname="zzgg_cat34highpurity"
sigA_rootname="CombineFHSL_"+massname+"_"+year+"_"+sigcatAname+".root"
bbggA_rootname="CombineFHSL_"+massname+"_"+year+"_"+bbggcatAname+".root"
zzggA_rootname="CombineFHSL_"+massname+"_"+year+"_"+zzggcatAname+".root"
dataA_rootname="Data_"+year+"_"+sigcatAname+"_"+massname+".root"
dataA_treename="Data_13TeV_"+sigcatAname
sigA_treename="gghh_125_13TeV_"+sigcatAname
bbggA_treename="gghh_125_13TeV_"+bbggcatAname
zzggA_treename="gghh_125_13TeV_"+zzggcatAname
sigcatBname="combineFHSL_cat34lowpurity"
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
    def costheta1(event):
        W1=ak.zip({
        "pt":event['W1_pt'],
        "eta":event['W1_eta'],
        "phi":event['W1_phi'],
        "mass":event['W1_mass']},with_name="Momentum4D")
        H1=ak.zip({
        "pt":event['W1_W2_pt'],
        "eta":event['W1_W2_eta'],
        "phi":event['W1_W2_phi'],
        "mass":event['W1_W2_mass']},with_name="Momentum4D")
        H2=ak.zip({
        "pt":event['Diphoton_pt'],
        "eta":event['Diphoton_eta'],
        "phi":event['Diphoton_phi'],
        "mass":event['Diphoton_mass']},with_name="Momentum4D")
        HH=H1+H2
        # H1 rest frame
        boost_vec=vector.obj(px=H1.px / H1.E,py=H1.py / H1.E,pz=H1.pz / H1.E)
        #     boost_vec=vector.obj(px=HH.px / (HH.E*HH.mass),py=HH.py / (HH.E*HH.mass),pz=HH.pz / (HH.E*HH.mass))
        W1_rest=W1.boost(boost_vec)
        HH_rest=HH.boost(boost_vec)
        # calculate W1 HH momentum magnitude
        p1 = np.sqrt(W1_rest.px**2 + W1_rest.py**2 + W1_rest.pz**2)
        p2 = np.sqrt(HH_rest.px**2 + HH_rest.py**2 + HH_rest.pz**2)
        # calculate  W1 unit vector and HH unit vector 
        ux1 = W1_rest.px / p1
        uy1 = W1_rest.py / p1
        uz1 = W1_rest.pz / p1
        ux2 = HH_rest.px / p2
        uy2 = HH_rest.py / p2
        uz2 = HH_rest.pz / p2
        # The dot product of two unit vectors is equal to cos theta
        cos_theta = ux1 * ux2 + uy1 * uy2 + uz1 * uz2
        dummy = ak.zeros_like(W1.pt)
        dummy = ak.ones_like(ak.fill_none(dummy, 0))
        costheta1 = ak.where(((H1.pt<0)|(H2.pt<0)|(W1.pt<0)),(ak.ones_like(dummy)*(-999)),cos_theta)
        return costheta1
    def costheta2(event):
        W2=ak.zip({
        "pt":event['W2_pt'],
        "eta":event['W2_eta'],
        "phi":event['W2_phi'],
        "mass":event['W2_mass']},with_name="Momentum4D")
        H1=ak.zip({
        "pt":event['W1_W2_pt'],
        "eta":event['W1_W2_eta'],
        "phi":event['W1_W2_phi'],
        "mass":event['W1_W2_mass']},with_name="Momentum4D")
        H2=ak.zip({
        "pt":event['Diphoton_pt'],
        "eta":event['Diphoton_eta'],
        "phi":event['Diphoton_phi'],
        "mass":event['Diphoton_mass']},with_name="Momentum4D")
        HH=H1+H2
        # H1 rest frame
        boost_vec=vector.obj(px=H2.px / H2.E,py=H2.py / H2.E,pz=H2.pz / H2.E)
    #     boost_vec=vector.obj(px=HH.px / (HH.E*HH.mass),py=HH.py / (HH.E*HH.mass),pz=HH.pz / (HH.E*HH.mass))
        W2_rest=W2.boost(boost_vec)
        HH_rest=HH.boost(boost_vec)
        # calculate W1 HH momentum magnitude
        p1 = np.sqrt(W2_rest.px**2 + W2_rest.py**2 + W2_rest.pz**2)
        p2 = np.sqrt(HH_rest.px**2 + HH_rest.py**2 + HH_rest.pz**2)
        # calculate  W1 unit vector and HH unit vector 
        ux1 = W2_rest.px / p1
        uy1 = W2_rest.py / p1
        uz1 = W2_rest.pz / p1
        ux2 = HH_rest.px / p2
        uy2 = HH_rest.py / p2
        uz2 = HH_rest.pz / p2
        # The dot product of two unit vectors is equal to cos theta
        cos_theta = ux1 * ux2 + uy1 * uy2 + uz1 * uz2
        dummy = ak.zeros_like(W2.pt)
        dummy = ak.ones_like(ak.fill_none(dummy, 0))
        
        return ak.where(((H1.pt<0)|(H2.pt<0)|(W2.pt<0)),(ak.ones_like(dummy)*(-999)),cos_theta)
    def get_photon_variables(events):
        # get Diphoton pt 
        events['Diphoton_pt'] = events['Diphoton_pt']
        # get Diphoton eta
        events['Diphoton_eta'] = events['Diphoton_eta']
        # get Diphoton phi
        events['Diphoton_phi'] = events['Diphoton_phi']
        # get Diphoton dR
        events['Diphoton_dR'] = events['Diphoton_dR']

        # get min and max ID photon info
        pho_pt=ak.concatenate([ak.unflatten(events.LeadPhoton_pt,counts=1),ak.unflatten(events.SubleadPhoton_pt,counts=1)],axis=1)
        pho_eta=ak.concatenate([ak.unflatten(events.LeadPhoton_eta,counts=1),ak.unflatten(events.SubleadPhoton_eta,counts=1)],axis=1)
        pho_phi=ak.concatenate([ak.unflatten(events.LeadPhoton_phi,counts=1),ak.unflatten(events.SubleadPhoton_phi,counts=1)],axis=1)
        pho_mass=ak.concatenate([ak.unflatten(events.LeadPhoton_mass,counts=1),ak.unflatten(events.SubleadPhoton_mass,counts=1)],axis=1)
        photonID=ak.concatenate([ak.unflatten(events.Diphoton_minID_modified,counts=1),ak.unflatten(events.Diphoton_minID_modified,counts=1)],axis=1)
        #sort photonID by max to min
        photonID_sorted = photonID[ak.argsort(photonID,axis=1,ascending=False)]
        events["Diphoton_minID_modified"] = photonID_sorted[:,1]
        events["Diphoton_maxID_modified"] = photonID_sorted[:,0]
        photon = ak.zip({"pt":pho_pt,"eta":pho_eta,"phi":pho_phi,"mass":pho_mass})
        events['LeadPhoton_pt'] = photon.pt[:,0]
        events['LeadPhoton_eta'] = photon.eta[:,0]
        events['LeadPhoton_phi'] = photon.phi[:,0]
        events['SubleadPhoton_pt'] = photon.pt[:,1]
        events['SubleadPhoton_eta'] = photon.eta[:,1]
        events['SubleadPhoton_phi'] = photon.phi[:,1]
        # get the minPhotonID and maxPhotonID
        events['Diphoton_minID'] = events['Diphoton_minID_modified']
        
        events['Diphoton_maxID'] = events['Diphoton_maxID_modified']
        return events
    def get_FHSL_jet_W_variables(events, is_FH_SL):
        # get 4 jets 4D info
        events['jet_1_pt'] = events['jet_1_pt']
        events['jet_2_pt'] = events['jet_2_pt']
        events['jet_3_pt'] = events['jet_3_pt']
        events['jet_4_pt'] = events['jet_4_pt']
        events['jet_1_eta'] = events['jet_1_eta']
        events['jet_2_eta'] = events['jet_2_eta']
        events['jet_3_eta'] = events['jet_3_eta']
        events['jet_4_eta'] = events['jet_4_eta']
        events['jet_1_phi'] = events['jet_1_phi']
        events['jet_2_phi'] = events['jet_2_phi']
        events['jet_3_phi'] = events['jet_3_phi']
        events['jet_4_phi'] = events['jet_4_phi']
        events['jet_1_mass'] = events['jet_1_mass']
        events['jet_2_mass'] = events['jet_2_mass']
        events['jet_3_mass'] = events['jet_3_mass']
        events['jet_4_mass'] = events['jet_4_mass']
        # get jet multiplicity
        events['nGoodAK4jets'] = events['nGoodAK4jets']
        events['nGoodAK8jets'] = events['nGoodAK8jets']
        # get 4 jets distance: distance between 1st and 2nd, 1st and 3rd, 1st and 4th, 2nd and 3rd, 2nd and 4th, 3rd and 4th mean min(pt1, pt2) * deltaR**2
        jet_1_pt = np.where(events.jet_1_pt>0, events.jet_1_pt, -999)
        jet_2_pt = np.where(events.jet_2_pt>0, events.jet_2_pt, -999)
        jet_3_pt = np.where(events.jet_3_pt>0, events.jet_3_pt, -999)
        jet_4_pt = np.where(events.jet_4_pt>0, events.jet_4_pt, -999)
        jet_1_eta = np.where(events.jet_1_pt>0, events.jet_1_eta, -999)
        jet_2_eta = np.where(events.jet_2_pt>0, events.jet_2_eta, -999)
        jet_3_eta = np.where(events.jet_3_pt>0, events.jet_3_eta, -999)
        jet_4_eta = np.where(events.jet_4_pt>0, events.jet_4_eta, -999)
        jet_1_phi = np.where(events.jet_1_pt>0, events.jet_1_phi, -999)
        jet_2_phi = np.where(events.jet_2_pt>0, events.jet_2_phi, -999)
        jet_3_phi = np.where(events.jet_3_pt>0, events.jet_3_phi, -999)
        jet_4_phi = np.where(events.jet_4_pt>0, events.jet_4_phi, -999)
        jet_1_mass = np.where(events.jet_1_pt>0, events.jet_1_mass, -999)
        jet_2_mass = np.where(events.jet_2_pt>0, events.jet_2_mass, -999)
        jet_3_mass = np.where(events.jet_3_pt>0, events.jet_3_mass, -999)
        jet_4_mass = np.where(events.jet_4_pt>0, events.jet_4_mass, -999)
        
        
        jet_1_4D = vector.obj(pt=jet_1_pt, eta=jet_1_eta, phi=jet_1_phi, mass=jet_1_mass)
        jet_2_4D = vector.obj(pt=jet_2_pt, eta=jet_2_eta, phi=jet_2_phi, mass=jet_2_mass)
        jet_3_4D = vector.obj(pt=jet_3_pt, eta=jet_3_eta, phi=jet_3_phi, mass=jet_3_mass)
        jet_4_4D = vector.obj(pt=jet_4_pt, eta=jet_4_eta, phi=jet_4_phi, mass=jet_4_mass)
        
        jet_1_2_dR = jet_1_4D.deltaR(jet_2_4D)
        jet_1_2_dR = np.where(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), jet_1_2_dR, -999)
        jet_1_3_dR = jet_1_4D.deltaR(jet_3_4D)
        jet_1_3_dR = np.where(np.logical_and(events.jet_1_pt>0, events.jet_3_pt>0), jet_1_3_dR, -999)
        jet_1_4_dR = jet_1_4D.deltaR(jet_4_4D)
        jet_1_4_dR = np.where(np.logical_and(events.jet_1_pt>0, events.jet_4_pt>0), jet_1_4_dR, -999)
        jet_2_3_dR = jet_2_4D.deltaR(jet_3_4D)
        jet_2_3_dR = np.where(np.logical_and(events.jet_2_pt>0, events.jet_3_pt>0), jet_2_3_dR, -999)
        jet_2_4_dR = jet_2_4D.deltaR(jet_4_4D)
        jet_2_4_dR = np.where(np.logical_and(events.jet_2_pt>0, events.jet_4_pt>0), jet_2_4_dR, -999)
        jet_3_4_dR = jet_3_4D.deltaR(jet_4_4D)
        jet_3_4_dR = np.where(np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0), jet_3_4_dR, -999)

        
        
        distance_12 = np.where(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), np.minimum(events.jet_1_pt, events.jet_2_pt)*jet_1_2_dR**2, -999)
        distance_13 = np.where(np.logical_and(events.jet_1_pt>0, events.jet_3_pt>0), np.minimum(events.jet_1_pt, events.jet_3_pt)*jet_1_3_dR**2, -999)
        distance_14 = np.where(np.logical_and(events.jet_1_pt>0, events.jet_4_pt>0), np.minimum(events.jet_1_pt, events.jet_4_pt)*jet_1_4_dR**2, -999)
        distance_23 = np.where(np.logical_and(events.jet_2_pt>0, events.jet_3_pt>0), np.minimum(events.jet_2_pt, events.jet_3_pt)*jet_2_3_dR**2, -999)
        distance_24 = np.where(np.logical_and(events.jet_2_pt>0, events.jet_4_pt>0), np.minimum(events.jet_2_pt, events.jet_4_pt)*jet_2_4_dR**2, -999)
        distance_34 = np.where(np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0), np.minimum(events.jet_3_pt, events.jet_4_pt)*jet_3_4_dR**2, -999)
        events['distance_12'] = distance_12
        events['distance_13'] = distance_13
        events['distance_14'] = distance_14
        events['distance_23'] = distance_23
        events['distance_24'] = distance_24
        events['distance_34'] = distance_34


        # lepton is electron or muons depend on electron_iso_pt >0 or muons_all_1_pt_pt >0
        # TOFIX: need to change muons_all_1_pt back to muons_iso_pt
        lepton = vector.obj(pt=np.where(events.electron_iso_pt>0, events.electron_iso_pt, events.muons_all_1_pt), eta=np.where(events.electron_iso_pt>0, events.electron_iso_eta, events.muons_all_1_eta), phi=np.where(events.electron_iso_pt>0, events.electron_iso_phi, events.muons_all_1_phi), mass=np.where(events.electron_iso_pt>0, events.electron_iso_mass, events.muons_all_1_mass))
        
        # get W1 and W2 4D info
        if is_FH_SL == 'FH':
            W1 = jet_1_4D + jet_2_4D
            W2 = jet_3_4D + jet_4_4D
        if is_FH_SL == 'SL':
            events['lepton_pt'] = lepton.pt
            events['lepton_eta'] = lepton.eta
            events['lepton_phi'] = lepton.phi
            events['lepton_iso_MET_mt'] = np.sqrt(lepton.mass*lepton.mass+2*(lepton.pt*events.PuppiMET_pt-lepton.pt*events.PuppiMET_pt))
            events['leptonpt_METpt'] = lepton.pt*events.PuppiMET_pt
            events['lepton_E'] = lepton.E
            events['lepton_Et'] = np.sqrt(lepton.pt**2 + lepton.mass**2)
            
            W1 = jet_1_4D + lepton
            W2 = W1 # attention: for SL, W2 will all set to -999 in deal_with_none_object
            
        events['W1_pt'] = W1.pt
        events['W1_eta'] = W1.eta
        events['W1_phi'] = W1.phi
        events['W1_mass'] = W1.mass
        events['W2_pt'] = W2.pt
        events['W2_eta'] = W2.eta
        events['W2_phi'] = W2.phi
        events['W2_mass'] = W2.mass
        # get W1 and W2 dR
        W1_W2_dR = W1.deltaR(W2)
        # jet_1_4D.pt > 0 and jet_2_4D.pt > 0 and jet_3_4D.pt > 0 and jet_4_4D.pt > 0
        condition_4jets = np.logical_and(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0))
        condition_12_jets = np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0)
        condition_13_jets = np.logical_and(events.jet_1_pt>0, events.jet_3_pt>0)
        condition_14_jets = np.logical_and(events.jet_1_pt>0, events.jet_4_pt>0)
        condition_23_jets = np.logical_and(events.jet_2_pt>0, events.jet_3_pt>0)
        condition_24_jets = np.logical_and(events.jet_2_pt>0, events.jet_4_pt>0)    
        condition_34_jets = np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0)
        events['W1_W2_dR'] = W1_W2_dR
        
        # get W1+W2 4D info
        W1_W2_mass = (W1+W2).mass
        W1_W2_mass = W1_W2_mass
        W1_W2_pt = (W1+W2).pt
        W1_W2_pt = W1_W2_pt
        W1_W2_eta = (W1+W2).eta
        W1_W2_eta = W1_W2_eta
        W1_W2_phi = (W1+W2).phi
        W1_W2_phi = W1_W2_phi
        events['W1_W2_mass'] = W1_W2_mass
        events['W1_W2_pt'] = W1_W2_pt
        events['W1_W2_eta'] = W1_W2_eta
        events['W1_W2_phi'] = W1_W2_phi
        
        # get W1 and W2 dR with Diphoton
        diphoton_4D = vector.obj(pt=events.Diphoton_pt, eta=events.Diphoton_eta, phi=events.Diphoton_phi, mass=events.Diphoton_mass)
        
        W1_diphoton_dR = W1.deltaR(diphoton_4D)
        W2_diphoton_dR = W2.deltaR(diphoton_4D)
        # W_pts[arg_min] > 0 
        events['W1_diphoton_dR'] = W1_diphoton_dR
        events['W2_diphoton_dR'] = W2_diphoton_dR
        
        # get maximum dR between 4 jets and diphoton
        jet_1_diphoton_dR = np.where(jet_1_pt>0, jet_1_4D.deltaR(diphoton_4D), -999)
        jet_2_diphoton_dR = np.where(jet_2_pt>0, jet_2_4D.deltaR(diphoton_4D), -999)
        jet_3_diphoton_dR = np.where(jet_3_pt>0, jet_3_4D.deltaR(diphoton_4D), -999)
        jet_4_diphoton_dR = np.where(jet_4_pt>0, jet_4_4D.deltaR(diphoton_4D), -999)
        max_dR_jet_diphoton = np.maximum(jet_1_diphoton_dR, np.maximum(jet_2_diphoton_dR, np.maximum(jet_3_diphoton_dR, jet_4_diphoton_dR)))
        events['max_dR_jet_diphoton'] = max_dR_jet_diphoton
        
        # get of two max bscores from 7 jets
        bscore_list = [events.jet_1_btagDeepFlavB, events.jet_2_btagDeepFlavB, events.jet_3_btagDeepFlavB, events.jet_4_btagDeepFlavB, events.jet_5_btagDeepFlavB, events.jet_6_btagDeepFlavB]
        # sum each two bscore and get the maximum
        bscore_sum_list = [bscore_list[i]+bscore_list[j] for i in range(6) for j in range(i+1, 6)]
        max_bscore_sum = np.max(bscore_sum_list, axis=0)
        events['jet_1_btagDeepFlavB'] = events['jet_1_btagDeepFlavB']
        events['jet_2_btagDeepFlavB'] = events['jet_2_btagDeepFlavB']
        events['jet_3_btagDeepFlavB'] = events['jet_3_btagDeepFlavB']
        events['jet_4_btagDeepFlavB'] = events['jet_4_btagDeepFlavB']
        events['sum_two_max_bscores'] = max_bscore_sum
        
        # get costhetastar
        objects1 = diphoton_4D
        W1_W2_4D = vector.obj(pt=events.W1_W2_pt, eta=events.W1_W2_eta, phi=events.W1_W2_phi, mass=events.W1_W2_mass)
        objects2= W1_W2_4D
        HH=objects1+objects2 #obj could be HH condidate    
        
        p= np.sqrt(HH.px**2 + HH.py**2 + HH.pz**2)
        
        events['costhetastar'] = ak.where((objects2.pt<0),-999,HH.pz/p)
        # get costheta1 and costheta2
        events['costheta1'] = costheta1(events)
        events['costheta2'] = costheta2(events)

        return events
    def deal_with_none_object(events, is_FH_SL):
        
        if is_FH_SL == "FH":
            # for lepton
            events['lepton_pt'] = np.ones_like(events.W2_pt) * -999
            events['lepton_eta'] = np.ones_like(events.W2_pt) * -999
            events['lepton_phi'] = np.ones_like(events.W2_pt) * -999
            events['lepton_iso_MET_mt'] = np.ones_like(events.W2_pt) * -999
            events['leptonpt_METpt'] = np.ones_like(events.W2_pt) * -999
            events['lepton_E'] = np.ones_like(events.W2_pt) * -999
            events['lepton_Et'] = np.ones_like(events.W2_pt) * -999
            
            # for W1 and W2
            # condition is jet_1_pt > 0 and jet_2_pt > 0
            condition_12_jets = np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0)
            events['W1_pt'] = np.where(condition_12_jets, events.W1_pt, -999)
            events['W1_eta'] = np.where(condition_12_jets, events.W1_eta, -999)
            events['W1_phi'] = np.where(condition_12_jets, events.W1_phi, -999)
            events['W1_mass'] = np.where(condition_12_jets, events.W1_mass, -999)
            # condition is jet_3_pt > 0 and jet_4_pt > 0
            condition_34_jets = np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0)
            events['W2_pt'] = np.where(condition_34_jets, events.W2_pt, -999)
            events['W2_eta'] = np.where(condition_34_jets, events.W2_eta, -999)
            events['W2_phi'] = np.where(condition_34_jets, events.W2_phi, -999)
            events['W2_mass'] = np.where(condition_34_jets, events.W2_mass, -999)
            # for W1 and W2 dR
            events['W1_W2_dR'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_dR, -999)
            # for W1+W2
            events['W1_W2_mass'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_mass, -999)
            events['W1_W2_pt'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_pt, -999)
            events['W1_W2_eta'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_eta, -999)
            events['W1_W2_phi'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.W1_W2_phi, -999)
            # for W1 and W2 dR with Diphoton
            events['W1_diphoton_dR'] = np.where(condition_12_jets, events.W1_diphoton_dR, -999)
            events['W2_diphoton_dR'] = np.where(condition_34_jets, events.W2_diphoton_dR, -999)
            # for sum of two max bscore
            events['sum_two_max_bscores'] = np.where(np.logical_and(np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0), np.logical_and(events.jet_3_pt>0, events.jet_4_pt>0)), events.sum_two_max_bscores, -999)
            # for costhetastar
            events['costhetastar'] = np.where(np.logical_and(condition_12_jets, condition_34_jets), events.costhetastar, -999)
            # for costheta1 and costheta2
            events['costheta1'] = np.where(condition_12_jets, events.costheta1, -999)
            events['costheta2'] = np.where(condition_34_jets, events.costheta2, -999)
            events['electrons_all_1_pt'] = np.ones_like(events.W2_pt) * -999
            events['electrons_all_2_pt'] = np.ones_like(events.W2_pt) * -999
            events['electrons_all_1_eta'] = np.ones_like(events.W2_pt) * -999
            events['electrons_all_2_eta'] = np.ones_like(events.W2_pt) * -999
            events['electrons_all_1_phi'] = np.ones_like(events.W2_pt) * -999
            events['electrons_all_2_phi'] = np.ones_like(events.W2_pt) * -999
            events['electrons_all_1_mass'] = np.ones_like(events.W2_pt) * -999
            events['electrons_all_2_mass'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_1_pt'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_2_pt'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_1_eta'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_2_eta'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_1_phi'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_2_phi'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_1_mass'] = np.ones_like(events.W2_pt) * -999
            events['muons_all_2_mass'] = np.ones_like(events.W2_pt) * -999
        if is_FH_SL == "SL":
            # for lepton
            events['lepton_pt'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_pt'], -999)
            events['lepton_eta'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_eta'], -999)
            events['lepton_phi'] =  np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_phi'], -999)
            events['lepton_iso_MET_mt'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_iso_MET_mt'], -999)
            events['leptonpt_METpt'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['leptonpt_METpt'], -999)
            events['lepton_E'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_E'], -999)
            events['lepton_Et'] = np.where(np.logical_or(events.electron_iso_pt>0, events.muons_all_1_pt>0), events['lepton_Et'], -999)
            # for W1
            # condition is jet_1_pt > 0 and jet_2_pt > 0
            condition_12_jets = np.logical_and(events.jet_1_pt>0, events.jet_2_pt>0)
            events['W1_pt'] = np.where(condition_12_jets, events.W1_pt, -999)
            events['W1_eta'] = np.where(condition_12_jets, events.W1_eta, -999)
            events['W1_phi'] = np.where(condition_12_jets, events.W1_phi, -999)
            events['W1_mass'] = np.where(condition_12_jets, events.W1_mass, -999)
            # for W2 is all -999
            events['W2_pt'] = np.ones_like(events.W1_pt)*-999
            events['W2_eta'] = np.ones_like(events.W1_pt)*-999
            events['W2_phi'] = np.ones_like(events.W1_pt)*-999
            events['W2_mass'] = np.ones_like(events.W1_pt)*-999
            # for W1 and W2 dR
            events['W1_W2_dR'] = np.ones_like(events.W1_pt)*-999
            # for W1+W2
            events['W1_W2_mass'] = np.ones_like(events.W1_pt)*-999
            events['W1_W2_pt'] = np.ones_like(events.W1_pt)*-999
            events['W1_W2_eta'] = np.ones_like(events.W1_pt)*-999
            events['W1_W2_phi'] = np.ones_like(events.W1_pt)*-999
            # for W1 and W2 dR with Diphoton
            events['W1_diphoton_dR'] = np.where(condition_12_jets, events.W1_diphoton_dR, -999)
            events['W2_diphoton_dR'] = np.ones_like(events.W1_pt)*-999
            # for sum of two max bscore
            events['sum_two_max_bscores'] = np.ones_like(events.W1_pt)*-999
            # for costhetastar is all -999
            events['costhetastar'] = np.ones_like(events.W1_pt)*-999
            # for costheta1
            events['costheta1'] = np.where(condition_12_jets, events.costheta1, -999)
            # for costheta2 is all -999
            events['costheta2'] = np.ones_like(events.W1_pt)*-999
            events['jet_1_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
            events['jet_2_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
            events['jet_3_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
            events['jet_4_btagDeepFlavB'] = np.ones_like(events.W1_pt)*-999
            
        return events
    def get_leptons_variables(events):
        events['nGoodisoleptons'] = events['nGoodisoleptons']
        events['nGoodnonisoleptons'] = events['nGoodnonisoleptons']
        events['nGoodisoelectrons'] = events['nGoodisoelectrons']
        events['nGoodnonisoelectrons'] = events['nGoodnonisoelectrons']
        events['nGoodisomuons'] = events['nGoodisomuons']
        events['nGoodnonisomuons'] = events['nGoodnonisomuons']
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
        category_cut = ((events["category"]==3) | (events["category"]==4))
        photonID_cut = (events["Diphoton_minID_modified"]>-0.7)
        events = events[ category_cut & photonID_cut]
        events['mx'] = np.ones(len(events))*int(mx)
        events['my'] = np.ones(len(events))*int(my)
        events = get_photon_variables(events)
        events = get_leptons_variables(events)
        events = get_met_variables(events)
        eventsFH=events[events["category"]==4]
        eventsSL=events[events["category"]==3]
        eventsFH = get_FHSL_jet_W_variables(eventsFH, "FH")
        eventsFH = deal_with_none_object(eventsFH, "FH")
        eventsSL = get_FHSL_jet_W_variables(eventsSL, "SL")
        eventsSL = deal_with_none_object(eventsSL, "SL")
        events = ak.concatenate([eventsFH, eventsSL])
        return events    
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
    PNN_score = model_predict(events_sig, model, loaded_scaler, input_features)
    events_sig['PNN_score'] = PNN_score
    print('sig events:', len(events_sig))
    PNN_score = model_predict(events_bbgg, model, loaded_scaler, input_features)
    events_bbgg['PNN_score'] = PNN_score
    print('bbgg events:', len(events_bbgg))
    PNN_score = model_predict(events_zzgg, model, loaded_scaler, input_features)
    events_zzgg['PNN_score'] = PNN_score
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
    print(i)
    result=process_sig_samples(signal_samples['FHpath'][i], signal_samples['SLpath'][i], signal_samples['BBGGpath'][i], signal_samples['ZZggpath'][i], signal_samples['sig_output_name'][i], signal_samples['bbgg_output_name'][i], signal_samples['zzgg_output_name'][i], input_features, cut1, cut2)
    print('high purity signal:', result)
def process_highpurity_sigfile(file):
    signal = file.split("/")[-1].split("_highpurity")[0].split("merge")[0]
    signaltype = "combineFHSL"
    name=file.split("/")[-1].split('merged_')[-1]
    if "down" in name:
        sys=name.split("_down")[0]
    elif "up" in name:
        sys=name.split("_up")[0]

    if "down" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity"
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
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity"
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
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34highpurity"
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
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity"
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
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity"
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
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity_" + sys + "Down01sigma"
    elif "up" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity_" + sys + "Up01sigma"
    elif "nominal" in file.split("/")[-1]:
        tree_name = "gghh_125_13TeV_" + signaltype + "_cat34lowpurity"
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
    highpurity_sigfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/CombineFHSL_MX"+Xmass+"_MH125_cat34_m*_highpurity.parquet")
    lowpurity_sigfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/CombineFHSL_MX"+Xmass+"_MH125_cat34_m*_lowpurity.parquet")
    lowpurity_bbggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/BBGG_MX"+Xmass+"_MH125_cat34_m*_lowpurity.parquet")
    highpurity_bbggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/BBGG_MX"+Xmass+"_MH125_cat34_m*_highpurity.parquet")
    highpurity_zzggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/ZZGG_MX"+Xmass+"_MH125_cat34_m*_highpurity.parquet")
    lowpurity_zzggfiles = glob.glob("./PBDT_HH_FHSL_combine_"+year+"/ZZGG_MX"+Xmass+"_MH125_cat34_m*_lowpurity.parquet")
    
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
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_combineFHSL_cat34highpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/CombineFHSL_MX' + Xmass + '*highpurity.root'
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
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_combineFHSL_cat34lowpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/CombineFHSL_MX' + Xmass + '*lowpurity.root'
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
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_bbgg_cat34highpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/BBGG_MX' + Xmass + '*highpurity.root'
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
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_bbgg_cat34lowpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/BBGG_MX' + Xmass + '*lowpurity.root'
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
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_zzgg_cat34highpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/ZZGG_MX' + Xmass + '*highpurity.root'
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
    command = 'hadd ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/MX'+Xmass+'_MH125/CombineFHSL_MX'+Xmass+'_MH125_'+year+'_zzgg_cat34lowpurity.root ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/ZZGG_MX' + Xmass + '*lowpurity.root'
    print(command)
    os.system(command)
    command = 'rm ./PBDT_HH_FHSL_combine_'+year+'/flashgginput/ZZGG_MX'+Xmass+'*lowpurity.root'
    os.system(command)
    boundary={str(Xmass):{"cut1":cut1,"cut2":cut2,"FHSL_highpurity":FHSL_highpurity,"FHSL_lowpurity":FHSL_lowpurity,"BBGG_highpurity":bbgg_highpurity,"BBGG_lowpurity":bbgg_lowpurity,"ZZGG_highpurity":zzgg_highpurity,"ZZGG_lowpurity":zzgg_lowpurity,"highpurity_sigeff":highpurity_sigeff,"lowpurity_sigeff":lowpurity_sigeff,"highpurity_sidebandnum":highpurity_sidebandnum,"lowpurity_sidebandnum":lowpurity_sidebandnum}}
    with open("./PBDT_HH_FHSL_combine_"+year+"/flashgginput/MX"+Xmass+"_MH125/boundaries.json", "w") as f:
        json.dump(boundary, f, indent=4)
    plt.savefig("./PBDT_HH_FHSL_combine_"+year+"/flashgginput/MX"+Xmass+"_MH125/dnnscore.png", dpi=140)
