# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 23:20:41 2016

@author: ganesh
"""

import sys
import numpy as np
import scipy
import scipy.io
from scipy import signal
import os
import librosa
import pickle
import ConfigParser as cp
from writehtk import writehtk
from gammatone import gtgram

class Error(Exception):
   """Base class for other exceptions"""
   pass

class ConfigError(Error):
   """Raised when a config setting is not found"""
   pass


def mvnormalize(feats, mvn_params, std_frac):
    f_mean = mvn_params[:,0, None]
    f_std = mvn_params[:,1, None]
    feats_norm = std_frac*(feats - f_mean)/f_std
    return feats_norm

def check_files(filelist):
    fp = open(filelist,'r')
    flist = fp.read().splitlines()    
    flist = filter(None, flist)
    for iter1,fline in enumerate(flist):
        infnm = fline.split(',')[0]
        opfnm = fline.split(',')[1]
        if  not(os.path.isfile(opfnm)):
            print('Error: Feature file '+opfnm+' not created')
        if not(os.path.getsize(opfnm) > 0):
            print('Error: Feature file '+opfnm+' is empty')

def mfcc_wrapper(y, sr, win_length, hop_length, window, n_mfcc=13, n_mels=40, 
                 n_fft = None, center=True):
    if n_fft is None:
        n_fft = int(2**(np.ceil(np.log(win_length)/np.log(2))))
    stft = librosa.core.stft(y=y, win_length=win_length, window=window,
                             hop_length=hop_length, center=center, n_fft=n_fft)
    powspec = np.abs(stft)**2
    melspec = librosa.feature.melspectrogram(S=powspec, hop_length=hop_length,
                                             n_fft=n_fft, n_mels=n_mels)
    mfcc = librosa.feature.mfcc(S=librosa.logamplitude(melspec), n_mfcc=n_mfcc)
    return np.real(mfcc)

def melspect_wrapper(y, sr, win_length, hop_length, window, n_mfcc=13, n_mels=40, 
                 n_fft = None, center=True):
    if n_fft is None:
        n_fft = int(2**(np.ceil*(np.log(win_length)/np.log(2))))
    stft = librosa.core.stft(y=y, win_length=win_length, window=window,
                             hop_length=hop_length, center=center, n_fft=n_fft)
    powspec = np.abs(stft)**2
    melspec = librosa.feature.melspectrogram(S=powspec, hop_length=hop_length,
                                             n_fft=n_fft, n_mels=n_mels)    
    return np.real(melspec)

    
def get_mfcc(filelist, config):
    # Read the filelist
    fp = open(filelist,'r')
    flist = fp.read().splitlines()
    flist = filter(None, flist)
    # Create output directory if non-existant
    opdir = os.path.dirname(flist[0].split(',')[1])
    if not os.path.exists(opdir):
        os.makedirs(opdir)
    # Read the relevant configs from the configfile    
    framelen = float(config['framelen'])
    frameshift = float(config['frameshift'])    
    wintype = config['wintype']
    n_mels = int(config['nbanks'])
    if wintype == 'rectangular':
        winfun = np.ones
    else:
        winfun = getattr(np, wintype)
    ncoef = int(config['ncoef'])
    mvn = config['mvn']
    mvn = mvn.upper() == 'TRUE'
    if 'std_frac' in config:
        std_frac = float(config['std_frac'])
    else:
        std_frac = 1.0
            
    # Iterate over the filelist to extract features
    if mvn:
        print ("Computing the mean and variance of features")
        feats_list = []
        for iter1,fline in enumerate(flist):
            infnm = fline.split(',')[0]
            opfnm = fline.split(',')[1]
            sig, fs = librosa.load(infnm, sr=None)
            sig = sig/max(abs(sig))
            dither = 1e-6*np.random.rand(sig.shape[0])
            sig = sig + dither
            win_length=int(fs*framelen*0.001)
            hop_length = int(fs*frameshift*0.001)
            feats = mfcc_wrapper(y=sig, sr=fs, n_mfcc=ncoef, 
            win_length=win_length, n_mels=n_mels, hop_length=hop_length,
            center=True, window=winfun(win_length))
            feats_list.append(feats)
        all_feats = np.concatenate(feats_list, axis=1)
        f_mean = np.mean(all_feats, axis=1)[:, None]
        f_std = np.std(all_feats, axis=1)[:, None]
        opdir = os.path.dirname(opfnm)
        mvn_params = np.concatenate((f_mean, f_std), axis=1)
        postfix = os.path.basename(filelist).split('.')[0]
        np.save(opdir+'/mvn_params_'+postfix+'.npy', mvn_params)     
        
    for iter1,fline in enumerate(flist):
        infnm = fline.split(',')[0]
        opfnm = fline.split(',')[1]
        sig, fs = librosa.load(infnm, sr=None)
        sig = sig/max(abs(sig))
        dither = 1e-6*np.random.rand(sig.shape[0])
        sig = sig + dither
        win_length=int(fs*framelen*0.001)
        hop_length = int(fs*frameshift*0.001)
        feats = mfcc_wrapper(y=sig, sr=fs, n_mfcc=ncoef, n_mels=n_mels,
        hop_length=hop_length, win_length=win_length,
        center=True, window=winfun(win_length))
        if mvn:
            feats = mvnormalize(feats, mvn_params, std_frac)
        writehtk(feats.T, frameshift, opfnm)
    fp.close()
    
def get_melspect(filelist, config):
    # Read the filelist
    fp = open(filelist,'r')
    flist = fp.read().splitlines()
    flist = filter(None, flist)
    # Create output directory if non-existant
    opdir = os.path.dirname(flist[0].split(',')[1])
    if not os.path.exists(opdir):
        os.makedirs(opdir)
    # Read the relevant configs from the configfile    
    framelen = float(config['framelen'])
    frameshift = float(config['frameshift'])    
    wintype = config['wintype']
    n_mels = int(config['nbanks'])
    if wintype == 'rectangular':
        winfun = np.ones
    else:
        winfun = getattr(np, wintype)    
    mvn = config['mvn']
    mvn = mvn.upper() == 'TRUE'
    if 'std_frac' in config:
        std_frac = float(config['std_frac'])
    else:
        std_frac = 1.0
    # Iterate over the filelist to extract features
    if mvn:
        feats_list = []
        for iter1,fline in enumerate(flist):
            infnm = fline.split(',')[0]
            opfnm = fline.split(',')[1]
            sig, fs = librosa.load(infnm, sr=None)
            sig = sig/max(abs(sig))
            dither = 1e-6*np.random.rand(sig.shape[0])
            sig = sig + dither
            win_length=int(fs*framelen*0.001)
            hop_length = int(fs*frameshift*0.001)
            feats = melspect_wrapper(y=sig, sr=fs,
            win_length=win_length, n_mels=n_mels, hop_length=hop_length,
            center=True, window=winfun(win_length))
            feats_list.append(feats)
        all_feats = np.concatenate(feats_list, axis=1)
        f_mean = np.mean(all_feats, axis=1)[:, None]
        f_std = np.std(all_feats, axis=1)[:, None]
        opdir = os.path.dirname(opfnm)
        mvn_params = np.concatenate((f_mean, f_std), axis=1)
        postfix = os.path.basename(filelist).split('.')[0]
        np.save(opdir+'/mvn_params_'+postfix+'.npy', mvn_params)    

        
    for iter1,fline in enumerate(flist):
        infnm = fline.split(',')[0]
        opfnm = fline.split(',')[1]
        sig, fs = librosa.load(infnm, sr=None)
        sig = sig/max(abs(sig))
        dither = 1e-6*np.random.rand(sig.shape[0])
        sig = sig + dither
        win_length=int(fs*framelen*0.001)
        hop_length = int(fs*frameshift*0.001)
        feats = melspect_wrapper(y=sig, sr=fs, n_mels=n_mels,
        hop_length=hop_length, win_length=win_length,
        center=True, window=winfun(win_length))
        if mvn:
            feats = mvnormalize(feats, mvn_params, std_frac)
        writehtk(feats.T, frameshift, opfnm)
    fp.close()
    
def get_powerspect(filelist, config):
    # Read the filelist
    fp = open(filelist,'r')
    flist = fp.read().splitlines()
    flist = filter(None, flist)
    # Create output directory if non-existant
    opdir = os.path.dirname(flist[0].split(',')[1])
    if not os.path.exists(opdir):
        os.makedirs(opdir)
    # Read the relevant configs from the configfile    
    framelen = float(config['framelen'])
    frameshift = float(config['frameshift'])    
    wintype = config['wintype']    
    if wintype == 'rectangular':
        winfun = np.ones
    else:
        winfun = getattr(np, wintype)    
    mvn = config['mvn']
    mvn = mvn.upper() == 'TRUE'
    if 'std_frac' in config:
        std_frac = float(config['std_frac'])
    else:
        std_frac = 1.0
    # Iterate over the filelist to extract features
    if mvn:
        feats_list = []
        for iter1,fline in enumerate(flist):
            infnm = fline.split(',')[0]
            opfnm = fline.split(',')[1]
            sig, fs = librosa.load(infnm, sr=None)
            sig = sig/max(abs(sig))
            dither = 1e-6*np.random.rand(sig.shape[0])
            sig = sig + dither
            win_length=int(fs*framelen*0.001)
            hop_length = int(fs*frameshift*0.001)
            stft = librosa.core.stft(y=sig, win_length=win_length, 
                                     window=winfun(win_length),
                                     hop_length=hop_length, center=True, 
                                     n_fft=None)
            feats = np.abs(stft)**2
            # Code for amplitude range compression
            if config['compression'] == 'log':
                feats = librosa.logamplitude(feats)
            elif config['compression'][0:4] == 'root':
                rootval = float(config['compression'].split('_')[1])                
                feats = np.sign(feats)*(np.abs(feats)**(1/rootval))
                if np.sum(np.isnan(feats)):
                    print('NaN Error in root compression for file: %s' %infnm)
                    exit()
            feats_list.append(feats)
        all_feats = np.concatenate(feats_list, axis=1)
        f_mean = np.mean(all_feats, axis=1)[:, None]
        f_std = np.std(all_feats, axis=1)[:, None]
        opdir = os.path.dirname(opfnm)
        mvn_params = np.concatenate((f_mean, f_std), axis=1)
        postfix = os.path.basename(filelist).split('.')[0]
        np.save(opdir+'/mvn_params_'+postfix+'.npy', mvn_params)    

        
    for iter1,fline in enumerate(flist):
        infnm = fline.split(',')[0]
        opfnm = fline.split(',')[1]
        sig, fs = librosa.load(infnm, sr=None)
        sig = sig/max(abs(sig))
        dither = 1e-6*np.random.rand(sig.shape[0])
        sig = sig + dither
        win_length=int(fs*framelen*0.001)
        hop_length = int(fs*frameshift*0.001)
        stft = librosa.core.stft(y=sig, win_length=win_length, 
                                     window=winfun(win_length),
                                     hop_length=hop_length, center=True, 
                                     n_fft=None)
        feats = np.abs(stft)**2
        if config['compression'] == 'log':
                feats = librosa.logamplitude(feats)
        elif config['compression'][0:4] == 'root':
            rootval = float(config['compression'].split('_')[1])                
            feats = np.sign(feats)*(np.abs(feats)**(1/rootval))
            if np.sum(np.isnan(feats)):
                print('NaN Error in root compression for file: %s' %infnm)
                exit()
        if mvn:
            feats = mvnormalize(feats, mvn_params, std_frac)
        writehtk(feats.T, frameshift, opfnm)
    fp.close()

def get_waveform(filelist, config):
    # Read the filelist
    fp = open(filelist,'r')
    flist = fp.read().splitlines()
    flist = filter(None, flist)
    # Create output directory if non-existant
    opdir = os.path.dirname(flist[0].split(',')[1])
    if not os.path.exists(opdir):
        os.makedirs(opdir)
    # Read the relevant configs from the configfile    
    framelen = float(config['framelen'])
    frameshift = float(config['frameshift'])    
    wintype = config['wintype']    
    if wintype == 'rectangular':
        winfun = np.ones
    else:
        winfun = getattr(np, wintype)    
    mvn = config['mvn']
    mvn = mvn.upper() == 'TRUE'
    if 'std_frac' in config:
        std_frac = float(config['std_frac'])
    else:
        std_frac = 1.0
    # Iterate over the filelist to extract features        
    for iter1,fline in enumerate(flist):
        infnm = fline.split(',')[0]
        opfnm = fline.split(',')[1]
        sig, fs = librosa.load(infnm, sr=None)
        sig = sig/max(abs(sig))
        dither = 1e-6*np.random.rand(sig.shape[0])
        sig = sig + dither
        win_length=int(fs*framelen*0.001)
        hop_length = int(fs*frameshift*0.001)
        feats = librosa.util.frame(y, frame_length=win_length, hop_length=hop_length)
        # Code for amplitude range compression
        if mvn:
            frame_mean = np.mean(feats, axis=0)[None, :]
            frame_std = np.std(feats, axis=0)[None, :]
            feats = std_frac*(feats - frame_mean)/frame_std   
        writehtk(feats.T, frameshift, opfnm)
    fp.close()

def get_gfb(filelist, config):
    # Read the filelist
    fp = open(filelist,'r')
    flist = fp.read().splitlines()
    flist = filter(None, flist)
    # Create output directory if non-existant
    opdir = os.path.dirname(flist[0].split(',')[1])
    if not os.path.exists(opdir):
        os.makedirs(opdir)
    # Read the relevant configs from the configfile    
    framelen = float(config['framelen'])
    frameshift = float(config['frameshift'])    
    wintype = config['wintype']    
    if wintype == 'rectangular':
        winfun = np.ones
    else:
        winfun = getattr(np, wintype)
    # Number of channels for gammatone filterbank
    if 'nbanks' in config:
        nbanks = int(config['nbanks'])
    else:
        raise ConfigError('nbanks parameter not set in config file')
    # Min frequency of Gammatone filterbank
    if 'min_freq' in config:
        min_freq = float(config['min_freq'])
    else:
        min_freq = 0            
    mvn = config['mvn']
    mvn = mvn.upper() == 'TRUE'
    if 'std_frac' in config:
        std_frac = float(config['std_frac'])
    else:
        std_frac = 1.0
    # Iterate over the filelist to extract features
    if mvn:
        feats_list = []
        for iter1,fline in enumerate(flist):
            infnm = fline.split(',')[0]
            opfnm = fline.split(',')[1]
            sig, fs = librosa.load(infnm, sr=None)
            sig = sig/max(abs(sig))
            dither = 1e-6*np.random.rand(sig.shape[0])
            sig = sig + dither
            win_length=int(fs*framelen*0.001)
            hop_length = int(fs*frameshift*0.001)
            feats = gtgram.gtgram(sig, fs, framelen*0.001, frameshift*0.001, 
                                  nbanks, min_freq)            
            # Code for amplitude range compression
            if config['compression'] == 'log':
                feats = librosa.logamplitude(feats)
            elif config['compression'][0:4] == 'root':
                rootval = float(config['compression'].split('_')[1])                
                feats = np.sign(feats)*(np.abs(feats)**(1/rootval))
                if np.sum(np.isnan(feats)):
                    print('NaN Error in root compression for file: %s' %infnm)
                    exit()
            feats_list.append(feats)
        all_feats = np.concatenate(feats_list, axis=1)
        f_mean = np.mean(all_feats, axis=1)[:, None]
        f_std = np.std(all_feats, axis=1)[:, None]
        opdir = os.path.dirname(opfnm)
        mvn_params = np.concatenate((f_mean, f_std), axis=1)
        postfix = os.path.basename(filelist).split('.')[0]
        np.save(opdir+'/mvn_params_'+postfix+'.npy', mvn_params)    

        
    for iter1,fline in enumerate(flist):
        infnm = fline.split(',')[0]
        opfnm = fline.split(',')[1]
        sig, fs = librosa.load(infnm, sr=None)
        sig = sig/max(abs(sig))
        dither = 1e-6*np.random.rand(sig.shape[0])
        sig = sig + dither
        win_length=int(fs*framelen*0.001)
        hop_length = int(fs*frameshift*0.001)
        feats = gtgram.gtgram(sig, fs, framelen*0.001, frameshift*0.001, 
                                  nbanks, min_freq)        
        if config['compression'] == 'log':
                feats = librosa.logamplitude(feats)
        elif config['compression'][0:4] == 'root':
            rootval = float(config['compression'].split('_')[1])                
            feats = np.sign(feats)*(np.abs(feats)**(1/rootval))
            if np.sum(np.isnan(feats)):
                print('NaN Error in root compression for file: %s' %infnm)
                exit()
        if mvn:
            feats = mvnormalize(feats, mvn_params, std_frac)
        writehtk(feats.T, frameshift, opfnm)
    fp.close()

    
def extractFeatures(filelist, configfile):
    configfp = cp.RawConfigParser()
    print('Reading the config file')
    configfp.read(configfile)      
    sect = configfp.sections()[0]
    config_dict = dict(configfp.items(sect))
    ftype = config_dict['feature']
    featExt_opt = {'mfcc': get_mfcc, 'melspect': get_melspect, 
                   'powerspect': get_powerspect, 'waveform': get_waveform,
                   'gfb': get_gfb}
    print('Extracting features')
    featExt_opt[ftype](filelist, config_dict)
    print('Checking the feature files')
    check_files(filelist)

if __name__=="__main__":
#    extractFeatures('test_flist_featExt.txt', 'example_config.conf')
    extractFeatures(sys.argv[1], sys.argv[2])
    
    
