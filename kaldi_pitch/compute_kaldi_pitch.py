# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:28:49 2017

@author: ganesh
"""

import os
import sys
import numpy as np
from subprocess import call
import scipy.io.wavfile
import HTK

def compute_kaldi_pitch(infile):
# Computes the Kaldi pitch features and returns two variables
# Usage: raw_pitch, proc_pitch = compute_kaldi_pitch(infile)
# infile = Full path to the wav file you want to extract pitch
# raw_pitch = (n_frames x 2) -- [NCCF smooth_pitch]
# proc_pitch = (n_frames x 3) -- [prob_of_voicing normalized_pitch pitch_deltas]
    fs, sig = scipy.io.wavfile.read(infile)
    fnm = os.path.basename(infile).split('.')[0]
    cmd = "echo "+fnm+" "+infile+" > ./temp.scp"
    os.system(cmd)
    os.system("mkdir tempdir")
    fext_cmd = "compute-kaldi-pitch-feats --sample-frequency="+str(fs)+ \
                " scp:./temp.scp ark:- | process-kaldi-pitch-feats ark:- ark:- |"+ \
                " copy-feats-to-htk --output-dir=tempdir --output-ext=htk ark:-"
    outcome = os.system(fext_cmd)
    print outcome
    htk_file = HTK.HTKFile()
    htk_file.load("./tempdir/"+fnm+".htk")
    proc_pitch = np.array(htk_file.data)
    
    fext_cmd = "compute-kaldi-pitch-feats --sample-frequency="+str(fs)+ \
                " scp:./temp.scp ark:- |"+ \
                " copy-feats-to-htk --output-dir=tempdir --output-ext=htk ark:-"
    os.system(fext_cmd)
    htk_file = HTK.HTKFile()
    htk_file.load("./tempdir/"+fnm+".htk")
    raw_pitch = np.array(htk_file.data)
    c = raw_pitch[:,0]
    l = -5.2+ 5.4*np.exp(7.5*(np.abs(c) - 1)) + 4.8*np.abs(c) - 2*np.exp(-10*np.abs(c)) + 4.2* np.exp(20*(np.abs(c) -1))
    pov = 1/(1+np.exp(-l))
    proc_pitch[:,0] = pov    
    return raw_pitch, proc_pitch
    
if __name__ =="__main__":
    infile = "../test.wav"
    raw_f0, proc_f0 = compute_kaldi_pitch(infile)