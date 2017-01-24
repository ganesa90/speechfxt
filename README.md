# speechfxt
Feature extraction pipeline for various commonly used speech features
###############
Usage:
Job filelist- This is the list of wav files for which you want features extracted and 
              the destination paths for the extracted features
              Format of the filelist:
                  <Full/Path/to/wav/file>,<Path/to/output/file>                  
config file - Look at the example_config.conf for the sample config file
              The config file specifies the feature type, window length, frame shift etc. for feature extraction

Running the code:
python extractFeatures.py <job_filelist> <config_file>

###############
Output format
The extracted features are saved in the HTK format.
