# This function converts a floating point array [-1,1] to 16bit PCM array
# This is used to write PCM format wav files from numpy float arrays
def float2pcm16(f):    
    f = f * 32768 ;
    f[f > 32767] = 32767;
    f[f < -32768] = -32768;
    i = np.int16(f)
    return i
