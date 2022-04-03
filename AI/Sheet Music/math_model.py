import librosa
from utils import createMIDI

def librosaModel(song, rate, instrument, Prob_threshold = 0.5):

    """=================Below functions use math to estimate the pitch================="""
    
    def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
        # Compute autocorrelation of input segment.
        r = librosa.autocorrelate(segment)
        
        # Define lower and upper limits for the autocorrelation argmax.
        i_min = sr/fmax
        i_max = sr/fmin
        r[:int(i_min)] = 0
        r[int(i_max):] = 0
        
        # Find the location of the maximum autocorrelation.
        i = r.argmax()
        f0 = float(sr)/i
        return f0

    
    def generate_sine(f0, sr, n_duration):
        # Generates sine function for the note
        n = np.arange(n_duration)
        return 0.2*np.sin(2*np.pi*f0*n/float(sr))

    
    def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
        n0 = onset_samples[i]
        n1 = onset_samples[i+1]
        f0 = estimate_pitch(x[n0:n1], sr)
        return generate_sine(f0, sr, n1-n0)
    
    """=================Above functions use math to estimate the pitch================="""

    # Compresses audio into one channel
    audio = librosa.to_mono(song)

    # Padding of 1 frame to capture onset
    audio = np.hstack([[0], audio])

    # Retrieves the head of notes
    onset_boundaries = librosa.onset.onset_detect(
        audio,
        sr=rate, units='samples', 
        backtrack = False,
    )

    # Generates a sine wave for each segment (Head of note to head of next note)
    y = np.concatenate([
        estimate_pitch_and_generate_sine(audio, onset_boundaries, i, sr=rate)
        for i in range(len(onset_boundaries)-1)
    ])

    # Normalise the Constant-Q Transform of the audio
    C = np.abs(librosa.cqt(y))
    C = (C - C.min()) / (C.max() - C.min())

    # Set only 1 note to be played at any time
    C = np.apply_along_axis(lambda x: ((x == x.max()) & (x > actlProb_threshold)), axis=1, arr=C.T)

    # Create the MIDI Object to use
    return createMIDI(song = y, rate = rate, instrument = instrument, actlProb = C, actlProb_threshold = 0.9)