import librosa
import pretty_midi as pm
import numpy as np

def getKey(song, rate):
    """
    From Kumhansl and Schmuckler as reported here:
    http://rnhart.net/articles/key-finding/

    This returns the most probable key the song is in
    """

    chroma = librosa.feature.chroma_cqt(song, rate).sum(1)
    major = [np.corrcoef(chroma, np.roll([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], i))[0, 1] for i in range(12)]
    minor = [np.corrcoef(chroma, np.roll([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], i))[0, 1] for i in range(12)]
    keySignature = (['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'][
        major.index(max(major)) if max(major) > max(minor) else minor.index(max(minor)) - 3]
                    + ('m' if max(major) < max(minor) else ''))
    return keySignature

# Idea: 2 Transformers, 1 to determine note start, 1 to determine note presence
def createMIDI(song, rate, instrument, actlProb, onProb = None, volProb = None, actlProb_threshold = 0.5, onProb_threshold = 0.5):    
    """Given the Model outputs, creates the corresponding MIDI Object
    Rows are the time frames and columns are the different pitch
    """

    # Default Volume
    if volProb == None: volProb = np.ones(actlProb.shape)

    # Creates MIDI Object and sets tempo (Assuming Constant)
    midi = pm.PrettyMIDI(initial_tempo=librosa.beat.tempo(song, rate).mean())

    # Set the key signature (Which should be constant for the entire song)
    midi.key_signature_changes += [pm.KeySignature(
        pm.key_name_to_key_number(getKey(song, rate)), 
        0
    )]

    # Create the track of the instrument currently playing
    track = pm.Instrument(program=pm.instrument_name_to_program(instrument), name=instrument)
    midi.instruments += [track]

    intervals, frameLenSecs = {}, librosa.frames_to_time(1, rate) # Time is in absolute seconds, not relative MIDI ticks
    onsets = (onProb > onProb_threshold).astype(np.int8) if onProb != None else None # Gets Onsets based on predicted probability
    frames = (actlProb > actlProb_threshold).astype(np.int8) # Ensure that any frame with an onset prediction is considered active.
    if onsets != None: frames = frames | onsets

    # End the pitch, adding the note to the MIDI object
    def EndPitch(pitch, endFrame):
        # Add note interval to MIDI object
        track.notes += [pm.Note(int(volProb[intervals[pitch], pitch] * 80 + 10), pitch + 24,
                                intervals[pitch] * frameLenSecs, endFrame * frameLenSecs)]
                                
        # To remove indication that it has an interval head
        del intervals[pitch]


    # Added silent frame at the end to terminate any still active notes
    # For every frame, at position i
    for i, frame in enumerate(np.vstack([frames, np.zeros(frames.shape[1])])):
        # For every pitch in the frame, and its active value
        for pitch, active in enumerate(frame):
            if active: # If played

                # If pitch has no head recorded
                if pitch not in intervals:
                    if onsets is None: intervals[pitch] = i # Insert it if no onset detected
                    elif onsets[i, pitch]: intervals[pitch] = i # Start a note only if we have predicted an onset
                    # Else: Even though the frame is active, there is no onset, so ignore it
                elif onsets is not None: # If there onset detected and pitch is recorded
                    if (onsets[i, pitch] and not onsets[i - 1, pitch]):
                        EndPitch(pitch, i)   # Pitch is already active, but because of a new onset, we should end the note
                        intervals[pitch] = i # and start a new one

                # Else: it is ignored as it is not tail or head of note
                    
            elif pitch in intervals: EndPitch(pitch, i) # If not played

    if track.notes: assert len(frames) * frameLenSecs >= track.notes[-1].end, 'Wrong MIDI sequence duration'
    return midi

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