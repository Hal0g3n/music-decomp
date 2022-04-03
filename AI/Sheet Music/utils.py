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