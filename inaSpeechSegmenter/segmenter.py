#!/usr/bin/env python
# encoding: utf-8

# The MIT License

# Copyright (c) 2018 Ina (David Doukhan - http://www.ina.fr/)

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Segmenter Module (segmenter.py)
--------------------------------
This module implements the core segmentation functionality for inaSpeechSegmenter. It converts media files into
16kHz mono WAV signals, extracts mel spectrogram and MFCC features, and applies a two-stage segmentation process:
first, an energy-based activity detection, and then a DNN-based segmentation with Viterbi post-processing.
The module divides the audio stream into labeled segments (e.g., speech, music, noise, and gender-specific speech),
which is essential for tasks like speaker diarization and voice activity detection. Its role in partitioning audio into
meaningful segments is why the file is aptly named segmenter.py.
"""


import os
import sys

import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import get_file
from .thread_returning import ThreadReturning

import shutil
import time
import random
import gc

from skimage.util import view_as_windows as vaw


from .pyannote_viterbi import viterbi_decoding
from .viterbi_utils import pred2logemission, diag_trans_exp, log_trans_exp
from .remote_utils import get_remote

from .io import media2sig16kmono
from .sidekit_mfcc import mfcc
import warnings

from .export_funcs import seg2csv, seg2textgrid

def _media2feats(medianame, tmpdir, start_sec, stop_sec, ffmpeg):
    """
    Converts a media file into feature representations for segmentation.

    This function processes an input media file (audio or video) by converting it into a 16kHz mono WAV signal and
    then extracting its mel spectrogram (mspec) and MFCC-derived log energy (loge) features. These "feats" (features)
    are used by the segmentation pipeline to detect activity and classify segments (e.g., speech, music, noise, or gender).
    Additionally, if the extracted log energy has fewer than 68 frames (about 720 milliseconds), the function pads the
    mel spectrogram accordingly to ensure robust segmentation.

    Returns:
        mspec: 2D numpy array representing the mel spectrogram features.
        loge: 1D numpy array of log energy values derived from the MFCC extraction.
        difflen: integer indicating the number of frames added as padding due to short signal duration.
    """
    
    sig = media2sig16kmono(medianame, tmpdir, start_sec, stop_sec, ffmpeg, 'float32')
    with warnings.catch_warnings():
        # ignore warnings resulting from empty signals parts
        warnings.filterwarnings('ignore', message='divide by zero encountered in log', category=RuntimeWarning)
        _, loge, _, mspec = mfcc(sig.astype(np.float32), get_mspec=True)

    # Management of short duration segments
    difflen = 0
    if len(loge) < 68:
        difflen = 68 - len(loge)
        warnings.warn("media %s duration is short. Robust results require length of at least 720 milliseconds" % medianame)
        mspec = np.concatenate((mspec, np.ones((difflen, 24)) * np.min(mspec)))

    return mspec, loge, difflen

def _energy_activity(loge, ratio):
    """
    Computes an energy-based activity mask from the log energy features, which is used for voice activity detection.
    
    This function calculates a threshold based on the mean of finite log energy values plus the logarithm of the provided ratio.
    It then creates a binary mask (raw_activity) that marks frames with log energy above the threshold as active.
    Finally, Viterbi decoding is applied to the log-transformed emission probabilities (obtained from the binary mask) 
    using a predefined transition cost. This process smooths the activity detection by enforcing temporal consistency, 
    ultimately aiding in segmenting the audio into active and inactive (or energy and no-energy) regions.
    """
    threshold = np.mean(loge[np.isfinite(loge)]) + np.log(ratio)
    raw_activity = (loge > threshold)
    return viterbi_decoding(pred2logemission(raw_activity),
                            log_trans_exp(150, cost0=-5))


def _get_patches(mspec, w, step):
    """
    Extracts and normalizes overlapping patches from a mel spectrogram for DNN-based segmentation.

    This function divides the input mel spectrogram (mspec) into overlapping patches of size (w, h) using a sliding window,
    where h is the number of mel bands. Each patch is flattened and normalized (zero mean, unit variance). Padding is added 
    at the boundaries to ensure the patch sequence aligns with the original spectrogram's temporal structure. It returns 
    the patches (reshaped to (n_patches, w, h)) along with a boolean array indicating which patches contain only finite values.
    
    These patches serve as the input features for the segmentation neural network, enabling the model to capture local 
    temporal and spectral characteristics crucial for distinguishing between different audio segments.
    """
    h = mspec.shape[1]
    data = vaw(mspec, (w,h), step=step)
    data.shape = (len(data), w*h)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in subtract', category=RuntimeWarning)
        data = (data - np.mean(data, axis=1).reshape((len(data), 1))) / np.std(data, axis=1).reshape((len(data), 1))
    lfill = [data[0,:].reshape(1, h*w)] * (w // (2 * step))
    rfill = [data[-1,:].reshape(1, h*w)] * (w // (2* step) - 1 + len(mspec) % 2)
    data = np.vstack(lfill + [data] + rfill )
    finite = np.all(np.isfinite(data), axis=1)
    data.shape = (len(data), w, h)
    return data, finite


def _binidx2seglist(binidx):
    """
    ss._binidx2seglist((['f'] * 5) + (['bbb'] * 10) + ['v'] * 5)
    Out: [('f', 0, 5), ('bbb', 5, 15), ('v', 15, 20)]
    
    #TODO: is there a pandas alternative??
    """
    curlabel = None
    bseg = -1
    ret = []
    for i, e in enumerate(binidx):
        if e != curlabel:
            if curlabel is not None:
                ret.append((curlabel, bseg, i))
            curlabel = e
            bseg = i
    ret.append((curlabel, bseg, i + 1))
    return ret


class DnnSegmenter:
    """
    DnnSegmenter is an abstract base class for performing deep neural network-based segmentation 
    of audio signals using pre-trained Keras models. It leverages mel spectrogram features (up to 24 bands) 
    generated via the SIDEKIT framework and applies a two-stage segmentation process: first, an initial segmentation 
    is provided (e.g., via energy detection), and then the DNN refines the segmentation through patch-based analysis 
    and Viterbi decoding for temporal smoothing.

    Child classes must define the following class attributes:
      - nmel: number of mel bands to use (maximum is 24).
      - viterbi_arg: parameter for Viterbi post-processing that influences state transition smoothing.
      - model_fname: filename of the serialized Keras model to load (should be available in the current directory or retrievable remotely).
      - inlabel: the label of segments to be processed by the DNN; segments with other labels remain unchanged.
      - outlabels: a tuple of labels representing the possible output classes from the neural network model (e.g., 'speech', 'music', etc.).

    The class is designed to be extended by concrete implementations (e.g., for speech/music detection or gender segmentation) 
    that specialize the segmentation process for specific audio classification tasks.
    """
    def __init__(self, batch_size):
        # load the DNN model

        model_path = get_remote(self.model_fname)

        self.nn = keras.models.load_model(model_path, compile=False)
        self.nn.run_eagerly = False
        self.batch_size = batch_size

    def __call__(self, mspec, lseg, difflen = 0):
        """
        Refines an initial segmentation of an audio signal using DNN-based processing.
    
        Parameters:
            mspec (numpy.ndarray): the mel spectrogram of the audio signal.
            lseg (list of tuples): preliminary segmentation as a list of (label, start, stop) tuples.
                                   Only segments with label equal to 'inlabel' are processed by the DNN.
            difflen (int): padding difference. It is 0 if mspec has at least 68 frames; otherwise, it is set to (68 - len(mspec))
                           to account for short-duration signals.
    
        Returns:
            list of tuples: A refined segmentation as a list of (label, start, stop) tuples. For segments with the
                            target 'inlabel', the DNN predicts new labels which are smoothed using Viterbi decoding,
                            while segments with other labels remain unchanged.
        
        Process:
            1. If the number of mel bands (nmel) is less than 24, the spectrogram is truncated.
            2. Overlapping patches are extracted from the spectrogram using a sliding window (_get_patches).
            3. For segments labeled as 'inlabel', corresponding patches are batched and fed into the DNN to obtain predictions.
            4. Predictions for each segment are post-processed with Viterbi decoding to enforce temporal consistency.
            5. The refined segmentation is returned, with updated labels for processed segments and original labels for others.
        """

        if self.nmel < 24:
            mspec = mspec[:, :self.nmel].copy()
        
        patches, finite = _get_patches(mspec, 68, 2)
        if difflen > 0:
            patches = patches[:-int(difflen / 2), :, :]
            finite = finite[:-int(difflen / 2)]
            
        assert len(finite) == len(patches), (len(patches), len(finite))
            
        batch = []
        for lab, start, stop in lseg:
            if lab == self.inlabel:
                batch.append(patches[start:stop, :])

        if len(batch) > 0:
            batch = np.expand_dims(np.concatenate(batch), 3)
            rawpred = self.nn.predict(batch, batch_size=self.batch_size, verbose=2)
        gc.collect()
            
        ret = []
        for lab, start, stop in lseg:
            if lab != self.inlabel:
                ret.append((lab, start, stop))
                continue

            l = stop - start
            r = rawpred[:l] 
            rawpred = rawpred[l:]
            r[finite[start:stop] == False, :] = 0.5
            pred = viterbi_decoding(np.log(r), diag_trans_exp(self.viterbi_arg, len(self.outlabels)))
            for lab2, start2, stop2 in _binidx2seglist(pred):
                ret.append((self.outlabels[int(lab2)], start2+start, stop2+start))            
        return ret


class SpeechMusic(DnnSegmenter):
    """
    A DNN-based segmenter specialized for voice activity detection to classify audio segments into 'speech' or 'music'.
    It uses an energy-based detection approach (processing segments labeled as 'energy') and a pre-trained Keras model 
    ('keras_speech_music_cnn.hdf5') with 21 mel bands and Viterbi post-processing (viterbi_arg=150) to refine the segmentation.
    """
    # Voice activity detection: requires energetic activity detection
    outlabels = ('speech', 'music')
    model_fname = 'keras_speech_music_cnn.hdf5'
    inlabel = 'energy'
    nmel = 21
    viterbi_arg = 150

class SpeechMusicNoise(DnnSegmenter):
    """
    A DNN-based segmenter specialized for voice activity detection that classifies audio segments into 'speech', 'music', or 'noise'.
    It processes segments labeled as 'energy' using a pre-trained Keras model ('keras_speech_music_noise_cnn.hdf5') with 21 mel bands 
    and Viterbi post-processing (viterbi_arg=80) to refine the segmentation.
    """
    # Voice activity detection: requires energetic activity detection
    outlabels = ('speech', 'music', 'noise')
    model_fname = 'keras_speech_music_noise_cnn.hdf5'
    inlabel = 'energy'
    nmel = 21
    viterbi_arg = 80
    
class Gender(DnnSegmenter):
    """
    A DNN-based segmenter specialized for gender classification in speech segments.
    It processes segments labeled as 'speech' using a pre-trained Keras model ('keras_male_female_cnn.hdf5') with 24 mel bands,
    and refines the segmentation into 'female' and 'male' labels using Viterbi post-processing (viterbi_arg=80).
    """
    # Gender Segmentation, requires voice activity detection
    outlabels = ('female', 'male')
    model_fname = 'keras_male_female_cnn.hdf5'
    inlabel = 'speech'
    nmel = 24
    viterbi_arg = 80


class Segmenter:
    """
    Segmenter is the high-level interface for performing end-to-end audio segmentation in the inaSpeechSegmenter project.
    It integrates multiple segmentation stages—starting with energy-based voice activity detection, followed by DNN-based 
    segmentation (for speech/music/noise classification) and optional gender segmentation—to partition an audio file into 
    meaningful segments. The class handles media conversion to a 16kHz mono WAV signal, feature extraction (mel spectrogram 
    and MFCC), and applies post-processing (e.g., Viterbi decoding) to produce final segment labels with time boundaries.
    
    The __init__ method allows configuration of the voice activity detection engine ('sm' for speech/music or 'smn' for 
    speech/music/noise), toggles gender detection, and sets parameters such as the batch size and energy ratio for the initial 
    VAD. The __call__ method processes a single media file, while batch_process supports processing multiple files with robust 
    error handling and configurable output formats (CSV or TextGrid). This class is central to the inaSpeechSegmenter pipeline, 
    enabling applications such as speaker diarization, speech/music discrimination, and gender-specific segmentation.
    """
    def __init__(self, vad_engine='smn', detect_gender=True, ffmpeg='ffmpeg', batch_size=32, energy_ratio=0.03):
        """
        Initializes the Segmenter by configuring and loading the necessary neural network
        models for audio segmentation.

        Parameters:
            vad_engine (str): Specifies the voice activity detection engine to use.
                              Options are:
                              'sm'  for speech/music (used in ICASSP 2017 and MIREX 2018),
                              'smn' for speech/music/noise (a more recent implementation).
                              Defaults to 'smn'.
            detect_gender (bool): If True, further splits speech segments into 'male' and
                                  'female' using a dedicated gender model. If False,
                                  speech segments remain labeled as 'speech'. Defaults to True.
            ffmpeg (str): The command or path to invoke the ffmpeg executable, used for
                          converting input media to a 16kHz mono WAV signal.
                          Defaults to 'ffmpeg'.
            batch_size (int): Batch size for neural network inference. Higher values
                              may improve speed on GPUs but require more memory.
                              Defaults to 32.
            energy_ratio (float): The ratio used to set the threshold in the initial
                                  energy-based voice activity detection relative to the
                                  mean log energy. Lower values increase sensitivity.
                                  Defaults to 0.03.

        Raises:
            Exception: If the specified ffmpeg command is not found in the system path
                       or is not executable.
            AssertionError: If vad_engine is not 'sm' or 'smn', or if detect_gender
                            is not True or False.
        """
        # --- FFmpeg Check ---
        if ffmpeg is not None:
            # Use shutil.which to check if ffmpeg command exists and is executable
            if shutil.which(ffmpeg) is None:
                raise Exception(f"ffmpeg command '{ffmpeg}' not found or not executable. "
                                f"Please ensure ffmpeg is installed and in your system's PATH, "
                                f"or provide the full path.")
        self.ffmpeg = ffmpeg

        # --- Store Configuration ---
        self.energy_ratio = energy_ratio
        self.batch_size = batch_size # Store batch_size if needed elsewhere

        # Store detect_gender flag internally for post-processing logic
        self._detect_gender_enabled = bool(detect_gender)
        # Store VAD instance for accessing outlabels later
        self._vad_instance = None

        # --- Select and Load VAD Model ---
        assert vad_engine in ['sm', 'smn'], "vad_engine must be 'sm' or 'smn'"
        if vad_engine == 'sm':
            self.vad = SpeechMusic(batch_size)
        elif vad_engine == 'smn':
            self.vad = SpeechMusicNoise(batch_size)
        # Keep a reference to the VAD instance
        self._vad_instance = self.vad

        # --- Load Gender Model (if required) ---
        # Public attribute `detect_gender` can be used externally if needed
        self.detect_gender = self._detect_gender_enabled
        if self.detect_gender:
            self.gender = Gender(batch_size)
        else:
            # Ensure self.gender is None if not used, prevents potential AttributeError
            self.gender = None


    def segment_feats(self, mspec, loge, difflen, start_sec):
        """
        Performs segmentation on feature representations.

        Parameters:
            mspec: mel spectrogram of the audio signal.
            loge: log energy values from the MFCC extraction.
            difflen: Padding difference (0 if mspec has at least 68 frames, otherwise 68 - len(mspec)).
            start_sec: the starting time offset in seconds for the audio signal.

        Returns:
            A list of tuples (label, start, stop) representing the refined segmentation.
            For segments labeled as 'energy', the DNN-based segmentation is applied and refined using Viterbi decoding.
            Non-target segments are passed through unchanged.
        """
        
        # perform energy-based activity detection
        lseg = []
        for lab, start, stop in _binidx2seglist(_energy_activity(loge, self.energy_ratio)[::2]):
            if lab == 0:
                lab = 'noEnergy'
            else:
                lab = 'energy'
            lseg.append((lab, start, stop))

        # perform voice activity detection
        lseg = self.vad(mspec, lseg, difflen)

        # perform gender segmentation on speech segments
        if self.detect_gender:
            lseg = self.gender(mspec, lseg, difflen)

        return [(lab, start_sec + start * .02, start_sec + stop * .02) for lab, start, stop in lseg]

    def merge_short_gaps(self, segment_list, speech_labels, max_gap_duration_sec, min_speech_duration_sec=0.0):
        """
        Merges short non-speech segments that occur between speech segments.
        Optionally filters out very short speech segments after merging.
    
        Args:
            segment_list (list): List of (label, start_time_sec, stop_time_sec) tuples,
                                 sorted chronologically.
            speech_labels (set): A set containing all labels that should be treated as 'speech'
                                 (e.g., {'speech'}, or {'male', 'female'}).
            max_gap_duration_sec (float): The maximum duration (in seconds) of a
                                          non-speech segment to be merged if it's
                                          between two speech segments.
            min_speech_duration_sec (float): Optional. If greater than 0, speech segments
                                            shorter than this duration (in seconds) will
                                            be removed from the final list. Defaults to 0.0 (no filtering).
    
        Returns:
            list: A new segment list with short gaps merged and optionally short speech
                  segments filtered.
        """
        if not segment_list:
            return []
    
        # Ensure speech_labels is a set for efficient checking
        if not isinstance(speech_labels, set):
            speech_labels = set(speech_labels)
    
        merged_list = []
        i = 0
        while i < len(segment_list):
            current_label, current_start, current_stop = segment_list[i]
    
            # Look ahead: Check if the pattern is Speech - Gap - Speech
            if (current_label in speech_labels and
                    i + 2 < len(segment_list)): # Check if there are at least two more segments
    
                gap_label, gap_start, gap_stop = segment_list[i+1]
                next_label, next_start, next_stop = segment_list[i+2]
    
                # Check if the pattern matches: Speech - NonSpeech - Speech
                if (gap_label not in speech_labels and
                        next_label in speech_labels):
    
                    gap_duration = gap_stop - gap_start
                    # Check if the gap is short enough to merge
                    if gap_duration <= max_gap_duration_sec:
                        # Merge by extending the current speech segment to the end of the next one
                        merged_segment = (current_label, current_start, next_stop)
                        merged_list.append(merged_segment)
                        # Skip the next two segments (gap and the second speech segment)
                        i += 3
                        continue # Continue to the next potential segment after the merge
    
            # If no merge pattern was found or applied, add the current segment
            merged_list.append((current_label, current_start, current_stop))
            i += 1
    
        # --- Optional filtering stage ---
        if min_speech_duration_sec > 0.0:
            final_list = []
            for label, start, stop in merged_list:
                duration = stop - start
                # Keep segment if it's not speech OR if it is speech and meets minimum duration
                if label not in speech_labels or duration >= min_speech_duration_sec:
                    final_list.append((label, start, stop))
            return final_list
        else:
            # If no filtering, return the list with merged gaps
            return merged_list


    def __call__(self, medianame, tmpdir=None, start_sec=None, stop_sec=None):
        """
        Processes a media file to perform segmentation and applies post-processing
        to merge short non-speech gaps.

        Parameters:
            medianame: path or URL to the media file.
            tmpdir: directory for storing temporary files.
            start_sec: start time in seconds; audio before this point is ignored.
            stop_sec: end time in seconds; audio after this point is ignored.

        Returns:
            A list of tuples (label, start, stop) representing the final segmentation
            after gap merging and optional short segment filtering.
        """

        mspec, loge, difflen = _media2feats(medianame, tmpdir, start_sec, stop_sec, self.ffmpeg)
        # Use 0 as start_sec if None, for relative time calculations in segment_feats
        effective_start_sec = start_sec if start_sec is not None else 0

        # --- Step 1: Get the initial segmentation ---
        # segment_feats returns segments with absolute times (relative to 0 or original start_sec)
        lseg_initial = self.segment_feats(mspec, loge, difflen, effective_start_sec)

        # --- Step 2: Define parameters for post-processing ---
        # Determine which labels indicate speech based on configuration
        # Access the internal flag set during __init__
        if self._detect_gender_enabled:
            # Gender model outputs 'female', 'male'
            speech_labels_to_merge = {'female', 'male'}
        else:
            # Access the VAD instance stored during __init__
            if self._vad_instance:
                 # Find the label(s) corresponding to speech in the VAD model's output labels
                 speech_labels_to_merge = {label for label in self._vad_instance.outlabels if 'speech' in label.lower()}
            else:
                 # Fallback if _vad_instance wasn't set (shouldn't happen with current init)
                 warnings.warn("VAD instance not found, defaulting speech labels to {'speech'}")
                 speech_labels_to_merge = {'speech'}

            # Handle case where no speech label is dynamically found
            if not speech_labels_to_merge:
                 warnings.warn("Could not determine speech labels from VAD model for merging.")
                 # Fallback or default, e.g., {'speech'} if guaranteed
                 speech_labels_to_merge = {'speech'}


        # --- Tune these parameters based on your specific needs and evaluation ---
        # Example values - adjust these after testing!
        max_gap_to_fill_sec = 0.3 # Merge non-speech gaps up to 300ms long
        min_speech_duration_to_keep_sec = 0.1 # Optional: Remove speech segments shorter than 100ms

        # --- Step 3: Apply the merging function ---
        # Ensure the merge_short_gaps function is defined or imported
        lseg_merged = self.merge_short_gaps(
            segment_list=lseg_initial,
            speech_labels=speech_labels_to_merge,
            max_gap_duration_sec=max_gap_to_fill_sec,
            min_speech_duration_sec=min_speech_duration_to_keep_sec
        )

        # --- Step 4: Return the processed list ---
        return lseg_merged

    
    def batch_process(self, linput, loutput, tmpdir=None, verbose=False, skipifexist=False, nbtry=1, trydelay=2., output_format='csv'):
        """
        Processes a batch of media files for segmentation.

        This method handles batch processing with support for error handling, temporary file management, and 
        configurable output formats (CSV or TextGrid). It iteratively processes the input list, applies the segmentation 
        pipeline to each file, and writes the segmentation results to the specified output files.

        Parameters:
            linput: List of input media file paths or URLs.
            loutput: List of output file paths for segmentation results.
            tmpdir: Directory for storing temporary files.
            verbose: If True, prints progress messages.
            skipifexist: If True, skips files that already have an output.
            nbtry: Number of attempts to try downloading or processing a file.
            trydelay: Delay between retry attempts.
            output_format: Format for output segmentation files ('csv' or 'textgrid').

        Returns:
            A tuple containing:
                - Total batch processing duration.
                - Number of successfully processed files.
                - Average processing time per file.
                - A list of messages detailing the processing status of each file.
        """
        
        if verbose:
            print('batch_processing %d files' % len(linput))

        if output_format == 'csv':
            fexport = seg2csv
        elif output_format == 'textgrid':
            fexport = seg2textgrid
        else:
            raise NotImplementedError()
            
        t_batch_start = time.time()
        
        lmsg = []
        fg = featGenerator(linput.copy(), loutput.copy(), tmpdir, self.ffmpeg, skipifexist, nbtry, trydelay)
        i = 0
        for feats, msg in fg:
            lmsg += msg
            i += len(msg)
            if verbose:
                print('%d/%d' % (i, len(linput)), msg)
            if feats is None:
                break
            mspec, loge, difflen = feats
            #if verbose == True:
            #    print(i, linput[i], loutput[i])
            b = time.time()
            lseg = self.segment_feats(mspec, loge, difflen, 0)
            fexport(lseg, loutput[len(lmsg) -1])
            lmsg[-1] = (lmsg[-1][0], lmsg[-1][1], 'ok ' + str(time.time() -b))

        t_batch_dur = time.time() - t_batch_start
        nb_processed = len([e for e in lmsg if e[1] == 0])
        if nb_processed > 0:
            avg = t_batch_dur / nb_processed
        else:
            avg = -1
        return t_batch_dur, nb_processed, avg, lmsg


def medialist2feats(lin, lout, tmpdir, ffmpeg, skipifexist, nbtry, trydelay):
    """
    Processes a list of media files to extract features for segmentation.

    If an output file already exists, the file is skipped. For remote files, multiple access attempts
    are made if needed. This function is designed for batch processing of media files.
    """
    ret = None
    msg = []
    while ret is None and len(lin) > 0:
        src = lin.pop(0)
        dst = lout.pop(0)

        # if file exists: skipp
        if skipifexist and os.path.exists(dst):
            msg.append((dst, 1, 'already exists'))
            continue

        # create storing directory if required
        dname = os.path.dirname(dst)
        if not os.path.isdir(dname):
            os.makedirs(dname)
        
        itry = 0
        while ret is None and itry < nbtry:
            try:
                ret = _media2feats(src, tmpdir, None, None, ffmpeg)
            except:
                itry += 1
                errmsg = sys.exc_info()[0]
                if itry != nbtry:
                    time.sleep(random.random() * trydelay)
        if ret is None:
            msg.append((dst, 2, 'error: ' + str(errmsg)))
        else:
            msg.append((dst, 0, 'ok'))
            
    return ret, msg

    
def featGenerator(ilist, olist, tmpdir=None, ffmpeg='ffmpeg', skipifexist=False, nbtry=1, trydelay=2.):
    """
    Generator that yields feature extraction results and processing messages for a batch of media files.

    It creates a threaded process to extract features using medialist2feats and yields the extracted features
    along with status messages until all files have been processed.
    """
    thread = ThreadReturning(target = medialist2feats, args=[ilist, olist, tmpdir, ffmpeg, skipifexist, nbtry, trydelay])
    thread.start()
    while True:
        ret, msg = thread.join()
        if len(ilist) == 0:
            break
        thread = ThreadReturning(target = medialist2feats, args=[ilist, olist, tmpdir, ffmpeg, skipifexist, nbtry, trydelay])
        thread.start()
        yield ret, msg
    yield ret, msg
