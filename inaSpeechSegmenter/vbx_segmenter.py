"""
This module implements a comprehensive voice femininity scoring system that integrates several key components:
- VBx Feature Extraction: Computes x-vector embeddings from audio signals using a selected backend (currently ONNX).
- Voice Activity Detection (VAD): Uses inaSpeechSegmenter's segmentation to detect speech regions.
- Gender Detection: Applies a pre-trained gender detection model to estimate a voice femininity score.

Additionally, the module includes utility functions to process VAD annotations, manage x-vector retention, and compute 
features from audio signals (e.g., mel spectrograms and VBx features). It is designed for applications in speaker verification 
and gender-based audio analysis.
"""

import os
from abc import ABC, abstractmethod
import platform  # Import platform module
import numpy as np
import onnxruntime as ort
import logging
#import torch.backends

import keras
from pyannote.core import Segment, Annotation, Timeline

#utils import get_remote

# Setup logger if not already done
logger = logging.getLogger(__name__)
# Ensure basic config in case it wasn't set globally
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)


STEP = 24
WINLEN = 144
FEAT_DIM = 64
EMBED_DIM = 256
SR = 16000


def is_mid_speech(start, stop, a_vad):
    """
    Determines if the midpoint of a segment falls within a speech region detected by VAD.

    This function calculates the midpoint of the segment (average of start and stop) and checks if it lies within any
    speech segment in the VAD annotation (a_vad). It returns True if the midpoint is inside at least one speech segment,
    indicating that the segment likely contains speech.

    Note: The previous implementation iterated through all VAD segments for each x-vector. This version has been rewritten for
    clarity and returns True as soon as a mid-speech match is found.
    """
    midpoint = (start + stop) / 2

    # Check each VAD segment to see if it contains the midpoint.
    for seg, _, _ in a_vad.itertracks(yield_label=True):
        if (seg.start < midpoint) and (midpoint < seg.end):
            return True

    return False
    
def add_needed_vectors(xvectors, t_mid):
    """
    Ensure that at least 50% of the predictions have midpoints within speech segments.

    This function checks if the current number of x-vectors (xvectors) is less than 50% of the total candidates
    (represented by t_mid). If so, it sorts the candidates in descending order by a certain criterion (first column in t_mid),
    and appends additional x-vectors from t_mid to reach the minimum required number. This helps maintain a sufficient number
    of predictions that are likely to be in speech segments.
    """
    min_pred = round(0.5 * len(t_mid))
    if len(xvectors) < min_pred:
        # Sort array descending
        t_mid = np.asarray(t_mid)
        t_mid = t_mid[t_mid[:, 0].argsort()][::-1]
        diff = min_pred - len(xvectors)
        for _, k, s, x in t_mid[len(xvectors):len(xvectors) + diff]:
            xvectors.append((k, (s.start, s.stop), x))
    return xvectors


def get_femininity_score(g_preds):
    """
    Computes a weighted voice femininity score using gender prediction probabilities.

    Instead of a binary decision for each segment, this function calculates the overall femininity score 
    as the weighted average of the female probabilities, with weights based on the segment durations.
    
    Parameters:
        g_preds (list): List of tuples (start, stop, p) where p is the predicted probability of being female.
    
    Returns:
        float: The weighted femininity score, between 0 and 1.
    """
    total_duration = 0.0
    female_weight = 0.0
    
    for start, stop, p in g_preds:
        duration = stop - start
        total_duration += duration
        female_weight += duration * p
        
    return female_weight / total_duration if total_duration > 0 else 0.0


def get_annot_VAD(vad_tuples):
    """
    Create a VAD annotation from a list of VAD tuples.

    This function takes a list of tuples (label, start, end) and constructs an Annotation object,
    including only those segments labeled as "speech". Each speech segment is added as a Segment with the label "speech".
    
    Returns:
        Annotation: An annotation containing only the speech segments.
    """
    annot_vad = Annotation()
    for lab, start, end in vad_tuples:
        if lab == "speech":
            annot_vad[Segment(start, end), '_'] = lab
    return annot_vad


def get_features(signal, LC=150, RC=149):

    """
    Extracts VBx features from an audio signal for speaker diarization and related tasks.

    This function was entirely compiled the VBx script 'predict.py', available at:
    ttps://github.com/BUTSpeechFIT/VBx/blob/master/VBx/predict.py 
    
    It computes a feature representation of the input signal by performing the following steps:
      - Applies a Povey window to frame the signal.
      - Constructs a Mel filterbank matrix with specified parameters (e.g., FEAT_DIM, LOFREQ, HIFREQ).
      - Adds dithering to the signal for numerical stability.
      - Applies symmetric padding to the signal.
      - Computes the power spectrum using a triangular filterbank (HTK-compatible) via fbank_htk.
      - Normalizes the features using a floating window CMVN (Cepstral Mean and Variance Normalization).

    Parameters:
        signal (numpy.ndarray): The input audio signal.
        LC (int): Left context for CMVN (default 150).
        RC (int): Right context for CMVN (default 149).

    Returns:
        numpy.ndarray: A 2D array of normalized features suitable for use in the VBx x-vector extraction pipeline.
    """
    
    """
    This code function is entirely copied from the VBx script 'predict.py'
    https://github.com/BUTSpeechFIT/VBx/blob/master/VBx/predict.py
    """

    noverlap = 240
    winlen = 400
    window = povey_window(winlen)
    fbank_mx = mel_fbank_mx(
        winlen, SR, NUMCHANS=FEAT_DIM, LOFREQ=20.0, HIFREQ=7600, htk_bug=False)

    np.random.seed(3)  # for reproducibility
    signal = add_dither((signal * 2 ** 15).astype(int))
    seg = np.r_[signal[noverlap // 2 - 1::-1], signal, signal[-1:-winlen // 2 - 1:-1]]
    fea = fbank_htk(seg, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True)
    fea = cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)
    return fea


class VoiceFemininityScoring:
    """
    A class to compute a voice femininity score using a combination of VBx x-vector extraction, 
    voice activity detection, and gender detection.
    
    (For more information about the backend, see : https://github.com/BUTSpeechFIT/VBx)
    
    The process involves:
      - Extracting x-vector embeddings from audio using a VBx extractor (currently supporting the ONNX backend).
      - Applying voice activity detection (VAD) to identify speech regions via inaSpeechSegmenter.
      - Using a pre-trained gender detection MLP model (with criteria "bgc" or "vfp") to predict gender probabilities
        for the speech segments.
    
    The computed femininity score reflects the proportion of speech segments that are predicted to be female, and
    can be used in applications like speaker diarization and gender-based audio analysis.
    
    The __init__ method loads the required models and sets parameters (such as VAD thresholds) based on the chosen 
    gender detection criteria and backend.
    """

    def __init__(self, gd_model_criteria="bgc", backend='onnx'):
        """
        Load VBx model weights according to the chosen backend and initialize the voice activity detection and 
        gender detection models.

        Parameters:
            gd_model_criteria (str): Criterion for selecting the gender detection model.
                                     Options are "bgc" (default) and "vfp".
            backend (str): Backend for the VBx x-vector extractor. Currently, only 'onnx' is supported.
        
        The method:
            - Instantiates the VBx x-vector extractor (using the ONNX backend).
            - Loads the gender detection model based on the specified criterion, setting an appropriate VAD threshold.
            - Initializes the voice activity detection (VAD) model from inaSpeechSegmenter.
        """
        
        # VBx Extractor
        assert backend in ['onnx'], "Backend should be 'onnx' (or 'pytorch' if uncommented)."
        if backend == "onnx":
            self.xvector_model = OnnxBackendExtractor()
        # elif backend == "pytorch":
        #     self.xvector_model = TorchBackendExtractor()

        # Gender detection model
        # TODO : tell which (bad) model was provided !
        assert gd_model_criteria in ["bgc", "vfp"], "Gender detection model Criteria must be 'bgc' (default) or 'vfp'"
        gd_model = None
        if gd_model_criteria == "bgc":
            gd_model = "interspeech2023_all.hdf5"
            self.vad_thresh = 0.7
        elif gd_model_criteria == "vfp":
            gd_model = "interspeech2023_cvfr.hdf5"
            self.vad_thresh = 0.62
        self.gender_detection_mlp_model = keras.models.load_model(
            get_remote(gd_model),
            compile=False)

        # Voice activity detection model
        self.vad = Segmenter(vad_engine='smn', detect_gender=False)

    def apply_vad(self, xvectors, a_vad):
        midpoint_seg = []
        n_xvectors = []
        for key, (start, stop), x in xvectors:

            # Keep segment label whose segment midpoint is in a speech segment
            if is_mid_speech(start, stop, a_vad):
                seg_total_duration = stop - start
                seg_cropped = Timeline([Segment(start, stop)]).crop(a_vad.get_timeline())
                # At least x % of the segment is detected as speech
                if seg_cropped.duration() / seg_total_duration >= self.vad_thresh:
                    n_xvectors.append((key, (start, stop), x))
                # Save overlap ratio with vad
                midpoint_seg.append(((seg_cropped.duration() / seg_total_duration), key, Segment(start, stop), x))

        # Add vectors with vad-overlap if too many predictions have been removed
        return add_needed_vectors(n_xvectors, midpoint_seg)

    def __call__(self, fpath, tmpdir=None):
        """
        Processes an audio file to compute its voice femininity score.
    
        The function performs the following steps:
          - Converts the input media file (fpath) to a 16kHz mono WAV signal using ffmpeg.
          - Extracts Mel band features from the signal.
          - Applies voice activity detection (VAD) via inaSpeechSegmenter to obtain speech segments.
          - Computes VBx x-vector embeddings from the extracted features.
          - Refines the x-vectors using the VAD results.
          - Uses a pre-trained gender detection model (MLP) to predict gender probabilities for each x-vector.
          - Computes a femininity score based on the proportion of speech segments predicted as female.
          
        Returns:
            score (float or None): The voice femininity score (before final sigmoid activation), or None if no speech is detected.
            speech_duration (float): Total duration (in seconds) of detected speech segments.
            nb_vectors (int): The number of x-vectors retained after VAD-based filtering.
        """
        basename, ext = os.path.splitext(os.path.basename(fpath))[0], os.path.splitext(os.path.basename(fpath))[1]

        # Read "wav" file
        signal = media2sig16kmono(fpath, tmpdir, dtype="float64")
        duration = len(signal) / SR

        # Applying voice activity detection
        vad_seg = self.vad(fpath)
        annot_vad = get_annot_VAD(vad_seg)
        speech_duration = annot_vad.label_duration("speech")

        if speech_duration:

            # Extract Mel band features from the signal.
            features = get_features(signal)

            # Compute x-vector embeddings using the VBx extractor.
            # NOTE: This is the most computationally expensive step and ideally should be performed
            # after applying VAD to avoid processing non-speech segments.
            # Currently, the VBx extractor returns tuples in the form (key, (seg_start, seg_end), xvector).
            # In the future, this process can be modularized into:
            #   M1: Identifying (seg_start, seg_end) segments from the feature length.
            #   M2: Computing x-vectors from features given the identified segments.
            #   M3: Applying VAD filtering between M1 and M2 to discard non-speech segments.
            x_vectors = self.xvector_model(basename, features, duration)

            # Apply VAD filtering to refine x-vector selection before gender detection.
            x_vectors = self.apply_vad(x_vectors, annot_vad)

            # Applying gender detection (pretrained Multi layer perceptron)
            x = np.asarray([x for _, _, x in x_vectors])
            gender_pred = self.gender_detection_mlp_model.predict(x, verbose=0)
            if len(gender_pred) > 1:
                gender_pred = np.squeeze(gender_pred)

            # Link segment start/stop from x-vectors extraction to gender predictions.
            gender_pred = np.asarray(
                [(segtup[0], segtup[1], pred) for (_, segtup, _), pred in zip(x_vectors, gender_pred)])

            score, nb_vectors = get_femininity_score(gender_pred), len(gender_pred)

        else:
            score, nb_vectors = None, 0

        return score, speech_duration, nb_vectors


class VBxExtractor(ABC):
    """
    VBxExtractor is an abstract base class for extracting x-vector embeddings from audio features.

    This class defines a common interface for x-vector extraction across different backends.
    Subclasses must implement the __init__ method to initialize the model and the get_embedding method 
    to compute an embedding for a given segment of features.

    The __call__ method divides the input feature matrix (fea) into overlapping segments of fixed length (WINLEN),
    computes an x-vector for each segment, and collects these into a list of tuples in the form:
        (unique_key, (seg_start, seg_end), xvector)

    Each key is generated using the basename and the segment boundaries, and the x-vector is scaled by 10
    for output standardization (to achieve a standard deviation of 1). 

    Note: The segmentation logic currently implemented in __call__ could be refactored into three distinct steps:
        M1: Identify segment boundaries (start, end) based on the feature length.
        M2: Compute x-vectors from the features using these boundaries.
        M3: Optionally filter segments based on voice activity detection (VAD) before computing x-vectors.
    """
    @abstractmethod
    def __init__(self):
        """
        Method to be implemented by each extractor.
        Initialize model according to the chosen backend.
        """
        pass

    def __call__(self, basename, fea, duration):
        """
        Extract x-vector embeddings from the input feature matrix.

        Parameters:
            basename (str): A base name used for generating unique keys for each segment.
            fea (numpy.ndarray): The input feature matrix.
            duration (float): The total duration of the audio signal, used to determine the last segment's boundaries.

        Returns:
            list: A list of tuples, each containing:
                - a unique key (str) for the segment,
                - a tuple (seg_start, seg_end) representing the segment's start and end times (in seconds),
                - the x-vector embedding (numpy.ndarray) for that segment, scaled by 10.
        """
        # SHOULD BE FACTORIZED and use feats, with list (start, end)
        # number of lines could be divided by 2
        xvectors = []
        start = 0
        for start in range(0, len(fea) - WINLEN, STEP):
            data = fea[start:start + WINLEN]
            xvector = self.get_embedding(data)
            key = f'{basename}_{start:08}-{(start + WINLEN):08}'
            if np.isnan(xvector).any():
                logger.warning(f'NaN found, not processing: {key}{os.linesep}')
            else:
                seg_start = round(start / 100.0, 3)
                seg_end = round(start / 100.0 + WINLEN / 100.0, 3)
                xvectors.append((key, (seg_start, seg_end), xvector))

        #  Last segment
        if len(fea) - start - STEP >= 10:
            data = fea[start + STEP:len(fea)]
            xvector = self.get_embedding(data)
            key = f'{basename}_{(start + STEP):08}-{len(fea):08}'
            if np.isnan(xvector).any():
                logger.warning(f'NaN found, not processing: {key}{os.linesep}')
            else:
                seg_start = round((start + STEP) / 100.0, 3)
                seg_end = round(duration, 3)
                xvectors.append((key, (seg_start, seg_end), xvector))

        # Multiply all vbx vectors by 10 (output standardization to get std=1)
        return [(key, seg, x * 10) for key, seg, x in xvectors]


class OnnxBackendExtractor(VBxExtractor):
    """
    An x-vector extractor implementation that uses an ONNX model for inference.

    This class loads the pre-trained x-vector extraction model from an ONNX file
    ("final.onnx") using the ONNX Runtime. It dynamically selects the best
    available execution provider based on the operating system and hardware:
    - Windows: Prefers DirectML (DmlExecutionProvider), then CUDA, then CPU.
    - Linux:   Prefers ROCm (ROCMExecutionProvider), then CUDA, then CPU.
    - Other:   Prefers CPU or other available providers like CoreML if relevant.

    Requires appropriate onnxruntime package:
    - onnxruntime-directml for DirectML support on Windows.
    - onnxruntime-gpu for CUDA (Win/Linux) and ROCm (Linux) support.
    - onnxruntime (base package) provides CPU support.
    """
    def __init__(self):
        """
        Initializes the extractor by loading the ONNX model and setting up the
        inference session with the best available execution provider.
        """
        # Define constants needed if __call__ is implemented here
        # self.STEP = 24
        # self.WINLEN = 144

        model_path = get_remote("final.onnx") # Assumes final.onnx exists or get_remote works
        so = ort.SessionOptions()
        so.log_severity_level = 3 # Reduce ONNX logging verbosity

        # Determine available providers in the installed ONNX Runtime
        available_providers = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime Execution Providers: {available_providers}")

        # Define preferred provider order based on OS
        preferred_providers = []
        system = platform.system()

        if system == "Windows":
            # Order: DirectML -> CUDA -> CPU
            preferred_providers.append('DmlExecutionProvider')
            preferred_providers.append('CUDAExecutionProvider')
            preferred_providers.append('CPUExecutionProvider')
        elif system == "Linux":
            # Order: ROCm -> CUDA -> CPU
            preferred_providers.append('ROCMExecutionProvider')
            preferred_providers.append('CUDAExecutionProvider')
            preferred_providers.append('CPUExecutionProvider')
        else: # macOS or other OS
            # Add other providers if relevant, e.g., CoreML for macOS
            # Example: Check and add CoreML if available
            if 'CoreMLExecutionProvider' in available_providers:
                preferred_providers.append('CoreMLExecutionProvider')
            preferred_providers.append('CPUExecutionProvider')

        # Filter the preferred list to only include available providers
        valid_providers = [p for p in preferred_providers if p in available_providers]

        # Ensure CPU is always an option if everything else fails
        if not valid_providers:
             logger.warning("No preferred providers were available. Defaulting to CPU.")
             if 'CPUExecutionProvider' in available_providers:
                 valid_providers = ['CPUExecutionProvider']
             else:
                 # This is highly unlikely unless ORT installation is broken
                 raise RuntimeError("CPUExecutionProvider is not available in ONNX Runtime!")
        elif 'CPUExecutionProvider' not in valid_providers:
             # Append CPU if it wasn't already added (e.g., if only GPU providers were preferred but unavailable)
             if 'CPUExecutionProvider' in available_providers:
                  valid_providers.append('CPUExecutionProvider')


        logger.info(f"Attempting to load ONNX model with providers: {valid_providers}")

        try:
            # Load the model with the prioritized list of available providers
            model = ort.InferenceSession(model_path, so, providers=valid_providers)
            # Log the provider actually being used
            actual_provider = model.get_providers()
            logger.info(f"ONNX Runtime using provider(s): {actual_provider}")

        except Exception as e:
            logger.error(f"Failed to initialize ONNX InferenceSession with providers {valid_providers}: {e}")
            logger.warning("Falling back to CPUExecutionProvider only.")
            try:
                 # Fallback explicitly to CPU if the preferred list failed
                 if 'CPUExecutionProvider' not in available_providers:
                      raise RuntimeError("CPUExecutionProvider not available even for fallback!")
                 model = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
                 logger.info("ONNX Runtime using provider(s): ['CPUExecutionProvider']")
            except Exception as fallback_e:
                 logger.error(f"Failed to initialize ONNX InferenceSession even with CPU! Error: {fallback_e}")
                 raise fallback_e # Re-raise the exception if CPU also fails

        # Store model and input/output names
        self.input_name = model.get_inputs()[0].name
        self.label_name = model.get_outputs()[0].name
        self.model = model

    def get_embedding(self, fea):
        """
        Compute an x-vector embedding for a given segment of features using the
        loaded ONNX model.

        Parameters:
            fea (numpy.ndarray): A 2D array of features for a segment (expected shape e.g., [WINLEN, FEAT_DIM]).

        Returns:
            numpy.ndarray: The computed x-vector embedding (1D array).
        """
        # Ensure input is float32, transpose to [FEAT_DIM, WINLEN], and add batch dimension [1, FEAT_DIM, WINLEN]
        # Adjust transpose if model expects [batch, time, features] instead of [batch, features, time]
        input_data = fea.astype(np.float32).transpose()[np.newaxis, :, :]
        # Alternatively, if model expects [batch, time, features]:
        # input_data = fea.astype(np.float32)[np.newaxis, :, :]

        # Run inference
        result = self.model.run(
            [self.label_name],
            {self.input_name: input_data}
        )
        # Squeeze the batch dimension from the output -> [embed_dim]
        return result[0].squeeze()

    # You still need the __call__ method here to make the class fully functional
    # It should contain the logic to loop through the features `fea`,
    # extract segments, call `self.get_embedding` on each, and format the output list.
    def __call__(self, basename, fea, duration):
        """
        Extract x-vector embeddings from the full input feature matrix by segmenting it.

        Parameters:
            basename (str): A base name used for generating unique keys for each segment.
            fea (numpy.ndarray): The input feature matrix (e.g., [num_frames, FEAT_DIM]).
            duration (float): The total duration of the audio signal in seconds.

        Returns:
            list: A list of tuples, each containing:
                - a unique key (str) for the segment,
                - a tuple (seg_start_sec, seg_end_sec),
                - the x-vector embedding (numpy.ndarray), scaled by 10.
        """
        # Define constants locally or access them via self if defined in __init__
        STEP = 24
        WINLEN = 144
        # FEAT_DIM = 64 # Not needed directly here, inferred from fea.shape

        xvectors = []
        start_frame = 0
        num_frames = fea.shape[0]
        frames_per_sec = 100.0 # Assuming 10ms frame shift for timestamp calculation

        # Loop through overlapping windows
        for start_frame in range(0, num_frames - WINLEN + 1, STEP):
            end_frame = start_frame + WINLEN
            data = fea[start_frame:end_frame, :] # Get segment [WINLEN, FEAT_DIM]

            try:
                xvector = self.get_embedding(data)
                key = f'{basename}_{start_frame:08d}-{end_frame:08d}'

                # Check for NaNs before proceeding
                if np.isnan(xvector).any():
                    logger.warning(f'NaN found in xvector, not processing segment: {key}')
                else:
                    seg_start_sec = round(start_frame / frames_per_sec, 3)
                    seg_end_sec = round(end_frame / frames_per_sec, 3)
                    # Append tuple: (key, (start_sec, end_sec), embedding * 10)
                    xvectors.append((key, (seg_start_sec, seg_end_sec), xvector * 10))
            except Exception as e:
                 logger.error(f"Error processing segment {start_frame}-{end_frame}: {e}")


        # Handle the last partial segment if it's long enough
        last_segment_start = start_frame + STEP # start_frame holds the beginning of the last full segment processed
        if num_frames - last_segment_start >= 10: # Minimum length check
            data = fea[last_segment_start:num_frames, :]
            try:
                xvector = self.get_embedding(data) # Model should handle variable length input if designed for it, otherwise needs padding
                key = f'{basename}_{last_segment_start:08d}-{num_frames:08d}'

                if np.isnan(xvector).any():
                    logger.warning(f'NaN found in last xvector, not processing segment: {key}')
                else:
                    seg_start_sec = round(last_segment_start / frames_per_sec, 3)
                    # Use the full duration for the end time of the last segment
                    seg_end_sec = round(duration, 3)
                    xvectors.append((key, (seg_start_sec, seg_end_sec), xvector * 10))
            except Exception as e:
                 logger.error(f"Error processing last segment {last_segment_start}-{num_frames}: {e}")


        return xvectors
