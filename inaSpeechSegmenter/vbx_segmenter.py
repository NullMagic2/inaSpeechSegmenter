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
import numpy as np
import onnxruntime as ort
import logging
#import torch.backends

import keras
from pyannote.core import Segment, Annotation, Timeline

# from .resnet import ResNet101
from .features_vbx import povey_window, mel_fbank_mx, add_dither, fbank_htk, cmvn_floating_kaldi
from .segmenter import Segmenter
from .io import media2sig16kmono
from .remote_utils import get_remote

#torch.backends.cudnn.enabled = True
logger = logging.getLogger(__name__)
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

    Note: The current implementation iterates through all VAD segments for each x-vector. An optimization is recommended 
    for processing large numbers of x-vectors.
    """
    m = (start + stop) / 2
    is_speech = [True if seg.start < m < seg.end else False for seg, _, _ in a_vad.itertracks(yield_label=True)]
    return np.any(is_speech)


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
    Computes the voice femininity score from gender prediction segments.

    This function creates an annotation from gender predictions, where each prediction is a tuple
    (start, stop, p) and p (a probability) is interpreted as female if p >= 0.5. It then computes the
    femininity score as the ratio of the number of segments labeled as female to the total number of segments.

    Returns:
        float: The femininity score, representing the proportion of segments identified as female.
    """
    a_temp = Annotation()
    for start, stop, p in g_preds:
        a_temp[Segment(start, stop), '_'] = (p >= 0.5)

    # Return binary score and number of retained predictions
    return len(a_temp.label_timeline(True)) / len(a_temp)


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

    This class loads the pre-trained x-vector extraction model from an ONNX file ("final.onnx") using the ONNX Runtime.
    It sets up the inference session with a preference for CUDA if available (falling back to CPU otherwise). The input and output 
    tensor names are retrieved from the model, and the get_embedding method runs the model on a given segment of features to produce 
    an x-vector embedding.
    """
    def __init__(self):
        model_path = get_remote("final.onnx")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            model = ort.InferenceSession(model_path, so, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except:
            model = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
        self.input_name = model.get_inputs()[0].name
        self.label_name = model.get_outputs()[0].name
        self.model = model

    def get_embedding(self, fea):
        """
        Compute an x-vector embedding for a given segment of features.

        Parameters:
            fea (numpy.ndarray): A 2D array of features for a segment.

        Returns:
            numpy.ndarray: The x-vector embedding obtained by running inference on the input features.
        """
        return self.model.run(
            [self.label_name],
            {self.input_name: fea.astype(np.float32).transpose()[np.newaxis, :, :]}
        )[0].squeeze()
