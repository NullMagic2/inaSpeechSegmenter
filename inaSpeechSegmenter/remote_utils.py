"""
Model Retrieval Module for inaSpeechSegmenter:
Defines remote URLs for pre-trained model files and provides the get_remote function to retrieve them.
get_remote first checks if the model is available locally (e.g., in a Docker container at /root/.keras/inaSpeechSegmenter/)
and, if not, downloads it using Keras's get_file utility to the local cache (~/.keras/inaSpeechSegmenter/).
"""

import os
from keras.utils import get_file

ISS_url = 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/models/'
VFS_url = 'https://github.com/ina-foss/inaSpeechSegmenter/releases/download/interspeech23/'

dmodels = {
    'keras_speech_music_cnn.hdf5': ISS_url,
    'keras_speech_music_noise_cnn.hdf5': ISS_url,
    'keras_male_female_cnn.hdf5': ISS_url,
    'interspeech2023_all.hdf5': VFS_url,
    'interspeech2023_cvfr.hdf5': VFS_url,
    'final.onnx': VFS_url,
    'raw_81.pth': VFS_url,
}


def get_remote(model_fname):
    """
    Retrieves the specified model file for inaSpeechSegmenter.
    It first checks for a locally cached version (useful in Docker environments) at /root/.keras/inaSpeechSegmenter/.
    If not found, it downloads the model using Keras's get_file utility from the appropriate remote URL as defined in dmodels.
    """
    # if in a docker image, try to get the file in /root/.keras
    rootpath = '/root/.keras/inaSpeechSegmenter/' + model_fname
    if os.access(rootpath, os.R_OK):
        return rootpath

    # standard keras get file
    # check if the file is in /home/$USER/.keras/inaSpeechSegmenter/ and download it if required
    url = dmodels[model_fname]
    return get_file(model_fname, url + model_fname, cache_subdir='inaSpeechSegmenter')
