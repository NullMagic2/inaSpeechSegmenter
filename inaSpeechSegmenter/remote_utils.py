"""
Model Retrieval Module for inaSpeechSegmenter
-----------------------------------------------
This module defines the remote locations for pre-trained model files used by inaSpeechSegmenter and
provides a utility function to retrieve these models.

Key components:
- ISS_url and VFS_url: Base URLs for downloading models from different releases.
- dmodels: a dictionary mapping model file names to their corresponding base URLs.
- get_remote: a function that checks if a requested model file exists locally (e.g., in a Docker image's
  /root/.keras/inaSpeechSegmenter/ directory) and returns its path if available. Otherwise, it uses
  Keras's get_file utility to download the model file to the local cache (typically ~/.keras/inaSpeechSegmenter/).

This setup ensures that the required models are available for inaSpeechSegmenter, either by accessing
pre-existing files or by downloading them from the remote repositories.
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
    Retrieves the path to a model file, either from a local cache or by downloading it remotely.

    The function first checks if the model file is available locally (e.g., in a Docker image,
    the /root/.keras/inaSpeechSegmenter/ directory). If the file exists and is readable, its path is returned.
    Otherwise, it constructs the download URL using the base URL from dmodels and uses Keras's get_file
    to download the model file into the local cache directory (typically ~/.keras/inaSpeechSegmenter/).
    """
    
    # If in a docker image, try to get the file in /root/.keras
    rootpath = '/root/.keras/inaSpeechSegmenter/' + model_fname
    if os.access(rootpath, os.R_OK):
        return rootpath

    # Standard keras get file
    # check if the file is in /home/$USER/.keras/inaSpeechSegmenter/ and download it if required
    url = dmodels[model_fname]
    return get_file(model_fname, url + model_fname, cache_subdir='inaSpeechSegmenter')
