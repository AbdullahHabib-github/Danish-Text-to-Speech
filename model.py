"""
doc string
"""
import os
from functools import lru_cache
from pathlib import Path
import sherpa_onnx
from huggingface_hub import hf_hub_download

def get_file( repo_id: str, filename: str, subfolder: str = ".", ) -> str:
    """
    doc string
    """
    model_filename = hf_hub_download(
        repo_id = repo_id,
        filename = filename,
        subfolder = subfolder,
)
    return model_filename

@lru_cache(maxsize = 10)
def get_vits_piper(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
    """
    doc string
    """
    data_dir = "/tmp/espeak-ng-data"
    name = "da_DK-talesyntese-medium"
    model = get_file(
        repo_id = repo_id,
        filename = f"{name}.onnx",
        subfolder = ".",
    )
    tokens = get_file(
        repo_id = repo_id,
        filename = "tokens.txt",
        subfolder = ".")
    print(model)
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                model = model,
                lexicon = "",
                data_dir = data_dir,
                tokens = tokens,
                length_scale = 1.0 / speed,
            ),
            provider = "cpu",
            debug = True,
            num_threads = 2,
        )
    )
    tts = sherpa_onnx.OfflineTts(tts_config)
    return tts

@lru_cache(maxsize = 10)
def get_pretrained_model(repo_id: str, speed: float) -> sherpa_onnx.OfflineTts:
     """
     doc string
     """
     tts = get_vits_piper(repo_id, speed)
     return tts
