"""
ComfyUI_Muse - ComfyUI Node Package for Muse AI Music Generation

Muse is an open-source model for reproducible long-form song generation
with fine-grained style control.

GitHub: https://github.com/yuhui1038/Muse
Paper: https://arxiv.org/abs/2601.03973
"""

import os
import sys
import torch
import torchaudio
import soundfile as sf
import numpy as np
import uuid
import gc
import folder_paths
import logging
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# Add util directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

# Register model folder path
folder_paths.add_model_folder_path("muse", os.path.join(folder_paths.models_dir, "muse"))


def get_model_base_dir():
    """Get the base directory for Muse models."""
    paths = folder_paths.get_folder_paths("muse")
    for p in paths:
        if os.path.exists(p):
            return p
    base = paths[0] if paths else os.path.join(folder_paths.models_dir, "muse")
    os.makedirs(base, exist_ok=True)
    return base


MODEL_BASE_DIR = get_model_base_dir()


def auto_download_models():
    """Download models if not present. Called on first use."""
    muse_path = os.path.join(MODEL_BASE_DIR, "Muse-0.6b")
    codec_path = os.path.join(MODEL_BASE_DIR, "mucodec", "mucodec.pt")
    audioldm_path = os.path.join(current_dir, "mucodec", "tools", "audioldm_48k.pth")

    # Build download list with sizes for user feedback
    downloads_needed = []
    total_size = 0
    if not os.path.exists(muse_path):
        downloads_needed.append(("Muse-0.6b", muse_path, "1.2GB"))
        total_size += 1.2
    if not os.path.exists(codec_path):
        downloads_needed.append(("mucodec.pt", codec_path, "4GB"))
        total_size += 4
    if not os.path.exists(audioldm_path):
        downloads_needed.append(("audioldm_48k.pth", audioldm_path, "5GB"))
        total_size += 5

    if not downloads_needed:
        return True

    print("")
    print("=" * 60)
    print("[Muse] First-time setup: Downloading required models")
    print("=" * 60)
    print(f"[Muse] Models to download: {len(downloads_needed)} (~{total_size:.1f}GB total)")
    for name, _, size in downloads_needed:
        print(f"       - {name} ({size})")
    print("[Muse] This may take several minutes depending on your connection...")
    print("")

    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        for i, (name, path, size) in enumerate(downloads_needed, 1):
            print(f"[Muse] [{i}/{len(downloads_needed)}] Downloading {name} ({size})...")

            if name == "Muse-0.6b":
                snapshot_download(
                    repo_id="bolshyC/Muse-0.6b",
                    local_dir=path,
                    local_dir_use_symlinks=False,
                )
            elif name == "mucodec.pt":
                os.makedirs(os.path.dirname(path), exist_ok=True)
                hf_hub_download(
                    repo_id="AcademiCodec/MuCodec",
                    filename="mucodec.pt",
                    local_dir=os.path.dirname(path),
                    local_dir_use_symlinks=False,
                )
            elif name == "audioldm_48k.pth":
                os.makedirs(os.path.dirname(path), exist_ok=True)
                hf_hub_download(
                    repo_id="AcademiCodec/MuCodec",
                    filename="audioldm_48k.pth",
                    local_dir=os.path.dirname(path),
                    local_dir_use_symlinks=False,
                )
            print(f"[Muse] ✓ {name} downloaded successfully")

        print("")
        print("=" * 60)
        print("[Muse] All models downloaded successfully!")
        print("[Muse] You're ready to generate music.")
        print("=" * 60)
        print("")
        return True

    except ImportError:
        print("[Muse] ERROR: huggingface_hub not installed.")
        print("[Muse] Run: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"[Muse] ERROR: Download failed: {e}")
        print("[Muse] Try running manually: python install.py")
        print(f"[Muse] Location: {current_dir}")
        return False


class MuseModelManager:
    """Singleton manager for Muse model instances."""
    _instance = None
    _muse_pipe = None
    _codec_pipe = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MuseModelManager, cls).__new__(cls)
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def get_muse_pipeline(self, model_name: str = "Muse-0.6b", dtype: str = "bfloat16"):
        """Get or create the Muse generation pipeline."""
        # Auto-download if needed
        auto_download_models()

        from muse_pipeline import MusePipeline

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        model_path = os.path.join(MODEL_BASE_DIR, model_name)

        # Use class variable explicitly
        if MuseModelManager._muse_pipe is None or MuseModelManager._muse_pipe.model_path != model_path:
            if MuseModelManager._muse_pipe is not None:
                MuseModelManager._muse_pipe.unload()
                MuseModelManager._muse_pipe = None

            MuseModelManager._muse_pipe = MusePipeline(
                model_path=model_path,
                device=MuseModelManager._device,
                dtype=dtype_map.get(dtype, torch.bfloat16),
                lazy_load=True,
            )
            torch.cuda.empty_cache()
            gc.collect()

        return MuseModelManager._muse_pipe

    def get_codec_pipeline(self, codec_name: str = "mucodec.pt"):
        """Get or create the MuCodec pipeline."""
        # Auto-download if needed
        auto_download_models()

        from muse_pipeline import MuCodecPipeline

        codec_path = os.path.join(MODEL_BASE_DIR, "mucodec", codec_name)

        # Use class variable explicitly
        if MuseModelManager._codec_pipe is None:
            MuseModelManager._codec_pipe = MuCodecPipeline(
                codec_path=codec_path,
                device=MuseModelManager._device,
                lazy_load=True,
            )
            torch.cuda.empty_cache()
            gc.collect()

        return MuseModelManager._codec_pipe

    @classmethod
    def unload_muse(cls):
        """Unload just the Muse model to free VRAM for decoding."""
        if cls._muse_pipe is not None:
            print("[Muse] Unloading Muse model to free VRAM...")
            cls._muse_pipe.unload()
            cls._muse_pipe = None
            # Aggressive cleanup
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[Muse] VRAM after unload: {allocated:.2f}GB")

    @classmethod
    def unload_all(cls):
        """Unload all models from memory."""
        if cls._muse_pipe is not None:
            cls._muse_pipe.unload()
            cls._muse_pipe = None
        if cls._codec_pipe is not None:
            cls._codec_pipe.unload()
            cls._codec_pipe = None
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def is_muse_loaded(cls):
        """Check if Muse model is currently loaded."""
        return cls._muse_pipe is not None and cls._muse_pipe.model is not None

    @classmethod
    def unload_codec(cls):
        """Unload just the MuCodec to free VRAM for generation."""
        if cls._codec_pipe is not None:
            print("[Muse] Unloading MuCodec to free VRAM...")
            cls._codec_pipe.unload()
            cls._codec_pipe = None
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"[Muse] VRAM after codec unload: {allocated:.2f}GB")

    @classmethod
    def is_codec_loaded(cls):
        """Check if MuCodec is currently loaded."""
        return cls._codec_pipe is not None and cls._codec_pipe.codec is not None


class Muse_Generate:
    """Generate music from lyrics and style using Muse."""

    @classmethod
    def INPUT_TYPES(cls):
        model_list = []
        if os.path.exists(MODEL_BASE_DIR):
            model_list = [d for d in os.listdir(MODEL_BASE_DIR)
                         if os.path.isdir(os.path.join(MODEL_BASE_DIR, d))
                         and not d.startswith('.') and d != 'mucodec']
        if not model_list:
            model_list = ["Muse-0.6b"]

        return {
            "required": {
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[Verse]\nYour lyrics here\n\n[Chorus]\nCatchy hook here",
                    "dynamicPrompts": False,
                    "placeholder": "[Verse]\nLyrics...\n\n[Chorus]\nHook...",
                }),
                "style_description": ("STRING", {
                    "multiline": True,
                    "default": "Pop, upbeat, synth, 120 BPM",
                    "dynamicPrompts": False,
                    "placeholder": "Genre, mood, instruments, tempo",
                }),
                "duration_seconds": ("INT", {
                    "default": 30,
                    "min": 5,
                    "max": 300,
                    "step": 5,
                    "display": "slider",
                }),
                "instrumental": ("BOOLEAN", {"default": False}),
                "model_name": (model_list, {"default": model_list[0]}),
                "dtype": (["bfloat16", "float16"], {"default": "bfloat16"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MUSE_TOKENS", "STRING")
    RETURN_NAMES = ("tokens", "info")
    FUNCTION = "generate"
    CATEGORY = "Muse"

    def generate(self, lyrics, style_description, duration_seconds, instrumental,
                 model_name, dtype, seed, temperature=1.0, top_p=0.9):
        from muse_pipeline import MuseConfig

        # Unload MuCodec first to free VRAM for Muse model
        if MuseModelManager.is_codec_loaded():
            MuseModelManager.unload_codec()

        # Convert duration to tokens (roughly 25 tokens per second)
        tokens_per_second = 25
        max_tokens = int(duration_seconds * tokens_per_second)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)

        manager = MuseModelManager()
        pipe = manager.get_muse_pipeline(model_name, dtype)

        # Build style string
        style = style_description.strip() if style_description.strip() else "Music"
        if instrumental:
            style = f"{style}, Instrumental, No Vocals"

        # Handle empty lyrics for instrumental
        if not lyrics.strip():
            if instrumental:
                lyrics = "[Instrumental]"
            else:
                lyrics = "[Verse]\n(lyrics here)"

        config = MuseConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=20,  # Model's recommended value
            repetition_penalty=1.1,
            do_sample=True,
        )

        print(f"[Muse] Generating {duration_seconds}s audio (~{max_tokens} tokens)")
        print(f"[Muse] Style: {style[:100]}...")
        print(f"[Muse] Instrumental: {instrumental}")

        try:
            with torch.inference_mode():
                result = pipe.generate(
                    lyrics=lyrics,
                    global_style=style,
                    segment_styles=None,
                    config=config,
                    seed=seed,
                )

            num_tokens = len(result.get("tokens", []))
            actual_duration = num_tokens / tokens_per_second
            info = f"Generated {num_tokens} tokens (~{actual_duration:.1f}s audio)"
            print(f"[Muse] {info}")

            return (result, info)

        except Exception as e:
            print(f"[Muse] Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return ({"tokens": [], "raw_text": str(e)}, f"Error: {e}")


class Muse_Decode:
    """Decode Muse audio tokens to waveform using MuCodec."""

    @classmethod
    def INPUT_TYPES(cls):
        codec_dir = os.path.join(MODEL_BASE_DIR, "mucodec")
        codec_list = []
        if os.path.exists(codec_dir):
            codec_list = [f for f in os.listdir(codec_dir) if f.endswith('.pt')]
        if not codec_list:
            codec_list = ["mucodec.pt"]

        return {
            "required": {
                "tokens": ("MUSE_TOKENS",),
                "codec_name": (codec_list, {"default": codec_list[0]}),
                "sample_rate": (["48000", "24000", "16000"], {"default": "48000"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "filepath")
    FUNCTION = "decode"
    CATEGORY = "Muse"

    def decode(self, tokens, codec_name, sample_rate):
        token_list = tokens.get("tokens", [])
        raw_text = tokens.get("raw_text", "")
        sample_rate = int(sample_rate)

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"muse_{uuid.uuid4().hex[:8]}.wav"
        out_path = os.path.join(output_dir, filename)

        # Always unload Muse before decoding to prevent OOM
        # The two models together exceed most GPU memory limits
        if MuseModelManager.is_muse_loaded():
            MuseModelManager.unload_muse()

        if not token_list:
            print(f"[Muse] No tokens to decode")
            silence = torch.zeros(1, 1, sample_rate)
            return ({"waveform": silence, "sample_rate": sample_rate}, "No tokens")

        print(f"[Muse] Decoding {len(token_list)} tokens...")

        try:
            codec = MuseModelManager().get_codec_pipeline(codec_name)

            with torch.inference_mode():
                result = codec.decode(token_list, sample_rate)

            waveform = result["waveform"]
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)

            # Save audio
            wav_np = waveform.cpu().numpy()
            if wav_np.ndim == 2:
                wav_np = wav_np.T
            sf.write(out_path, wav_np, sample_rate)

            duration = waveform.shape[-1] / sample_rate
            print(f"[Muse] Saved {duration:.1f}s audio to {out_path}")

            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)

            return ({"waveform": waveform, "sample_rate": sample_rate}, out_path)

        except Exception as e:
            print(f"[Muse] Decode failed: {e}")
            import traceback
            traceback.print_exc()
            silence = torch.zeros(1, 1, sample_rate)
            return ({"waveform": silence, "sample_rate": sample_rate}, f"Error: {e}")


class Muse_UnloadModels:
    """Unload all Muse models from memory."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Muse"
    OUTPUT_NODE = True

    def unload(self):
        manager = MuseModelManager()
        manager.unload_all()
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            return (f"Unloaded. VRAM: {allocated:.2f}GB",)
        return ("Models unloaded",)


NODE_CLASS_MAPPINGS = {
    "Muse_Generate": Muse_Generate,
    "Muse_Decode": Muse_Decode,
    "Muse_UnloadModels": Muse_UnloadModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Muse_Generate": "Muse Music Generator",
    "Muse_Decode": "Muse Audio Decoder",
    "Muse_UnloadModels": "Muse Unload Models",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
__version__ = "2.0.0"
