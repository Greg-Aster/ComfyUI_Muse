"""
Muse Inference Pipeline - Transformers-based inference for Muse music generation.

Based on: https://github.com/yuhui1038/Muse
"""

import os
import re
import torch
import gc
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class MuseConfig:
    """Configuration for Muse model."""
    max_new_tokens: int = 3000
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class MusePipeline:
    """
    Pipeline for Muse music generation using transformers.
    Generates audio tokens from lyrics and style descriptions.
    """

    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        lazy_load: bool = True,
    ):
        self.model_path = model_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model = None
        self.tokenizer = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the Muse model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.model is not None:
            return

        print(f"Loading Muse model from {self.model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.dtype,
            trust_remote_code=True,
            device_map="auto",
        )

        self.model.eval()
        print("Muse model loaded successfully.")

    def _ensure_loaded(self):
        """Ensure model is loaded before inference."""
        if self.model is None:
            self._load_model()

    def _format_prompt(self, lyrics: str, global_style: str, segment_styles: Dict[str, str] = None) -> str:
        """Format the prompt for Muse model input."""
        # Check if instrumental mode (detected by keywords in style or lyrics)
        is_instrumental = (
            "instrumental" in global_style.lower() or
            "no vocal" in global_style.lower() or
            lyrics.strip().lower() == "[instrumental]"
        )

        if is_instrumental:
            # Strong instrumental prompt - emphasize no vocals multiple ways
            content = f"Generate an INSTRUMENTAL piece of music with NO VOCALS, NO SINGING, NO VOICE.\n\n"
            content += f"Style: {global_style}\n\n"
            content += "This is purely instrumental music. Do not include any vocals, singing, humming, or voice.\n"
            content += "Focus on: melody, harmony, rhythm, instruments only.\n"
        else:
            content = f"Generate a song with the following specifications:\n\n"
            content += f"Global Style: {global_style}\n\n"
            content += f"Lyrics:\n{lyrics}\n"

            if segment_styles:
                content += "\nSegment-specific styles:\n"
                for segment, style in segment_styles.items():
                    content += f"- {segment.capitalize()}: {style}\n"

        return content

    def _build_messages(self, user_content: str) -> List[Dict[str, str]]:
        """Build chat messages for the model."""
        return [
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        lyrics: str,
        global_style: str,
        segment_styles: Dict[str, str] = None,
        config: MuseConfig = None,
        seed: int = -1,
    ) -> Dict[str, Any]:
        """
        Generate audio tokens from lyrics and style.

        Args:
            lyrics: Song lyrics with section markers
            global_style: Overall style description
            segment_styles: Optional per-segment styles
            config: Generation configuration
            seed: Random seed (-1 for random)

        Returns:
            Dictionary with raw_text and parsed tokens
        """
        self._ensure_loaded()

        if config is None:
            config = MuseConfig()

        # Set seed if specified
        if seed >= 0:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        # Format prompt
        user_content = self._format_prompt(lyrics, global_style, segment_styles)
        messages = self._build_messages(user_content)

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                top_k=config.top_k if config.do_sample else None,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse audio tokens from generated text
        tokens = self._parse_audio_tokens(generated_text)

        return {
            "raw_text": generated_text,
            "tokens": tokens,
        }

    def _parse_audio_tokens(self, text: str) -> List[int]:
        """Parse audio tokens from generated text.

        Muse outputs audio tokens in <AUDIO_XXXXX> format.
        These need to be converted to MuCodec codes by subtracting vocab_offset.
        """
        # Remove thinking tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Find all AUDIO tokens in <AUDIO_XXXXX> format
        tokens = re.findall(r'<AUDIO_(\d+)>', text)
        token_ids = [int(t) for t in tokens]

        print(f"[Muse] Found {len(token_ids)} audio tokens in output")
        if token_ids:
            print(f"[Muse] Raw token IDs: min={min(token_ids)}, max={max(token_ids)}")
            print(f"[Muse] First 5 raw tokens: {token_ids[:5]}")

        # Check if tokens need vocab offset adjustment
        # Muse uses Qwen tokenizer with vocab_size=151643. Audio tokens are added
        # after the text vocab, so raw token IDs may be offset by 151643.
        # MuCodec codebook has 16384 entries (0-16383).
        # If tokens are already in valid range, use directly; otherwise subtract offset.
        vocab_offset = 151643  # Qwen tokenizer vocab size
        max_codebook = 16383   # MuCodec codebook max index

        if token_ids and max(token_ids) > max_codebook:
            # Tokens have vocab offset applied, need to subtract
            print(f"[Muse] Tokens appear offset-adjusted, subtracting {vocab_offset}")
            codes = [t - vocab_offset for t in token_ids]
        else:
            # Tokens are already raw MuCodec codes
            print(f"[Muse] Tokens are raw MuCodec codes, using directly")
            codes = token_ids

        if codes:
            print(f"[Muse] Codes before clamping: min={min(codes)}, max={max(codes)}")

        # Clamp to valid codebook range (0-16383)
        codes = [max(0, min(c, max_codebook)) for c in codes]

        if codes:
            print(f"[Muse] Final codes (0-16383): min={min(codes)}, max={max(codes)}")

        return codes

    def offload(self):
        """Offload model from GPU to free memory."""
        if self.model is not None:
            self.model.cpu()
            torch.cuda.empty_cache()
            gc.collect()

    def unload(self):
        """Completely unload model from memory."""
        if self.model is not None:
            # For models loaded with device_map="auto", need aggressive cleanup
            try:
                # Try to remove accelerate hooks if present
                if hasattr(self.model, '_hf_hook'):
                    delattr(self.model, '_hf_hook')
                if hasattr(self.model, 'hf_device_map'):
                    delattr(self.model, 'hf_device_map')
            except:
                pass

            try:
                # Move to CPU first to free GPU memory
                self.model.to('cpu')
            except:
                pass

            try:
                # Clear all submodules
                for name, module in list(self.model.named_modules()):
                    if hasattr(module, '_parameters'):
                        for param_name in list(module._parameters.keys()):
                            module._parameters[param_name] = None
            except:
                pass

            # Clear the reference
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Aggressive cleanup
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


class MuCodecPipeline:
    """
    Pipeline for MuCodec audio decoding.
    Converts audio tokens to waveform using the bundled MuCodec.
    """

    # MuCodec native sample rate
    NATIVE_SAMPLE_RATE = 48000

    def __init__(
        self,
        codec_path: str,
        audioldm_path: str = None,
        device: torch.device = None,
        lazy_load: bool = True,
    ):
        self.codec_path = codec_path
        self.audioldm_path = audioldm_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.codec = None

        if not lazy_load:
            self._load_codec()

    def _load_codec(self):
        """Load the MuCodec model."""
        if self.codec is not None:
            return

        print(f"[MuCodec] Loading from {self.codec_path}...")
        print(f"[MuCodec] Device: {self.device}")

        try:
            # Import the bundled MuCodec using absolute path
            import sys
            import os
            mucodec_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mucodec")
            if mucodec_dir not in sys.path:
                sys.path.insert(0, mucodec_dir)

            from generate import MuCodec

            # MuCodec expects the audioldm weights in mucodec/tools/audioldm_48k.pth
            self.codec = MuCodec(
                model_path=self.codec_path,
                layer_num=7,
                load_main_model=True,
                device=str(self.device),
            )
            print("[MuCodec] Loaded successfully.")

        except Exception as e:
            raise RuntimeError(
                f"[MuCodec] Failed to load: {e}\n"
                f"Make sure you have downloaded the required weights:\n"
                f"  1. mucodec.pt -> ComfyUI/models/muse/mucodec/mucodec.pt\n"
                f"  2. audioldm_48k.pth -> ComfyUI/custom_nodes/ComfyUI_Muse/mucodec/tools/audioldm_48k.pth"
            )

    def _ensure_loaded(self):
        """Ensure codec is loaded before decoding."""
        if self.codec is None:
            self._load_codec()

    def decode(self, tokens: List[int], sample_rate: int = 48000) -> Dict[str, Any]:
        """
        Decode audio tokens to waveform.

        Args:
            tokens: List of MuCodec code values (already offset-adjusted)
            sample_rate: Output sample rate (will resample if different from 48kHz)

        Returns:
            Dictionary with waveform tensor and sample_rate
        """
        self._ensure_loaded()

        if not tokens:
            raise ValueError("No audio tokens provided for decoding")

        # Convert to tensor with shape (batch, 1, time)
        codes = torch.tensor(tokens, dtype=torch.long)

        # Debug: show token statistics
        print(f"[MuCodec] Token stats: min={min(tokens)}, max={max(tokens)}, unique={len(set(tokens))}")
        print(f"[MuCodec] First 10 tokens: {tokens[:10]}")

        if codes.ndim == 1:
            codes = codes.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
        elif codes.ndim == 2:
            codes = codes.unsqueeze(0)  # (1, T) -> (1, 1, T)

        print(f"[MuCodec] Decoding {codes.shape[-1]} tokens, shape={codes.shape}...")

        # Aggressive memory cleanup before decode to prevent fragmentation
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[MuCodec] VRAM before decode: {allocated:.2f}GB")

        # Monitor resources during decode
        import psutil
        import time
        process = psutil.Process()
        cpu_before = process.cpu_percent()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        start_time = time.time()

        # Calculate actual duration from token count (~25 tokens/second)
        # Add small buffer for safety, but don't over-allocate
        tokens_per_second = 25.0
        actual_duration = len(tokens) / tokens_per_second
        # Clamp to reasonable range: min 10.24s (MuCodec minimum), max 120s
        # Round up to nearest 10.24s chunk (MuCodec's internal chunk size)
        chunk_size = 10.24
        decode_duration = max(chunk_size, min(120.0,
            ((actual_duration // chunk_size) + 1) * chunk_size))

        print(f"[MuCodec] Token duration: {actual_duration:.1f}s, decode buffer: {decode_duration:.1f}s")

        # Decode using MuCodec's code2sound
        # Using fewer diffusion steps (15 instead of 20) to reduce peak memory
        waveform = self.codec.code2sound(
            codes,
            prompt=None,
            duration=decode_duration,  # Dynamic based on actual token count
            guidance_scale=1.5,
            num_steps=15,  # Reduced from 20 for lower memory usage
            disable_progress=False,
        )

        elapsed = time.time() - start_time
        cpu_after = process.cpu_percent()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        print(f"[MuCodec] Decode took {elapsed:.2f}s, CPU: {cpu_after:.1f}%, RAM: {mem_after:.0f}MB (delta: {mem_after-mem_before:+.0f}MB)")

        # waveform shape is (channels, samples) at 48kHz
        waveform = waveform.detach().cpu().float()

        # Debug: show waveform statistics
        print(f"[MuCodec] Waveform stats: min={waveform.min().item():.4f}, max={waveform.max().item():.4f}, mean={waveform.mean().item():.4f}, std={waveform.std().item():.4f}")

        # Resample if needed
        if sample_rate != self.NATIVE_SAMPLE_RATE:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, self.NATIVE_SAMPLE_RATE, sample_rate
            )

        print(f"[MuCodec] Decoded audio: {waveform.shape}, {sample_rate}Hz")

        return {
            "waveform": waveform,
            "sample_rate": sample_rate,
        }

    def offload(self):
        """Offload codec from GPU."""
        # MuCodec doesn't have a simple offload, just clear cache
        torch.cuda.empty_cache()
        gc.collect()

    def unload(self):
        """Completely unload codec from memory."""
        if self.codec is not None:
            del self.codec
            self.codec = None
        torch.cuda.empty_cache()
        gc.collect()
