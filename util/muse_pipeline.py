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
    """Configuration for Muse model.

    Default values match the model's generation_config.json for optimal results.
    """
    max_new_tokens: int = 3000
    temperature: float = 0.6  # Model default (was 1.0)
    top_p: float = 0.95       # Model default (was 0.9)
    top_k: int = 20           # Model default (was 50)
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

    def _parse_lyrics_sections(self, lyrics: str) -> List[Dict[str, str]]:
        """Parse lyrics into sections based on [Section] markers."""
        import re
        sections = []

        # Split by section markers like [Verse], [Chorus], etc.
        pattern = r'\[([^\]]+)\]'
        parts = re.split(pattern, lyrics)

        # parts will be: ['', 'Verse', 'lyrics...', 'Chorus', 'lyrics...', ...]
        i = 1
        while i < len(parts):
            section_name = parts[i].strip()
            section_text = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if section_text:
                sections.append({
                    "section": section_name,
                    "text": section_text
                })
            i += 2

        # If no sections found, treat entire lyrics as one verse
        if not sections and lyrics.strip():
            sections.append({
                "section": "Verse",
                "text": lyrics.strip()
            })

        return sections

    def _format_prompt(self, lyrics: str, global_style: str, segment_styles: Dict[str, str] = None) -> str:
        """Format the prompt for Muse model input.

        Matches the training data format from the Muse dataset:
        - Style: comma-separated style tags
        - Sections with section name, lyrics, and musical description
        """
        # Check if instrumental mode
        is_instrumental = (
            "instrumental" in global_style.lower() or
            "no vocal" in global_style.lower() or
            lyrics.strip().lower() == "[instrumental]" or
            not lyrics.strip()
        )

        # Build structured prompt matching training format
        content = f"Style: {global_style}\n\n"

        if is_instrumental:
            content += "[Instrumental]\n"
            content += "Generate instrumental music with no vocals.\n"
        else:
            # Parse lyrics into sections
            sections = self._parse_lyrics_sections(lyrics)

            for section in sections:
                content += f"[{section['section']}]\n"
                content += f"{section['text']}\n\n"

        return content

    def _build_messages(self, user_content: str) -> List[Dict[str, str]]:
        """Build chat messages for the model."""
        # Simple user message - model was trained to generate [SOA]..audio tokens..[EOA]
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

        # Apply chat template (thinking disabled to maximize audio tokens)
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate - disable EOS stopping to ensure we get the full requested duration
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                min_new_tokens=config.max_new_tokens,  # Force generation of full length
                temperature=config.temperature if config.do_sample else None,
                top_p=config.top_p if config.do_sample else None,
                top_k=config.top_k if config.do_sample else None,
                repetition_penalty=config.repetition_penalty,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=[],  # Disable early stopping on EOS
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

        Muse outputs audio tokens in <AUDIO_XXXXX> format, wrapped in [SOA]...[EOA].
        The XXXXX values are MuCodec codebook indices (0-16383).
        """
        # Remove thinking tags if present
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Try to extract content between [SOA] and [EOA] markers
        soa_match = re.search(r'\[SOA\](.*?)(?:\[EOA\]|$)', text, flags=re.DOTALL)
        if soa_match:
            audio_section = soa_match.group(1)
        else:
            audio_section = text

        # Find all AUDIO tokens in <AUDIO_XXXXX> format
        tokens = re.findall(r'<AUDIO_(\d+)>', audio_section)
        token_ids = [int(t) for t in tokens]

        # The XXXXX in <AUDIO_XXXXX> is already the MuCodec code (0-16383)
        max_codebook = 16383

        # Clamp to valid codebook range (0-16383)
        codes = [max(0, min(t, max_codebook)) for t in token_ids]

        if codes:
            print(f"[Muse] Parsed {len(codes)} audio tokens (range: {min(codes)}-{max(codes)})")

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

        if codes.ndim == 1:
            codes = codes.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
        elif codes.ndim == 2:
            codes = codes.unsqueeze(0)  # (1, T) -> (1, 1, T)

        # Memory cleanup before decode
        gc.collect()
        torch.cuda.empty_cache()

        # Calculate decode duration from token count (~25 tokens/second)
        tokens_per_second = 25.0
        actual_duration = len(tokens) / tokens_per_second
        chunk_size = 10.24  # MuCodec's internal chunk size
        decode_duration = max(chunk_size, min(120.0,
            ((actual_duration // chunk_size) + 1) * chunk_size))

        # Decode using MuCodec's code2sound
        waveform = self.codec.code2sound(
            codes,
            prompt=None,
            duration=decode_duration,
            guidance_scale=1.5,
            num_steps=15,  # Reduced for lower memory usage
            disable_progress=False,
        )

        waveform = waveform.detach().cpu().float()

        # Resample if needed
        if sample_rate != self.NATIVE_SAMPLE_RATE:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, self.NATIVE_SAMPLE_RATE, sample_rate
            )

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
