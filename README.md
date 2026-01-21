# ComfyUI_Muse

---

## ⚠️ EXPERIMENTAL RESEARCH PROJECT ⚠️

**THIS IS NOT A PRODUCTION-READY MUSIC GENERATOR**

This is a ComfyUI wrapper around a 0.6B parameter research model. Set your expectations accordingly.

---

## ❌ LYRICS DO NOT WORK

**The model CANNOT produce recognizable sung words.** The "lyrics" input affects melody rhythm and vocal-like sounds (humming, vocoded textures), but **you will NOT hear actual words being sung**. This is a fundamental limitation of the underlying Muse 0.6B model, not a bug in this implementation.

If you need AI-generated music with actual lyrics, this is not the tool for you.

---

## What This Actually Does

ComfyUI node package for **Muse** - an open-source AI music generation model from [Fudan NLP Lab](https://github.com/yuhui1038/Muse).

- Generates **melody and instrumental textures** based on style descriptions
- Produces **vocal-like sounds** (hums, vocoded sounds) - NOT intelligible lyrics
- Useful for **style exploration and melody generation experiments**

**GitHub:** https://github.com/Greg-Aster/ComfyUI_Muse

Based on: ["Muse: Towards Reproducible Long-Form Song Generation"](https://arxiv.org/abs/2601.03973)

## Known Limitations

| What You Might Expect | What Actually Happens |
|----------------------|----------------------|
| Lyrics sung clearly | Humming/vocoded sounds that follow rhythm |
| "Instrumental" = no vocals | May still have vocal-like textures |
| Consistent quality | Varies wildly by style and seed |
| Production-ready output | Research-grade experimental output |

## Features

- **One-click install** - Models auto-download on first use (~10GB)
- **Style-influenced generation** - Describe any style, genre, mood, instruments
- **Duration control** - 5-300 seconds
- **Memory efficient** - Automatic model swapping for 12-16GB GPUs

## Quick Start

### 1. Install the Package

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Greg-Aster/ComfyUI_Muse.git
cd ComfyUI_Muse
pip install -r requirements.txt
```

### 2. Restart ComfyUI and Load Workflow

Load the example workflow: **[example_workflows/basic_song_generation.json](example_workflows/basic_song_generation.json)**

Models download automatically on first run:
- Muse-0.6b (~1.2GB)
- MuCodec (~4GB)
- AudioLDM VAE (~5GB)

## Nodes

| Node | Description |
|------|-------------|
| **Muse Music Generator** | Generate audio tokens from lyrics and style description |
| **Muse Audio Decoder** | Convert tokens to waveform using MuCodec |
| **Muse Unload Models** | Manually free VRAM |

## Basic Workflow

```
[Muse Music Generator] → [Muse Audio Decoder] → [Preview Audio]
```

## Lyrics Format (Affects Rhythm/Melody Only)

> **Remember:** Lyrics do NOT produce sung words. They influence the melodic rhythm and phrasing.

Use section markers to structure your song:

```
[Verse]
Walking down the empty street
Memories beneath my feet

[Chorus]
But I keep moving on
```

The text affects:
- **Syllable rhythm** - More syllables = faster melodic phrasing
- **Section structure** - [Verse], [Chorus], etc. create musical transitions
- **NOT the actual words** - You will hear hums/vocoded sounds, not these words

**For instrumental:** Enable the `instrumental` toggle (note: may still have vocal-like textures).

## Style Examples

Describe the style in natural language:

- `"Pop, upbeat, synth, piano, 120 BPM"`
- `"Rock ballad, emotional, electric guitar, drums"`
- `"Lo-fi hip hop, chill, jazzy piano, vinyl crackle"`
- `"EDM, energetic, drop, bass heavy, festival vibes"`
- `"Acoustic folk, warm, fingerpicking guitar, intimate"`
- `"90s R&B, smooth, groovy bass"`

Note: Style tags like "female vocals" or "male vocals" may influence the timbre of vocal-like sounds but will not produce intelligible singing.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12GB | 16GB+ |
| System RAM | 16GB | 32GB+ |
| Storage | 12GB | 15GB |

## Memory Management

The Muse generator (~10GB) and MuCodec decoder (~5GB) together exceed most GPU limits. The system **automatically swaps** between them:

1. Before generation: Unloads MuCodec, loads Muse
2. Before decoding: Unloads Muse, loads MuCodec

This allows the workflow to run on 12-16GB GPUs.

## Troubleshooting

### "Model not found" / Download fails
Models should auto-download. If not, manually download:
```bash
# Muse model
huggingface-cli download bolshyC/Muse-0.6b --local-dir ComfyUI/models/muse/Muse-0.6b

# MuCodec
mkdir -p ComfyUI/models/muse/mucodec
wget -O ComfyUI/models/muse/mucodec/mucodec.pt https://huggingface.co/AcademiCodec/MuCodec/resolve/main/mucodec.pt

# AudioLDM VAE (inside the node package)
wget -O ComfyUI/custom_nodes/ComfyUI_Muse/mucodec/tools/audioldm_48k.pth https://huggingface.co/AcademiCodec/MuCodec/resolve/main/audioldm_48k.pth
```

### "CUDA out of memory"
- Reduce duration (shorter = less memory)
- Use `dtype: bfloat16` in the Generator
- Restart ComfyUI to clear fragmented memory

### No audio output (silence)
- Check console for token count - should be ~25 tokens/second
- Ensure all model weights are downloaded
- Try a different seed

## Technical Notes

- **Token format**: Muse outputs `<AUDIO_XXXXX>` tokens where XXXXX is a MuCodec codebook index (0-16383)
- **Sample rate**: Native 48kHz, can downsample to 24kHz or 16kHz
- **Duration mapping**: ~25 tokens per second of audio

## Credits

- **Muse**: [Fudan NLP Lab](https://github.com/yuhui1038/Muse)
- **MuCodec**: [Tencent AI Lab](https://github.com/AcademiCodec/MuCodec)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## License

MIT License

## Links

- Paper: [arXiv:2601.03973](https://arxiv.org/abs/2601.03973)
- Muse: https://github.com/yuhui1038/Muse
- MuCodec: https://github.com/AcademiCodec/MuCodec
