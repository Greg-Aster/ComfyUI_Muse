# ComfyUI_Muse

## AI Instrumental Music Generator

Generate instrumental music and melodies from style descriptions using Muse.

**GitHub:** https://github.com/Greg-Aster/ComfyUI_Muse

Based on: [Muse 0.6B](https://github.com/yuhui1038/Muse) from Fudan NLP Lab

---

## What This Does

- **Generates instrumental music** from text style descriptions
- **No lyrics/vocals** - this is instrumental-only by design
- **Style-driven** - describe genre, mood, instruments, tempo
- **Variable duration** - 5 to 300 seconds

## Important Notes

| This is... | This is NOT... |
|------------|----------------|
| A 0.6B research model | A production music tool |
| Experimental/variable quality | Consistently high quality |
| Good for melody exploration | A replacement for real composition |

## Quick Start

### 1. Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Greg-Aster/ComfyUI_Muse.git
cd ComfyUI_Muse
pip install -r requirements.txt
```

### 2. Restart ComfyUI

Models download automatically on first run (~10GB total).

### 3. Basic Workflow

```
[Muse Instrumental Generator] → [Muse Audio Decoder] → [Preview Audio]
```

## Nodes

| Node | Description |
|------|-------------|
| **Muse Instrumental Generator** | Generate audio tokens from style description |
| **Muse Audio Decoder** | Convert tokens to waveform |
| **Muse Unload Models** | Free VRAM |

## Style Examples

Describe your desired music style:

```
Lo-fi hip hop, chill, jazzy piano, warm bass, vinyl crackle, 85 BPM
```

```
Epic orchestral, cinematic, strings, brass, dramatic, 90 BPM
```

```
Ambient electronic, atmospheric, synth pads, ethereal, slow
```

```
Acoustic folk, fingerpicking guitar, warm, intimate, 100 BPM
```

```
EDM, energetic, bass heavy, synth leads, festival vibes, 128 BPM
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12GB | 16GB+ |
| System RAM | 16GB | 32GB+ |
| Storage | 12GB | 15GB |

## Troubleshooting

### CUDA out of memory
- Reduce duration
- Use bfloat16 dtype
- Restart ComfyUI to clear memory

### No audio output
- Check console for token count
- Try different seed
- Verify models downloaded

## Credits

- **Muse**: [Fudan NLP Lab](https://github.com/yuhui1038/Muse)
- **MuCodec**: [Tencent AI Lab](https://github.com/AcademiCodec/MuCodec)
- **ComfyUI**: [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)

## License

MIT License
