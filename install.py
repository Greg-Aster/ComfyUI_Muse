#!/usr/bin/env python3
"""
ComfyUI_Muse Installation Script
Downloads required model weights for Muse music generation.
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Weight download locations
WEIGHTS = {
    "muse_model": {
        "name": "Muse-0.6b",
        "source": "huggingface",
        "repo": "bolshyC/Muse-0.6b",
        "dest": "../../models/muse/Muse-0.6b",  # ComfyUI/models/muse/Muse-0.6b
    },
    "mucodec": {
        "name": "MuCodec checkpoint",
        "source": "url",
        "url": "https://huggingface.co/AcademiCodec/MuCodec/resolve/main/mucodec.pt",
        "dest": "../../models/muse/mucodec/mucodec.pt",
    },
    "audioldm": {
        "name": "AudioLDM VAE",
        "source": "url",
        "url": "https://huggingface.co/AcademiCodec/MuCodec/resolve/main/audioldm_48k.pth",
        "dest": "mucodec/tools/audioldm_48k.pth",
    },
}


def download_hf_repo(repo_id: str, dest_path: str):
    """Download a HuggingFace repository."""
    print(f"Downloading {repo_id} to {dest_path}...")
    try:
        subprocess.run(
            ["huggingface-cli", "download", repo_id, "--local-dir", dest_path],
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print("Error: huggingface-cli not found. Install with: pip install huggingface_hub")
        return False


def format_size(bytes):
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f}{unit}"
        bytes /= 1024
    return f"{bytes:.1f}TB"


def download_progress(block_num, block_size, total_size):
    """Progress callback for urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        downloaded_str = format_size(downloaded)
        total_str = format_size(total_size)
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\r  [{bar}] {percent:5.1f}% ({downloaded_str}/{total_str})", end='', flush=True)
    else:
        print(f"\r  Downloaded: {format_size(downloaded)}", end='', flush=True)


def download_file(url: str, dest_path: str):
    """Download a file from URL with progress indicator."""
    print(f"Downloading {url}")
    try:
        import urllib.request
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        urllib.request.urlretrieve(url, dest_path, reporthook=download_progress)
        print()  # Newline after progress bar
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def main():
    print("=" * 60)
    print("ComfyUI_Muse Installation")
    print("=" * 60)
    print()

    os.chdir(SCRIPT_DIR)

    for key, weight in WEIGHTS.items():
        dest = os.path.normpath(os.path.join(SCRIPT_DIR, weight["dest"]))

        # Check if already exists
        if os.path.exists(dest):
            print(f"[OK] {weight['name']} already exists at {dest}")
            continue

        print(f"\n[DOWNLOAD] {weight['name']}")

        if weight["source"] == "huggingface":
            success = download_hf_repo(weight["repo"], dest)
        elif weight["source"] == "url":
            success = download_file(weight["url"], dest)
        else:
            print(f"Unknown source type: {weight['source']}")
            success = False

        if success:
            print(f"[OK] Downloaded {weight['name']}")
        else:
            print(f"[FAIL] Could not download {weight['name']}")
            print(f"       Please download manually and place at: {dest}")

    print()
    print("=" * 60)
    print("Installation complete!")
    print()
    print("If any downloads failed, you can manually download:")
    print("  Muse model: huggingface-cli download bolshyC/Muse-0.6b --local-dir ComfyUI/models/muse/Muse-0.6b")
    print("  MuCodec: https://huggingface.co/AcademiCodec/MuCodec/resolve/main/mucodec.pt")
    print("  AudioLDM: https://huggingface.co/AcademiCodec/MuCodec/resolve/main/audioldm_48k.pth")
    print("=" * 60)


if __name__ == "__main__":
    main()
