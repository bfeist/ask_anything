#!/usr/bin/env python3
"""Step 3b: Transcode downloaded IA videos to web-ready format.

Reads MP4 files from the raw download directory (DOWNLOAD_DIR) and writes
web-optimised copies to an ``ask_anything_ia_videos_web`` sibling folder.

Transcoding rules (same targets as the ISSiRT transcode pipeline):
- Resolution  : ≤ 480p
- Video bitrate: ≤ 1 000 kb/s  (±10 % tolerance)
- Audio bitrate: ≤ 64 kb/s     (±5 % tolerance)

Per-file outcome:
- Already compliant → copy with ``-movflags +faststart``
- Only audio too high → copy video stream, re-encode audio to AAC 64 k
- Video too high / too tall → full 480p CBR transcode (GPU, falls back to CPU)

Usage:
    python scripts/3b_transcode_web.py [--dry-run] [--limit N] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from astro_ia_harvest.config import DOWNLOAD_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration constants  (mirrors ISSiRT 1_transcode_ia_raw_to_web.py)
# ---------------------------------------------------------------------------
TARGET_VIDEO_BITRATE = 1000          # kb/s
TARGET_AUDIO_BITRATE = 128           # kb/s — re-encode only if above this
VIDEO_BITRATE_TOLERANCE = 1.1        # allow 10 % above target
AUDIO_BITRATE_TOLERANCE = 1.05       # allow 5 % above target
DEFAULT_VIDEO_BITRATE = "1000k"
DEFAULT_AUDIO_BITRATE = "128k"

GPU_PRESET = "fast"
CPU_PRESET = "medium"

GOP_SIZE = 2          # seconds — keyframe interval for efficient seeking
MAX_B_FRAMES = 2
PROFILE = "main"
LEVEL = "3.1"

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2
MAX_HEIGHT = 480
MAX_WIDTH = 854

WEB_DIR_NAME = "ask_anything_ia_videos_web"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def default_web_dir() -> Path:
    """Place the web folder next to the raw download folder."""
    return DOWNLOAD_DIR.parent / WEB_DIR_NAME


# ---------------------------------------------------------------------------
# ffmpeg / ffprobe helpers
# ---------------------------------------------------------------------------


def get_video_info(video_path: Path, console: Console) -> dict | None:
    """Return ``{resolution, video_bitrate, audio_bitrate}`` via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        info: dict = {"resolution": None, "video_bitrate": None, "audio_bitrate": None}
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                w, h = stream.get("width"), stream.get("height")
                info["resolution"] = (w, h) if w and h else None
                br = stream.get("bit_rate")
                info["video_bitrate"] = int(br) // 1000 if br else None
            elif stream.get("codec_type") == "audio":
                br = stream.get("bit_rate")
                info["audio_bitrate"] = int(br) // 1000 if br else None
        return info
    except Exception as exc:
        console.print(f"[red]ffprobe error for {video_path.name}: {exc}[/red]")
        return None


def is_compliant(info: dict) -> bool:
    if not info or not info["resolution"]:
        return False
    _, h = info["resolution"]
    if h > MAX_HEIGHT:
        return False
    vbr = info["video_bitrate"]
    if vbr and vbr > TARGET_VIDEO_BITRATE * VIDEO_BITRATE_TOLERANCE:
        return False
    abr = info["audio_bitrate"]
    if abr and abr > TARGET_AUDIO_BITRATE * AUDIO_BITRATE_TOLERANCE:
        return False
    return True


# ---------------------------------------------------------------------------
# Transcode / copy operations
# ---------------------------------------------------------------------------


def _run(cmd: list[str], console: Console, label: str) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]✗ {label}: {exc}[/red]")
        if exc.stderr:
            console.print(f"[dim]{exc.stderr[:300]}[/dim]")
        return False


def copy_file(src: Path, dst: Path, console: Console) -> bool:
    ok = _run([
        "ffmpeg", "-i", str(src),
        "-c", "copy",
        "-movflags", "+faststart",
        "-fflags", "+genpts",
        "-y", str(dst),
    ], console, f"copy {src.name}")
    if ok:
        console.print(f"[green]✓[/green] Copied (faststart): {src.name} → {dst.name}")
    return ok


def transcode_audio_only(
    src: Path, dst: Path,
    audio_bitrate: str = DEFAULT_AUDIO_BITRATE,
    console: Console = None,
    progress: Progress = None,
    task_id: TaskID = None,
) -> bool:
    if progress and task_id is not None:
        progress.update(task_id, description=f"Audio transcode: {src.name}")
    ok = _run([
        "ffmpeg", "-i", str(src),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-ac", str(AUDIO_CHANNELS),
        "-profile:a", "aac_low",
        "-movflags", "+faststart",
        "-fflags", "+genpts",
        "-y", str(dst),
    ], console, f"audio-transcode {src.name}")
    if ok:
        console.print(f"[green]✓[/green] Audio re-encoded: {src.name} → {dst.name}")
    return ok


def transcode_to_480p(
    src: Path, dst: Path,
    video_bitrate: str = DEFAULT_VIDEO_BITRATE,
    audio_bitrate: str = DEFAULT_AUDIO_BITRATE,
    console: Console = None,
    progress: Progress = None,
    task_id: TaskID = None,
) -> bool:
    if progress and task_id is not None:
        progress.update(task_id, description=f"Transcoding 480p: {src.name}")

    gop = GOP_SIZE * 30
    buf = f"{int(video_bitrate.rstrip('k')) * 2}k"

    # --- GPU attempt (NVENC) ---
    gpu_cmd = [
        "ffmpeg", "-i", str(src),
        "-vf", "hwupload_cuda,scale_cuda=-2:480",
        "-c:v", "h264_nvenc",
        "-rc", "cbr",
        "-b:v", video_bitrate, "-maxrate", video_bitrate, "-bufsize", buf,
        "-g", str(gop), "-bf", str(MAX_B_FRAMES),
        "-profile:v", PROFILE, "-level:v", LEVEL,
        "-preset", GPU_PRESET,
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
        "-profile:a", "aac_low",
        "-movflags", "+faststart",
        "-fflags", "+genpts+igndts",
        "-avoid_negative_ts", "make_zero",
        "-y", str(dst),
    ]
    if _run(gpu_cmd, console, f"GPU {src.name}"):
        console.print(f"[green]✓[/green] GPU transcoded (480p CBR): {src.name} → {dst.name}")
        return True

    console.print("[yellow]⚠  GPU failed — falling back to CPU[/yellow]")

    # --- CPU fallback (libx264) ---
    cpu_cmd = [
        "ffmpeg", "-i", str(src),
        "-vf", "scale=-2:480",
        "-c:v", "libx264",
        "-crf", "23", "-maxrate", video_bitrate, "-bufsize", buf,
        "-g", str(gop), "-bf", str(MAX_B_FRAMES),
        "-profile:v", PROFILE, "-level:v", LEVEL,
        "-preset", CPU_PRESET, "-tune", "fastdecode",
        "-c:a", "aac", "-b:a", audio_bitrate,
        "-ar", str(AUDIO_SAMPLE_RATE), "-ac", str(AUDIO_CHANNELS),
        "-profile:a", "aac_low",
        "-movflags", "+faststart",
        "-fflags", "+genpts+igndts",
        "-avoid_negative_ts", "make_zero",
        "-y", str(dst),
    ]
    if _run(cpu_cmd, console, f"CPU {src.name}"):
        console.print(f"[green]✓[/green] CPU transcoded (480p): {src.name} → {dst.name}")
        return True

    return False


# ---------------------------------------------------------------------------
# System checks
# ---------------------------------------------------------------------------


def check_ffmpeg(console: Console) -> bool:
    import shutil
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            console.print(f"[red]{tool} not found in PATH[/red]")
            return False
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=10)
    except Exception as exc:
        console.print(f"[red]ffmpeg/ffprobe check failed: {exc}[/red]")
        return False
    console.print("[green]✓[/green] ffmpeg and ffprobe available")
    return True


def check_nvidia(console: Console) -> bool:
    import shutil
    smi = shutil.which("nvidia-smi")
    if not smi:
        console.print("[yellow]⚠  nvidia-smi not found — GPU unavailable[/yellow]")
        return False
    try:
        r = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        name = r.stdout.strip()
        if name:
            console.print(f"[green]✓[/green] NVIDIA GPU: {name}")
            return True
    except Exception:
        pass
    console.print("[yellow]⚠  No NVIDIA GPU detected[/yellow]")
    return False


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_videos(
    src_dir: Path,
    web_dir: Path,
    console: Console,
    dry_run: bool = False,
    limit: int = 0,
) -> bool:
    web_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(src_dir.glob("*.mp4"))
    if not mp4_files:
        console.print(f"[yellow]No MP4 files found in {src_dir}[/yellow]")
        return True

    if limit > 0:
        mp4_files = mp4_files[:limit]

    console.print(f"[blue]Source dir : {src_dir}[/blue]")
    console.print(f"[blue]Web dir    : {web_dir}[/blue]")
    console.print(f"[blue]Files found: {len(mp4_files)}[/blue]")
    if dry_run:
        console.print("[yellow]DRY RUN — no files will be written[/yellow]")
    console.print()

    processed = skipped = errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10,
    ) as progress:
        overall = progress.add_task("Processing videos…", total=len(mp4_files))

        for src_file in mp4_files:
            progress.update(overall, description=f"Processing: {src_file.name}")
            out_name = src_file.name
            dst_file = web_dir / out_name

            # --- Skip if up-to-date compliant output already exists ---
            if dst_file.exists():
                out_info = get_video_info(dst_file, console)
                if (
                    out_info
                    and is_compliant(out_info)
                    and dst_file.stat().st_mtime >= src_file.stat().st_mtime
                ):
                    console.print(f"[dim]⏭  Already up-to-date: {out_name}[/dim]")
                    skipped += 1
                    progress.advance(overall)
                    continue
                console.print(f"[dim]Output exists but outdated/non-compliant — will redo[/dim]")

            # --- Probe source ---
            info = get_video_info(src_file, console)
            if not info or not info["resolution"]:
                console.print(f"[red]✗ Cannot probe {src_file.name} — skipping[/red]")
                errors += 1
                progress.advance(overall)
                continue

            w, h = info["resolution"]
            vbr = info["video_bitrate"]
            abr = info["audio_bitrate"]
            console.print(
                f"[dim]{src_file.name}  {w}x{h}"
                f"  video={vbr or '?'} kb/s"
                f"  audio={abr or '?'} kb/s[/dim]"
            )

            video_high = h > MAX_HEIGHT or (
                vbr and vbr > TARGET_VIDEO_BITRATE * VIDEO_BITRATE_TOLERANCE
            )
            audio_high = abr and abr > TARGET_AUDIO_BITRATE * AUDIO_BITRATE_TOLERANCE

            if dry_run:
                if video_high:
                    action = f"transcode 480p @ {TARGET_VIDEO_BITRATE}k CBR + audio {TARGET_AUDIO_BITRATE}k"
                elif audio_high:
                    action = f"copy video + transcode audio to {TARGET_AUDIO_BITRATE}k"
                else:
                    action = "copy (already compliant)"
                console.print(f"[dim]  → would: {action}[/dim]")
                processed += 1
                progress.advance(overall)
                continue

            # --- Execute the appropriate operation ---
            if video_high:
                console.print(
                    f"[dim]  → transcode 480p @ {TARGET_VIDEO_BITRATE}k CBR, "
                    f"GOP={GOP_SIZE}s, audio {TARGET_AUDIO_BITRATE}k[/dim]"
                )
                ok = transcode_to_480p(
                    src_file, dst_file,
                    video_bitrate=f"{TARGET_VIDEO_BITRATE}k",
                    audio_bitrate=f"{TARGET_AUDIO_BITRATE}k",
                    console=console, progress=progress, task_id=overall,
                )
            elif audio_high:
                console.print(
                    f"[dim]  → copy video, re-encode audio → {TARGET_AUDIO_BITRATE}k[/dim]"
                )
                ok = transcode_audio_only(
                    src_file, dst_file,
                    audio_bitrate=f"{TARGET_AUDIO_BITRATE}k",
                    console=console, progress=progress, task_id=overall,
                )
            else:
                console.print("[dim]  → copy with faststart (already compliant)[/dim]")
                ok = copy_file(src_file, dst_file, console)

            if ok:
                processed += 1
            else:
                errors += 1

            progress.advance(overall)

    console.print()
    console.print(Panel.fit(Text("Done", style="bold blue")))
    console.print(f"[green]Processed : {processed}[/green]")
    console.print(f"[yellow]Skipped   : {skipped}[/yellow]")
    if errors:
        console.print(f"[red]Errors    : {errors}[/red]")
    return errors == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcode downloaded IA videos to web-ready format (Step 3b)"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=f"Output directory (default: {default_web_dir()})",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process at most N files (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be done without writing any files",
    )
    args = parser.parse_args()

    console = Console()
    console.print()
    console.print(Panel.fit(Text(
        "🎬 ask_anything — IA Video Web Transcoder (Step 3b)",
        style="bold blue",
    )))
    console.print(Panel.fit(Text(
        f"Target: ≤480p, ≤{TARGET_VIDEO_BITRATE} kb/s CBR, GOP={GOP_SIZE}s, "
        f"AAC ≤{TARGET_AUDIO_BITRATE} kb/s, faststart MP4",
        style="dim",
    )))
    console.print()

    if not check_ffmpeg(console):
        return 1

    gpu = check_nvidia(console)
    console.print(
        f"[green]GPU acceleration enabled[/green]"
        if gpu else
        "[yellow]⚠  No GPU — using CPU (libx264)[/yellow]"
    )
    console.print()

    web_dir = args.out_dir or default_web_dir()
    success = process_videos(
        src_dir=DOWNLOAD_DIR,
        web_dir=web_dir,
        console=console,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
