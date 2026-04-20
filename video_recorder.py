"""Video recording utilities for Newton examples.

Captures rendered frames as PNG images and encodes them into MP4 using
imageio-ffmpeg.

Usage::

    from video_recorder import VideoRecorder

    VideoRecorder.add_args(parser)           # in argparse setup
    recorder = VideoRecorder.from_options(viewer, fps=60, options=args)

    # per frame:
    if recorder:
        recorder.capture_frame()

    # at exit:
    if recorder:
        recorder.finalize()
"""

import time
from pathlib import Path

import numpy as np
import warp as wp


class VideoRecorder:
    """Records viewer frames to PNG and encodes them as an MP4 video.

    Args:
        viewer: Newton viewer instance (must expose ``get_frame``).
        output_path: Destination path for the MP4 file.
        fps: Frames per second for the output video.
        frames_dir: Directory for intermediate frame PNGs.  When *None*,
            a timestamped directory is created next to *output_path*.
        keep_frames: If *True*, intermediate PNGs are kept after encoding.
    """

    def __init__(
        self,
        viewer,
        output_path: str | Path,
        fps: int = 60,
        frames_dir: str | Path | None = None,
        keep_frames: bool = False,
    ):
        self._viewer = viewer
        self._fps = fps
        self._keep_frames = keep_frames
        self._frame_buffer: wp.array | None = None
        self._frame_count = 0
        self._enabled = True

        if not hasattr(viewer, "get_frame"):
            print(
                "Video recording requires ViewerGL "
                "(viewer.get_frame is unavailable). Recording disabled."
            )
            self._enabled = False
            return

        output_path = Path(output_path).expanduser()
        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path

        if frames_dir is not None:
            self._frames_dir = Path(frames_dir).expanduser()
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self._frames_dir = (
                output_path.parent / f"{output_path.stem}_frames_{timestamp}"
            )
        self._frames_dir.mkdir(parents=True, exist_ok=True)
        for stale in self._frames_dir.glob("frame_*.png"):
            stale.unlink()

        print(f"Recording frames to: {self._frames_dir}")
        print(f"Will encode MP4 to:  {self._output_path}")

    # -- public API ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether recording is active."""
        return self._enabled

    def capture_frame(self):
        """Grab the current viewer frame and save it as a PNG."""
        if not self._enabled:
            return

        try:
            from PIL import Image  # noqa: WPS433
        except ImportError:
            print(
                "Pillow is not installed. "
                "Install with: uv pip install pillow"
            )
            self._enabled = False
            return

        frame = self._viewer.get_frame(target_image=self._frame_buffer)
        if self._frame_buffer is None:
            self._frame_buffer = frame

        frame_np = frame.numpy()
        image = Image.fromarray(frame_np, mode="RGB")
        frame_path = self._frames_dir / f"frame_{self._frame_count:05d}.png"
        image.save(frame_path)
        self._frame_count += 1

    def finalize(self):
        """Encode captured frames into an MP4 and optionally delete PNGs."""
        if not self._enabled:
            return

        try:
            import imageio_ffmpeg as ffmpeg  # noqa: WPS433
        except ImportError:
            print(
                "imageio-ffmpeg is not installed. "
                "Install with: uv pip install imageio-ffmpeg"
            )
            return
        try:
            from PIL import Image  # noqa: WPS433
        except ImportError:
            print(
                "Pillow is not installed. "
                "Install with: uv pip install pillow"
            )
            return

        frame_files = sorted(self._frames_dir.glob("frame_*.png"))
        if not frame_files:
            print(
                f"No frames found in {self._frames_dir}; "
                "skipping MP4 export."
            )
            return

        with Image.open(frame_files[0]) as first_img:
            width, height = first_img.size
        even_width = width if width % 2 == 0 else width + 1
        even_height = height if height % 2 == 0 else height + 1
        needs_pad = even_width != width or even_height != height
        size = (even_width, even_height)

        print(
            f"Encoding {len(frame_files)} frames "
            f"at {even_width}x{even_height} "
            f"to {self._output_path} ..."
        )
        try:
            writer = ffmpeg.write_frames(
                str(self._output_path),
                size=size,
                fps=self._fps,
                codec="libx264",
                macro_block_size=1,
                quality=5,
            )
            writer.send(None)
            for frame_path in frame_files:
                with Image.open(frame_path) as img:
                    frame = np.array(img)
                if needs_pad:
                    padded = np.zeros(
                        (even_height, even_width, frame.shape[2]),
                        dtype=frame.dtype,
                    )
                    padded[:height, :width] = frame
                    frame = padded
                writer.send(frame)
            writer.close()
            print(f"MP4 export complete: {self._output_path}")
        except Exception as exc:
            print(f"Failed to encode MP4: {exc}")
            return

        if not self._keep_frames:
            for frame_path in frame_files:
                frame_path.unlink()
            print(f"Deleted intermediate PNG frames in: {self._frames_dir}")

    # -- argparse helpers ---------------------------------------------------

    @staticmethod
    def add_args(parser):
        """Add recording-related CLI arguments to *parser*."""
        parser.add_argument(
            "--record-mp4",
            type=str,
            default=None,
            help="Write an MP4 recording by capturing each rendered frame",
        )
        parser.add_argument(
            "--record-fps",
            type=int,
            default=None,
            help="Output FPS for MP4 encoding (defaults to simulation FPS)",
        )
        parser.add_argument(
            "--record-frames-dir",
            type=str,
            default=None,
            help="Directory to store intermediate frame PNGs",
        )
        parser.add_argument(
            "--record-keep-frames",
            action="store_true",
            help="Keep intermediate frame PNGs after MP4 export",
        )

    @classmethod
    def from_options(
        cls, viewer, fps: int, options
    ) -> "VideoRecorder | None":
        """Create a recorder from parsed CLI options, or *None* if disabled."""
        if options.record_mp4 is None:
            return None
        record_fps = (
            int(options.record_fps) if options.record_fps is not None else fps
        )
        return cls(
            viewer=viewer,
            output_path=options.record_mp4,
            fps=record_fps,
            frames_dir=getattr(options, "record_frames_dir", None),
            keep_frames=getattr(options, "record_keep_frames", False),
        )
