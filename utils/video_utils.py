import os
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip


def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def trim_video(input_path: str, output_path: str, start: float = 0, end: float = None):
    clip = VideoFileClip(input_path)
    if end is None:
        end = clip.duration
    clip.subclipped(start, end).write_videofile(output_path, codec="libx264", audio=False)