"""
Run once at startup to pre-download mediapipe tflite models
into /tmp (writable) and patch mediapipe's download_utils
so it uses /tmp instead of the read-only venv directory.
"""
import os
import shutil
import urllib.request
import mediapipe as mp

# The writable temp directory
TMP_MODEL_DIR = "/tmp/mediapipe_models/mediapipe/modules/pose_landmark"
os.makedirs(TMP_MODEL_DIR, exist_ok=True)

# The model mediapipe 0.10.21 needs (model_complexity=0 uses lite)
MODELS = [
    "pose_landmark_lite.tflite",
    "pose_landmark_full.tflite",   # model_complexity=1 (default)
]

GCS_URL = "https://storage.googleapis.com/mediapipe-assets/"

def download_models():
    for model_name in MODELS:
        dest = os.path.join(TMP_MODEL_DIR, model_name)
        if not os.path.exists(dest):
            url = GCS_URL + model_name
            print(f"Downloading {model_name} to {dest}...")
            try:
                with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
                    shutil.copyfileobj(r, f)
                print(f"✅ Downloaded {model_name}")
            except Exception as e:
                print(f"⚠️ Could not download {model_name}: {e}")
        else:
            print(f"✅ {model_name} already exists")

def patch_mediapipe():
    """
    Patch mediapipe's download_utils so it looks for models
    in /tmp/mediapipe_models instead of the read-only venv path.
    """
    import mediapipe.python.solutions.download_utils as du
    import mediapipe

    original_download = du.download_oss_model

    def patched_download(model_path: str):
        model_name = model_path.split("/")[-1]
        tmp_path = os.path.join(TMP_MODEL_DIR, model_name)

        # If model is in /tmp, copy it to a writable location mediapipe expects
        # OR monkey-patch the root path mediapipe uses
        mp_root = os.sep.join(os.path.abspath(mediapipe.__file__).split(os.sep)[:-1])
        dest_path = os.path.join(mp_root, model_path)

        # Try to make the destination writable
        dest_dir = os.path.dirname(dest_path)
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except Exception:
            pass

        if os.path.exists(dest_path):
            return  # Already there

        if os.path.exists(tmp_path):
            try:
                shutil.copy2(tmp_path, dest_path)
                print(f"✅ Copied {model_name} from /tmp to mediapipe package path")
                return
            except PermissionError:
                pass

        # Fall back to original (may fail on cloud)
        try:
            original_download(model_path)
        except Exception as e:
            print(f"⚠️ Original download failed: {e}")

    du.download_oss_model = patched_download
    print("✅ Mediapipe download_utils patched")

# Run both
download_models()
patch_mediapipe()
