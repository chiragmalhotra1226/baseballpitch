"""
Permanent fix for mediapipe PermissionError on Streamlit Cloud.

Strategy: Replace mediapipe's download_utils module entirely in sys.modules
BEFORE any mediapipe.solutions code imports it. This means our fake
download_utils is what mediapipe.python.solutions.pose sees when it does:
    import mediapipe.python.solutions.download_utils as download_utils

Our version copies the tflite files from our repo into /tmp and then
makes mediapipe believe the files are already in the venv by pointing
it to /tmp via os.path patching.
"""
import os
import sys
import shutil
import types

# ── Step 1: Download tflite files to /tmp immediately ─────────────────────────
_REPO_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "mediapipe_models"
)
_TMP_MP_DIR = "/tmp/mediapipe_modules/pose_landmark"
os.makedirs(_TMP_MP_DIR, exist_ok=True)

_MODELS_NEEDED = [
    "pose_landmark_lite.tflite",
    "pose_landmark_full.tflite",
]

for _model_name in _MODELS_NEEDED:
    _src = os.path.join(_REPO_MODELS_DIR, _model_name)
    _dst = os.path.join(_TMP_MP_DIR, _model_name)
    if os.path.exists(_src) and not os.path.exists(_dst):
        shutil.copy2(_src, _dst)
        print(f"✅ Copied {_model_name} to /tmp")
    elif os.path.exists(_dst):
        print(f"✅ {_model_name} already in /tmp")
    else:
        print(f"⚠️ WARNING: {_model_name} not found in repo at {_src}")

# ── Step 2: Import mediapipe so we can find its actual package path ────────────
import mediapipe as mp

_MP_PACKAGE_DIR = os.path.dirname(os.path.abspath(mp.__file__))
_VENV_MODEL_DIR = os.path.join(
    _MP_PACKAGE_DIR, "modules", "pose_landmark"
)

# ── Step 3: Try to place files into mediapipe's expected venv location ─────────
# On Streamlit Cloud, the venv dir is read-only for file creation
# BUT the modules/pose_landmark directory itself may already exist
# (mediapipe creates it during install). If so, we can write into it.
os.makedirs(_VENV_MODEL_DIR, exist_ok=True)

for _model_name in _MODELS_NEEDED:
    _src = os.path.join(_REPO_MODELS_DIR, _model_name)
    _dst = os.path.join(_VENV_MODEL_DIR, _model_name)
    if not os.path.exists(_dst) and os.path.exists(_src):
        try:
            shutil.copy2(_src, _dst)
            print(f"✅ Placed {_model_name} directly in mediapipe venv dir")
        except PermissionError:
            print(f"⚠️ Cannot write to venv dir — will patch download_utils")

# ── Step 4: Patch download_utils regardless, as a safety net ──────────────────
# This handles cases where venv write succeeded AND cases where it didn't.
# We replace the download function so it NEVER tries to write to venv.
import mediapipe.python.solutions.download_utils as _du

def _safe_download(model_path: str) -> None:
    """
    Replacement for mediapipe's download_oss_model.
    Checks /tmp first, then repo, never tries to write to venv.
    """
    model_name = model_path.split("/")[-1]
    
    # Check if file already exists in venv (put there in Step 3)
    venv_path = os.path.join(_MP_PACKAGE_DIR, model_path)
    if os.path.exists(venv_path):
        print(f"✅ {model_name} found in venv path, skipping download")
        return
    
    # Check /tmp
    tmp_path = os.path.join(_TMP_MP_DIR, model_name)
    if os.path.exists(tmp_path):
        # Try to copy from /tmp to venv
        try:
            os.makedirs(os.path.dirname(venv_path), exist_ok=True)
            shutil.copy2(tmp_path, venv_path)
            print(f"✅ Moved {model_name} from /tmp to venv")
            return
        except PermissionError:
            # Venv truly read-only — monkey-patch the abspath function
            print(f"⚠️ Venv read-only, patching mp path resolution for {model_name}")
            _patch_resource_path(model_path, tmp_path)
            return
    
    print(f"❌ {model_name} not found anywhere — original download will fail")

def _patch_resource_path(model_path: str, actual_path: str) -> None:
    """
    Last resort: patch mediapipe's internal resource path lookup
    so it finds the file in /tmp.
    """
    try:
        import mediapipe.python.solutions.pose as _pose_module
        original_init = _pose_module.Pose.__init__

        def patched_init(self, *args, **kwargs):
            # Before calling original __init__, ensure file exists
            # by temporarily monkeypatching os.path.abspath for mp paths
            original_abspath = os.path.abspath

            def patched_abspath(path):
                result = original_abspath(path)
                if "pose_landmark" in result and not os.path.exists(result):
                    tmp = os.path.join(_TMP_MP_DIR, os.path.basename(result))
                    if os.path.exists(tmp):
                        return tmp
                return result

            os.path.abspath = patched_abspath
            try:
                original_init(self, *args, **kwargs)
            finally:
                os.path.abspath = original_abspath

        _pose_module.Pose.__init__ = patched_init
        print("✅ Patched Pose.__init__ to use /tmp models")
    except Exception as e:
        print(f"⚠️ Path patch failed: {e}")

# Replace the download function
_du.download_oss_model = _safe_download
print("✅ setup_mediapipe complete — download_utils patched")
