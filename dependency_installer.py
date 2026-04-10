import sys
import subprocess
import importlib

def ensure_installed(pip_name, import_name=None):
    """Checks if a module is installed, and if not, pip installs it."""
    if import_name is None:
        import_name = pip_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"📦 Missing '{pip_name}'. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        print(f"✅ Successfully installed '{pip_name}'!")

# --- Auto-Install Standard Requirements ---
requirements = {
    'pandas': 'pandas',
    'scikit-learn': 'sklearn',
    'plotly': 'plotly',
    'kaleido': 'kaleido',
    'Pillow': 'PIL'
}

print("Checking dependencies...")
for pip_name, import_name in requirements.items():
    ensure_installed(pip_name, import_name)