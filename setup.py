"""
Setup configuration with dynamic project naming.
Enables pip installation (`pip install .` or `pip install -e .`) where the main package name
can be changed anytime in pyproject.toml, project_name.txt, or via the PROJECT_NAME env variable.
"""

import os
from pathlib import Path
import re
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent


def get_project_name() -> str:
    """Resolve active project name from env var, project_name.txt, or pyproject.toml."""
    # 1. Environment variable override (e.g. PROJECT_NAME=voxguard pip install .)
    if os.getenv("PROJECT_NAME"):
        return os.getenv("PROJECT_NAME").strip().replace("-", "_").lower()

    # 2. Check project_name.txt if present
    name_file = ROOT / "project_name.txt"
    if name_file.exists():
        try:
            name = name_file.read_text(encoding="utf-8").strip()
            if name:
                return name.replace("-", "_").lower()
        except Exception:
            pass

    # 3. Check pyproject.toml [project] name = "..."
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        try:
            match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', pyproject.read_text(encoding="utf-8"))
            if match:
                return match.group(1).replace("-", "_").lower()
        except Exception:
            pass

    # 4. Default project name
    return "voice_clone_detection"


project_name = get_project_name()
dist_name = project_name.replace("_", "-")

# Map subpackages inside src/ to both the active project_name and src
subpackages = find_packages(where="src")
packages = [project_name, "src"] + [f"{project_name}.{p}" for p in subpackages] + [f"src.{p}" for p in subpackages]

package_dir = {
    project_name: "src",
    "src": "src",
}
for p in subpackages:
    rel_path = f"src/{p.replace('.', '/')}"
    package_dir[f"{project_name}.{p}"] = rel_path
    package_dir[f"src.{p}"] = rel_path

setup(
    name=dist_name,
    version="1.0.0",
    description="Real-Time AI Voice Cloning & Deepfake Detection Framework",
    packages=packages,
    package_dir=package_dir,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.0",
        "scipy>=1.12.0",
        "soundfile>=0.12.1",
        "pydantic>=2.5.0",
        "torch>=2.2.0",
        "torchaudio>=2.2.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.28.0",
        "python-multipart>=0.0.9",
        "websockets>=12.0",
        "httpx>=0.27.0",
    ],
    entry_points={
        "console_scripts": list({
            f"{project_name} = src.api.server:start_server",
            f"{dist_name} = src.api.server:start_server",
        })
    },
)
