from pathlib import Path
import re
from setuptools import setup, find_packages

ROOT = Path(__file__).parent

def read_readme():
    p = ROOT / "README.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""

def read_requirements():
    req = ROOT / "requirements.txt"
    if not req.exists():
        return []
    lines = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-r"):
            continue
        # avoid local paths and editable flags in install_requires
        if line.startswith((".", "/")) or line.startswith("-e"):
            continue
        lines.append(line)
    return lines

def get_version():
    init_py = (ROOT / "ezphot" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.M)
    return m.group(1) if m else "0.0.0"

setup(
    name="ezphot",
    version=get_version(),
    description="Easy and flexible photometry toolkit in Python, from preprocessing to analysis, for various telescopes.",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Hyeonho Choi",
    python_requires=">=3.9",
    packages=find_packages(exclude=("docs*", "tests*", "examples*", "scripts*")),
    include_package_data=True,  # honors MANIFEST.in for non-.py files
    install_requires=read_requirements(),  # or list them here explicitly
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # change if different
        "Operating System :: OS Independent",
    ],
    # If you have a CLI, uncomment and point to your entry function:
    # entry_points={"console_scripts": ["ezphot=ezphot.cli:main"]},
)

