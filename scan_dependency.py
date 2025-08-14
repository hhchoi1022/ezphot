# requirements.py
"""
Scan the tippy package for third-party imports
(excluding standard library, local package imports, and selected folders),
and write them with pinned versions to requirements.txt.

Run:
    python requirements.py
"""

from pathlib import Path
import ast
import pkg_resources
from stdlib_list import stdlib_list

# Python version for stdlib detection
PYTHON_VERSION = "3.11"

# Folders to skip in scanning
exclude_dirs = {"docs", "analysis", "__pycache__"}

# Local package name to exclude
local_package_name = "tippy"

# Where to scan
PACKAGE_DIR = Path(local_package_name)

stdlib_mods = set(stdlib_list(PYTHON_VERSION))
imports = set()

for path in PACKAGE_DIR.rglob("*.py"):
    # Skip files inside excluded directories
    if any(part in exclude_dirs for part in path.parts):
        continue

    with open(path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read(), filename=str(path))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])

# Filter out stdlib and local package imports
third_party = sorted(
    m for m in imports
    if m not in stdlib_mods and m != local_package_name
)

# Map import names to PyPI package names if needed
mapping = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
}
pypi_packages = [mapping.get(pkg, pkg) for pkg in third_party]

# Attach versions from current environment
installed_pkgs = {dist.project_name.lower(): dist.version for dist in pkg_resources.working_set}

requirements_with_versions = []
for pkg in pypi_packages:
    ver = installed_pkgs.get(pkg.lower())
    if ver:
        requirements_with_versions.append(f"{pkg}=={ver}")
    else:
        requirements_with_versions.append(pkg)  # fallback without version

# Write to requirements.txt
req_file = Path("requirements.txt")
req_file.write_text("\n".join(requirements_with_versions) + "\n", encoding="utf-8")

print(f"Detected {len(pypi_packages)} external packages.")
print(f"Saved to {req_file.resolve()}")

