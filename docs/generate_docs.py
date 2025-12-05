#!/usr/bin/env python3
"""
Automatic documentation generator for the FAIRXAI framework.
Generates Sphinx .rst sources, UML/Dependency diagrams, and updates index.rst.
Compatible with Windows, Linux, and macOS.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Configuration
PROJECT_NAME = "FAIRXAI"
PROJECT_ROOT = Path(__file__).resolve().parent.parent / "fairxai"
DOCS_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = DOCS_ROOT / "source"
BUILD_DIR = DOCS_ROOT / "build"
API_REF_FILE = SOURCE_DIR / "api_reference.rst"
INDEX_FILE = SOURCE_DIR / "index.rst"


# Utility functions
def run_command(cmd, cwd=None):
    """Run a shell command and raise an error if it fails."""
    print(f"\n> {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(str(c) for c in cmd)}")

    print(result.stdout)
    return result


def safe_remove(path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def find_executable(name):
    """Return the path of an executable if available in venv/system PATH."""
    suffix = ".exe" if os.name == "nt" else ""
    for p in os.environ["PATH"].split(os.pathsep):
        exe_path = Path(p) / f"{name}{suffix}"
        if exe_path.exists():
            return exe_path
    return None


# Step 1: Generate .rst via sphinx-apidoc
def generate_apidoc():
    print("\n[1] Generating .rst files via sphinx-apidoc...")
    safe_remove(SOURCE_DIR / "fairxai")
    SOURCE_DIR.mkdir(exist_ok=True, parents=True)

    cmd = [
        sys.executable,
        "-m", "sphinx.ext.apidoc",
        "-o", str(SOURCE_DIR),
        str(PROJECT_ROOT)
    ]
    if not run_command(cmd):
        raise RuntimeError("Failed to run sphinx-apidoc.")


# Step 2: Generate diagrams (optional)
def generate_diagrams():
    print("\n[2] Generating dependency and UML diagrams (optional)...")

    pydeps_exec = find_executable("pydeps")
    pyreverse_exec = find_executable("pyreverse")

    diagrams_dir = SOURCE_DIR / "_static" / "diagrams"
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    # ---- PYDEPS ----
    if pydeps_exec:
        print("  -> Generating dependency graph with pydeps...")
        output_svg = diagrams_dir / "dependencies.svg"
        cmd = [
            "pydeps",
            str(PROJECT_ROOT),
            "--max-bacon", "3",
            "--show-deps",
            "--noshow",
            f"--output={output_svg}",  # compatibile anche con Windows
            "--format=svg"
        ]
        run_command(cmd)
    else:
        print("  pydeps not found — skipping dependency diagram")

    # ---- PYREVERSE ----
    if pyreverse_exec:
        print("  -> Generating UML with pyreverse...")
        with tempfile.TemporaryDirectory() as tmp:
            run_command([
                "pyreverse",
                "-f", "ALL",
                "-A", "-S",
                "-o", "dot",
                "-p", PROJECT_NAME,
                str(PROJECT_ROOT)
            ])
            for dot in Path(".").glob("classes_*.dot"):
                shutil.move(dot, diagrams_dir / dot.name)
    else:
        print("  pyreverse not found — skipping UML diagram")


# Step 3: Create single-page API reference
def generate_api_reference():
    print("\n[3] Creating API reference page...")

    content = f"""
{PROJECT_NAME} API Reference
{'=' * (len(PROJECT_NAME) + 15)}

.. toctree::
   :maxdepth: 4
   :glob:

   fairxai*
    """.strip()

    API_REF_FILE.write_text(content, encoding="utf-8")
    print(f"Created: {API_REF_FILE}")


def ensure_additional_docs():
    """Ensure 'usage.rst' and 'workflow.rst' exist to avoid Sphinx errors."""
    print("\n[4] Ensuring additional .rst files are present...")

    additional_files = {
        "usage.rst": """\
Usage
=====

This section describes how to use the FAIRXAI framework.

.. note::
   Add examples of project creation, model registration, and explanation pipeline execution here.
""",
        "workflow.rst": """\
Workflow
========

This section illustrates the typical workflow for using FAIRXAI.

.. note::
   Describe the main steps:
   1. Create a Project
   2. Register or load a model
   3. Run explainability pipelines
   4. Visualize results
"""
    }

    for filename, content in additional_files.items():
        file_path = SOURCE_DIR / filename
        if not file_path.exists():
            print(f"  -> Creating missing file: {filename}")
            file_path.write_text(content, encoding="utf-8")
        else:
            print(f"  -> {filename} already exists, skipping.")


# Step 4: Update index.rst
def update_index_rst():
    print("\n[5] Updating index.rst ...")

    index_content = f"""
.. {PROJECT_NAME} documentation master file

Welcome to {PROJECT_NAME}'s documentation
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   usage
   workflow
   api_reference
   modules
   indices

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
    """.strip()

    INDEX_FILE.write_text(index_content, encoding="utf-8")
    print(f"Updated: {INDEX_FILE}")


def build_html_docs():
    """
    Build Sphinx HTML docs.

    Strategy:
    - If Makefile / make.bat present in DOCS_ROOT, use them.
    - Otherwise fall back to calling sphinx-build via Python module (cross-platform).
    """
    # Use DOCS_ROOT and BUILD_DIR defined at top of your script
    docs_root = DOCS_ROOT  # already defined in your script
    source_dir = SOURCE_DIR
    build_html_dir = BUILD_DIR / "html"

    # 1) Try platform-native make
    if os.name == "nt":
        makefile = docs_root / "make.bat"
        if makefile.exists():
            try:
                run_command(["cmd", "/c", "make.bat", "html"], cwd=docs_root)
                print(f"HTML built at: {build_html_dir}")
                return
            except Exception as e:
                print(f"make.bat failed: {e}")
        else:
            print("make.bat not found in docs root, falling back to sphinx-build.")
    else:
        makefile = docs_root / "Makefile"
        if makefile.exists():
            try:
                run_command(["make", "html"], cwd=docs_root)
                print(f"HTML built at: {build_html_dir}")
                return
            except Exception as e:
                print(f"make html failed: {e}")
        else:
            print("Makefile not found in docs root, falling back to sphinx-build.")

    # 2) Fallback: run sphinx-build via Python module (no make needed)
    try:
        # Ensure output directory exists
        build_html_dir.mkdir(parents=True, exist_ok=True)
        # Use sphinx-build module (works in venv)
        cmd = [sys.executable, "-m", "sphinx", "-b", "html", str(source_dir), str(build_html_dir)]
        run_command(cmd, cwd=docs_root)
        print(f"HTML built at: {build_html_dir}")
    except Exception as e:
        print(f"sphinx-build fallback failed: {e}")
        raise e


# Main
def main():
    print(f"Starting documentation build for {PROJECT_NAME}...\n")

    generate_apidoc()
    # generate_diagrams()
    generate_api_reference()
    ensure_additional_docs()
    update_index_rst()
    build_html_docs()

    print("\nDocumentation generation completed.")


if __name__ == "__main__":
    main()
