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


# Step 2: Create single-page API reference
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


def build_html_docs():
    """Build Sphinx HTML documentation."""
    print("\n[4] Building HTML docs...")
    build_html_dir = BUILD_DIR / "html"
    build_html_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "sphinx", "-E", "-b", "html", str(SOURCE_DIR), str(build_html_dir)]
    run_command(cmd)
    print(f"HTML built at: {build_html_dir}")
    return build_html_dir


# Step 5: Deploy HTML to gh-pages
def deploy_to_gh_pages(html_dir):
    """Deploy the built HTML docs to the gh-pages branch."""
    print("\n[5] Deploying documentation to gh-pages...")

    # Get repo URL
    result = subprocess.run(["git", "config", "--get", "remote.origin.url"],
                            capture_output=True, text=True)
    repo_url = result.stdout.strip()
    if not repo_url:
        raise RuntimeError("Could not determine remote repository URL.")

    current_dir = Path.cwd()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print("Cloning gh-pages branch...")
        res = subprocess.run(
            ["git", "clone", "--branch", "gh-pages", "--single-branch", repo_url, tmpdir],
            capture_output=True, text=True
        )

        os.chdir(tmp_path)

        # Clean old files (keep .git)
        for item in tmp_path.iterdir():
            if item.name == ".git":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Copy new docs
        print("Copying new HTML files...")
        for item in html_dir.iterdir():
            dest = tmp_path / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Add .nojekyll
        (tmp_path / ".nojekyll").write_text("", encoding="utf-8")

        # Commit and push
        run_command(["git", "add", "."])
        run_command(['git', 'commit', '-m', 'Update documentation [auto]'])
        run_command(["git", "push", "origin", "gh-pages"])
        print("✅ Documentation successfully deployed to gh-pages!")

    os.chdir(current_dir)

# Main
def main():
    print(f"Starting documentation build for {PROJECT_NAME}...\n")

    generate_apidoc()
    generate_api_reference()
    html_dir = build_html_docs()
    deploy_to_gh_pages(html_dir)

    print("\n✨ Documentation generation and deployment completed successfully.")


if __name__ == "__main__":
    main()
