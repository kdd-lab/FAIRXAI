"""
Automatic documentation generator for a Python project (cross-platform)
Generates:
 - .rst API files (Sphinx autodoc)
 - UML class diagrams (Pyreverse)
 - Dependency graph (pydeps)
 - Usage overview
 - Workflow diagram (Mermaid)
 - HTML docs build

Author: Kode (FairXAI)
"""

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

# === CONFIGURATION ===
PROJECT_NAME = "FairXAI"
SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent / "fairxai"  # FAIRXAI/fairxai
DOCS_SOURCE = SOURCE_DIR / "source"
DOCS_BUILD = SOURCE_DIR / "build" / "html"
STATIC_DIR = DOCS_SOURCE / "_static"

os.makedirs(STATIC_DIR, exist_ok=True)


# === UTILITIES ===

def run_command(cmd: list[str], cwd=None):
    """Run a subprocess cross-platform and stream stdout."""
    print(f"\n> {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"⚠️ Command failed: {' '.join(cmd)}")


def check_graphviz():
    """Check if Graphviz is installed and accessible."""
    print("\n[0] Checking Graphviz...")
    result = shutil.which("dot")
    if not result:
        print("""
⚠️ Graphviz not found!
Please install it to generate UML and dependency diagrams:

Windows  → https://graphviz.org/download/
macOS    → brew install graphviz
Linux    → sudo apt install graphviz

Then restart your terminal.
""")
    else:
        print(f"✅ Graphviz found at: {result}")


# === STEP 1: sphinx-apidoc ===
def generate_sphinx_apidoc():
    print("\n[1] Generating .rst files with sphinx-apidoc...")
    run_command([
        sys.executable, "-m", "sphinx.ext.apidoc",
        "-f", "-o", str(DOCS_SOURCE), str(PROJECT_ROOT)
    ])


# === STEP 2: UML diagrams (Pyreverse) ===

def generate_pyreverse():
    print("\n[2] Generating UML class diagrams with Pyreverse...")
    uml_dir = STATIC_DIR / "uml"
    os.makedirs(uml_dir, exist_ok=True)

    # Directory da escludere durante la scansione
    EXCLUDED_DIRS = {
        'venv', '.venv', 'env', '.env',  # Virtual environments
        'docs', 'doc', '_build', 'build',  # Documentation
        '__pycache__', '.pytest_cache', '.tox',  # Cache
        '.git', '.svn', '.hg',  # Version control
        'node_modules', 'dist', 'egg-info',  # Build artifacts
        'tests', 'test'  # Test directories (opzionale)
    }

    # Create one diagram per top-level package to keep it readable
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filtra le directory da escludere in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith('.')]

        if "__init__.py" in files:
            # Verifica che siamo effettivamente in un package del progetto principale
            rel_path = Path(root).relative_to(PROJECT_ROOT)

            # Salta se siamo in una sottodirectory esclusa
            if any(part in EXCLUDED_DIRS for part in rel_path.parts):
                continue

            package_name = Path(root).name
            package_uml_dir = uml_dir / package_name
            os.makedirs(package_uml_dir, exist_ok=True)

            # Cambia la directory corrente per l'output
            original_dir = os.getcwd()

            try:
                # Sposta nella directory di output prima di eseguire pyreverse
                os.chdir(str(package_uml_dir))

                # Usa il formato DOT invece di PNG (più affidabile e nativo)
                # oppure usa 'dot' che è il formato Graphviz standard
                run_command([
                    "pyreverse",
                    "-o", "dot",  # Usa formato DOT nativo invece di PNG
                    "-p", package_name,
                    "-A",  # Mostra tutti i membri
                    "-S",  # Mostra anche i metodi statici
                    root  # Path del package da analizzare
                ])

                print(f"✅ Generated UML for {package_name}")

            except Exception as e:
                print(f"⚠️  Warning: Could not generate diagram for {package_name}: {e}")
            finally:
                # Ripristina sempre la directory originale
                os.chdir(original_dir)

    print(f"✅ UML diagrams saved to {uml_dir}")


# === STEP 3: Dependency graph (pydeps) ===
def generate_pydeps():
    print("\n[3] Generating dependency graph with pydeps...")
    dep_path = STATIC_DIR / "dependencies.svg"
    run_command([
        "pydeps", str(PROJECT_ROOT),
        "--max-bacon", "3", "--show-deps",
        "--noshow", "--output", str(dep_path)
    ])
    print(f"✅ Dependency graph saved to {dep_path}")


# === STEP 4: Generate usage.rst automatically ===
def extract_symbols(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    return classes, funcs


def generate_usage_rst():
    print("\n[4] Generating usage.rst automatically...")
    usage_path = DOCS_SOURCE / "usage.rst"

    with open(usage_path, "w", encoding="utf-8") as f:
        f.write("Usage and Components\n====================\n\n")
        f.write("Overview of classes and functions automatically extracted from source.\n\n")

        for root, _, files in os.walk(PROJECT_ROOT):
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    full = Path(root) / file
                    rel = full.relative_to(PROJECT_ROOT)
                    classes, funcs = extract_symbols(full)
                    if classes or funcs:
                        f.write(f"**{rel}**\n\n")
                        if classes:
                            f.write("  Classes:\n")
                            for c in classes:
                                f.write(f"   - {c}\n")
                        if funcs:
                            f.write("  Functions:\n")
                            for fn in funcs:
                                f.write(f"   - {fn}\n")
                        f.write("\n")

    print(f"✅ usage.rst created at {usage_path}")


# === STEP 5: Workflow diagram (Mermaid) ===
def generate_workflow_rst():
    print("\n[5] Generating workflow.rst (Mermaid)...")
    workflow_path = DOCS_SOURCE / "workflow.rst"
    content = """Workflow
========

High-level overview of the system workflow:

.. mermaid::

   graph TD
      A[Dataset Loader] --> B[Preprocessing]
      B --> C[Descriptor Builder]
      C --> D[Explainer Manager]
      D --> E[Evaluation & Reporting]
"""
    with open(workflow_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ workflow.rst created at {workflow_path}")


# === STEP 6: Build HTML ===
def build_html():
    print("\n[6] Building HTML documentation...")
    run_command([
        "sphinx-build", "-b", "html",
        str(DOCS_SOURCE), str(DOCS_BUILD)
    ])
    print(f"✅ HTML documentation built at: {DOCS_BUILD}")


# === MAIN ===
def main():
    print(f"=== Generating documentation for {PROJECT_NAME} ===")
    check_graphviz()
    generate_sphinx_apidoc()
    generate_pyreverse()
    generate_pydeps()
    generate_usage_rst()
    generate_workflow_rst()
    build_html()
    print("\n=== ✅ All done! ===")


if __name__ == "__main__":
    main()
