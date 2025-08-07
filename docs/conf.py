# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'fairxai')))
project = 'FairXAI'
copyright = '2025, Kode'
author = 'Kode'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.coverage",
              'sphinx.ext.viewcode',
              "sphinx.ext.napoleon",'sphinx.ext.duration',
              'sphinx.ext.autosummary'
              ]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
# The master toctree document.
master_doc = 'index'
# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

autodoc_default_options = {"members": True, "inherited-members": True}
autosummary_generate = True
autoclass_content='class'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "style.css" will overwrite the builtin "style.css".
html_static_path = ['_static']

html_theme_options = {
    'display_version': True,
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4
}