# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dynamirt'
copyright = '2026, Jonas Kupschus'
author = 'Jonas Kupschus'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "myst_nb"]

# Tutorial notebooks are rendered with their saved outputs and never executed
# during the build. Re-run them manually in Jupyter to update the docs.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "deflist"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

napoleon_use_ivar = True

# Topic pages use explicit headings for navigation. Dataclass fields remain
# in their docstrings instead of becoming individual navigation entries.
toc_object_entries = False
viewcode_follow_imported_members = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = 'dynamirt'
html_theme_options = {
    "repository_url": "https://github.com/JonasKup/dynamirt",
    "repository_provider": "github",
    "use_repository_button": True,
}
