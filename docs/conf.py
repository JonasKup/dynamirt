# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os, sys
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dynamirt'
copyright = '2026, Jonas Kupschus'
author = 'Jonas Kupschus'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_nb"]

# Tutorial notebooks are rendered with their saved outputs and never executed
# during the build. Re-run them manually in Jupyter to update the docs.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "deflist"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

napoleon_use_ivar = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_title = 'dynamirt'
html_theme_options = {
    "repository_url": "https://gitlab.git.nrw/jkupschus/dynamirt",
    "repository_provider": "gitlab",
    "use_repository_button": True,
}
