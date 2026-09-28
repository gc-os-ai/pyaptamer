# Configuration file for the Sphinx documentation builder.
#
# For a full list of options see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from importlib.metadata import version as get_version

# -- Project information -----------------------------------------------------

project = "pyaptamer"
copyright = "2026, German Center for Open Source AI"
author = "German Center for Open Source AI"

release = get_version("pyaptamer")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc / autosummary ---------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "none"
add_module_names = False

# numpydoc renders the docstrings; let autosummary build the member tables.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]

# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = "pyaptamer"
html_logo = "../assets/pyaptamer_banner.png"
html_favicon = "../assets/pyaptamer_icon.png"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gc-os-ai/pyaptamer",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pyaptamer/",
            "icon": "fa-solid fa-box",
        },
    ],
    "show_prev_next": False,
    "navigation_with_keys": False,
}

html_context = {
    "github_user": "gc-os-ai",
    "github_repo": "pyaptamer",
    "github_version": "main",
    "doc_path": "docs/source",
}


# -- Skip scikit-learn metadata-routing boilerplate -------


def _skip_member(app, what, name, obj, skip, options):
    if name == "get_metadata_routing" or name.endswith("_request"):
        return True
    return None


def setup(app):
    app.connect("autodoc-skip-member", _skip_member)
