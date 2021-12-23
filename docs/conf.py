# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# sisl documentation build configuration file, created by
# sphinx-quickstart on Wed Dec  2 19:55:34 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sys
import os
import pathlib
from datetime import date

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# make sure the source version is preferred (#3567)
_root = pathlib.Path(__file__).absolute().parent.parent

# If building this on RTD, mock out fortran sources
on_rtd = os.environ.get('READTHEDOCS', 'false').lower() == 'true'
if on_rtd:
    os.environ["SISL_NUM_PROCS"] = "1"
    os.environ["SISL_VIZ_NUM_PROCS"] = "1"

try:
    import sisl
    print(f"Located sisl here: {sisl.__path__}")
except:
    _pp = os.environ.get("PYTHONPATH", "")
    if len(_pp) > 0:
        os.environ["PYTHONPATH"] = f"{_root}:{_pp}"
    else:
        os.environ["PYTHONPATH"] = f"{_root}"
    del _pp
sys.path.insert(0, str(_root))

# Print standard information about executable and path...
print("python exec:", sys.executable)
print("sys.path:", sys.path)

import sisl


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    # plotting and advanced usage
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx.ext.inheritance_diagram',
    'nbsphinx',
    'sphinx_gallery.load_style',
]
napoleon_numpy_docstring = True

# There currently is a bug with mathjax >= 3, so we resort to 2.7.7
mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_HTMLorMML"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# Short-hand for :doi:
extlinks = {
    'issue': ('https://github.com/zerothi/sisl/issues/%s', 'issue %s'),
    'pull': ('https://github.com/zerothi/sisl/pull/%s', 'pull request %s'),
    'doi': ('https://dx.doi.org/%s', '%s'),
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The encoding of source files.
#source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = 'index'

# prepend/append this snippet in _all_ sources
rst_prolog = """
.. highlight:: python
"""
# Insert the links into the epilog (globally)
# This means that every document has access to the links
rst_epilog = ''.join(open('epilog.dummy').readlines())

autosummary_generate = True

# General information about the project.
project = 'sisl'
author = 'Nick Papior'
copyright = f"2015-{date.today().year}, {author}"

# If building this on RTD, mock out fortran sources
if on_rtd:
    nbsphinx_allow_errors = True
else:
    nbsphinx_allow_errors = False

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
version = sisl.__version__
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# Add __init__ classes to the documentation
autoclass_content = 'class'
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'special-members': '__init__,__call__',
    'inherited-members': True,
    'show-inheritance': True,
}

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
#today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['build', '**/setupegg.py', '**/setup.rst', '**/tests', '**.ipynb_checkpoints']
exclude_patterns.append("**/GUI with Python Demo.ipynb")
exclude_patterns.append("**/Building a plot class.ipynb")

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = 'autolink'

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['sisl.']

# If true, keep warnings as "system message" paragraphs in the built documents.
#keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = f"sisl {release} documentation"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "sisl"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
#html_logo = None

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
#html_favicon = None

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if os.path.exists('_static'):
    html_static_path = ['_static']
else:
    html_static_path = []


# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
#html_extra_path = []

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
#html_last_updated_fmt = '%b %d, %Y'

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_domain_indices = True

# If false, no index is generated.
html_use_modindex = True
html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, links to the reST sources are added to the pages.
#html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
#html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
#html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
#html_file_suffix = None

# Language to be used for generating the HTML full-text search index.
# Sphinx supports the following languages:
#   'da', 'de', 'en', 'es', 'fi', 'fr', 'hu', 'it', 'ja'
#   'nl', 'no', 'pt', 'ro', 'ru', 'sv', 'tr'
#html_search_language = 'en'

# A dictionary with options for the search language support, empty by default.
# Now only 'ja' uses this config value
#html_search_options = {'type': 'default'}

# The name of a javascript file (relative to the configuration directory) that
# implements a search results scorer. If empty, the default will be used.
#html_search_scorer = 'scorer.js'

# Output file base name for HTML help builder.
htmlhelp_basename = 'sisl'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
# The paper size ('letterpaper' or 'a4paper').
'papersize': 'a4paper',

# The font size ('10pt', '11pt' or '12pt').
'pointsize': '10pt',

# Additional stuff for the LaTeX preamble.
#'preamble': '',

# Latex figure (float) alignment
'figure_align': '!htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'sisl.tex', 'sisl Documentation',
   'Nick Papior', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'sisl', 'sisl Documentation',
     [author], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'sisl', 'sisl Documentation',
   author, None, 'One line description of project.',
   'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
#texinfo_no_detailmenu = False


#####
# Custom sisl documentation stuff down here
#####

# These two options should solve the "toctree contains reference to nonexisting document"
# problem.
# See here: numpydoc #69
#class_members_toctree = False
# If this is false we do not have double method sections
#numpydoc_show_class_members = False

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
# Python, numpy, scipy and matplotlib specify https as the default objects.inv
# directory. So please retain these links.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
    'skimage': ('https://scikit-image.org/docs/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
}

# Tell nbsphinx to wait, at least X seconds for each cell
nbsphinx_timeout = 600

# Insert a link to download the IPython notebook
nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base="docs") %}

.. raw:: html

     <div align="right">
     Download IPython notebook <a href="https://raw.githubusercontent.com/zerothi/sisl/master/{{ docname }}"> here</a>.
     <span style="white-space: nowrap;"><a href="https://mybinder.org/v2/gh/zerothi/sisl/master?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.</span>
     </div>

"""

nbsphinx_thumbnails = {}

import inspect


def sisl_method2class(meth):
    # Method to retrieve the class from a method (bounded and unbounded)
    # See stackoverflow.com/questions/3589311
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return None  # not required since None would have been implicitly returned anyway

# My custom detailed instructions for not documenting stuff


def sisl_skip(app, what, name, obj, skip, options):
    global autodoc_default_options
    # When adding routines here, please also add them
    # to the _templates/autosummary/class.rst file to limit
    # the documentation.
    if what == 'class':
        if name in ['ArgumentParser', 'ArgumentParser_out',
                    'is_keys', 'key2case', 'keys2case',
                    'line_has_key', 'line_has_keys', 'readline',
                    'step_to',
                    'isDataset', 'isDimension', 'isGroup',
                    'isRoot', 'isVariable']:
            return True
    #elif what == "attribute":
    #    return True

    # check for special methods (we don't want all)
    if (name.startswith("_") and
        name not in autodoc_default_options.get("special-members", '').split(',')):
        return True

    try:
        cls = sisl_method2class(obj)
    except:
        cls = None

    # Quick escape
    if cls is None:
        return skip

    # Currently inherited members will never be processed
    # Apparently they will be linked directly.
    # Now we have some things to disable the output of
    if "projncSile" in cls.__name__:
        if name in ["current", "current_parameter", "shot_noise",
                    "noise_power", "fano", "density_matrix",
                    "write_tbtav",
                    "orbital_COOP", "atom_COOP",
                    "orbital_COHP", "atom_COHP"]:
            return True
    if "SilePHtrans" in cls.__name__:
        if name in ["chemical_potential", "electron_temperature",
                    "kT", "current", "current_parameter", "shot_noise",
                    "noise_power"]:
            return True
    return skip


def setup(app):
    # Setup autodoc skipping
    app.connect('autodoc-skip-member', sisl_skip)

    import subprocess as sp
    if os.path.isfile('../conf_prepare.sh'):
        print("# Running ../conf_prepare.sh")
        sp.call(['bash', '../conf_prepare.sh'])
        print("\n# Done running ../conf_prepare.sh")
    elif os.path.isfile('conf_prepare.sh'):
        print("# Running conf_prepare.sh")
        sp.call(['bash', 'conf_prepare.sh'])
        print("\n# Done running conf_prepare.sh")
    print("")
