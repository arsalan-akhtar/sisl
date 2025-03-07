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
"""sisl documentation"""
from __future__ import annotations

import inspect
import logging
import os
import pathlib
import sys
from datetime import date
from functools import wraps
from typing import Literal

_log = logging.getLogger("sisl_doc")

_doc_root = pathlib.Path(__file__).absolute().parent
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# make sure the source version is preferred (#3567)
_root = _doc_root.parent
_src = _root / "src"

# add the exts folder
sys.path.insert(1, str(_doc_root))

# Print standard information about executable and path...
print("python exec:", sys.executable)
print("sys.path:", sys.path)

import numpy as np

import sisl

print(f"sisl: {sisl.__version__}, {sisl.__file__}")
# Extract debug-information, for completeness sake.
sisl.debug_info()
import pybtex

# Figure out if we can locate the tests:
sisl_files_tests = sisl.get_environ_variable("SISL_FILES_TESTS")
print(f"SISL_FILES_TESTS: {sisl_files_tests}")
print("  is directory: ", sisl_files_tests.is_dir())
if sisl_files_tests.is_dir():
    print("  content:")
    for _child in sisl_files_tests.iterdir():
        print(f"    ./{_child.relative_to(sisl_files_tests)}")


# Setting up generic things

# If building this on RTD, mock out fortran sources
on_rtd = os.environ.get("READTHEDOCS", "false").lower() == "true"
_doc_skip = list(
    map(lambda x: x.lower(), os.environ.get("_SISL_DOC_SKIP", "").split(","))
)
skip_notebook = "notebook" in _doc_skip

# If building this on RTD, mock out fortran sources
if on_rtd:
    os.environ["SISL_NUM_PROCS"] = "1"


# General information about the project.
project = "sisl"
author = "sisl developers"
copyright = f"2015-{date.today().year}, {author}"


# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # Small extension just to measure the speed of compilation.
    # "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    # allows to view code directly in the homepage
    "sphinx.ext.viewcode",
    # Enable redirections
    "sphinxext.rediraffe",
    # toggle-button on info/warning/...
    "sphinx_togglebutton",
    # allow copybutton on code-blocks
    "sphinx_copybutton",
    # design, grids etc.
    "sphinx_design",
    "sphinxcontrib.jquery",  # a bug in 4.1.0 means search didn't work without explicit extension
    "sphinx_inline_tabs",
    # plotting and advanced usage
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.inheritance_diagram",
    "nbsphinx",
    "sphinx_gallery.load_style",
    # bibtex stuff
    "sphinxcontrib.bibtex",
]


# Define the prefix block that should not be copied in code-blocks
copybutton_prompt_text = r"\$ |\$> |>>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"

# We use numpy style docs
napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Converts type-definitions to references
napoleon_preprocess_types = True
# Puts notes in boxes
napoleon_use_admonition_for_notes = True

# If numpydoc is available, then let sphinx report warnings
numpydoc_validation_checks = {"all", "EX01", "SA01", "ES01"}

# These two options should solve the "toctree contains reference to nonexisting document"
# problem.
# See here: numpydoc #69
# class_members_toctree = False
# If this is false we do not have double method sections
# numpydoc_show_class_members = False

# Attributes section will be formatted as methods
numpydoc_attributes_as_param_list = False


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
from sisl_modules.github_links import GHFormat, GHLink


def _link_constructor(type):
    return GHLink(type), GHFormat(type)


extlinks = {
    # If these are changed, please update pyproject.toml under towncrier section
    "issue": _link_constructor("issues"),
    "pull": _link_constructor("pull"),
    "discussion": _link_constructor("discussions"),
    "doi": ("https://doi.org/%s", "%s"),
}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# prepend/append this snippet in _all_ sources
rst_prolog = """
.. highlight:: python
"""
# Insert the links into the epilog (globally)
# This means that every document has access to the links
rst_epilog = """
.. Internal links:
.. _sisl-git: https://github.com/zerothi/sisl/
.. _pr: https://github.com/zerothi/sisl/pulls
.. _issue: https://github.com/zerothi/sisl/issues
.. _sisl-discord: https://discord.gg/5XnFXFdkv2
.. _gh-releases: https://github.com/zerothi/sisl/releases
.. _pypi-releases: https://pypi.org/project/sisl
.. _conda-releases: https://anaconda.org/conda-forge/sisl
.. _sisl-doi: https://doi.org/10.5281/zenodo.597181
.. _sisl-files: https://github.com/zerothi/sisl-files

.. These are external links:
.. _MPL: https://www.mozilla.org/en-US/MPL/2.0/
.. _Cython: https://cython.org/
.. _Python: https://www.python.org/
.. _NetCDF: https://www.unidata.ucar.edu/netcdf
.. _cmake: https://cmake.org
.. _scikit-build-core: https://scikit-build-core.readthedocs.io/en/latest/
.. _netcdf4-py: https://github.com/Unidata/netcdf4-python
.. _numpy: https://www.numpy.org/
.. _scipy: https://docs.scipy.org/doc/scipy
.. _pyparsing: https://github.com/pyparsing/pyparsing
.. _matplotlib: https://matplotlib.org/
.. _pytest: https://docs.pytest.org/en/stable/
.. _pathos: https://github.com/uqfoundation/pathos
.. _tqdm: https://github.com/tqdm/tqdm
.. _xarray: https://xarray.pydata.org/en/stable/index.html
.. _workshop: https://github.com/zerothi/ts-tbt-sisl-tutorial
.. _plotly: https://plotly.com/python/

.. DFT codes
.. _atom: https://siesta-project.org/SIESTA_MATERIAL/Pseudos/atom_licence.html
.. _Siesta: https://siesta-project.org
.. _TranSiesta: https://siesta-project.org
.. _TBtrans: https://siesta-project.org
.. _BigDFT: http://www.bigdft.org
.. _OpenMX: http://www.openmx-square.org
.. _VASP: https://www.vasp.at
.. _ScaleUp: https://www.secondprinciples.unican.es
.. _GULP: https://nanochemistry.curtin.edu.au/gulp/news.cfm

.. Other programs heavily used
.. _ASE: https://wiki.fysik.dtu.dk/ase
.. _kwant: https://kwant-project.org
.. _XCrySDen: http://www.xcrysden.org
.. _VMD: https://www.ks.uiuc.edu/Research/vmd
.. _Molden: http://www.cmbi.ru.nl/molden
.. _Wannier90: http://www.wannier.org
"""

autosummary_generate = True

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
version = str(sisl.__version__)
if "dev" in version:
    _log.info("got {version=}")
    v_pre, v_suf = version.split("+")
    # remove dev (we don't need step)
    v_pre = v_pre.split(".dev")[0]
    # remove g in gHASH
    v_suf = v_suf[1:]
    if "." in v_suf:
        v_suf = v_suf.split(".")[0]
    version = f"{v_pre}-{v_suf}"
release = version
print(f"sisl version {version}")


# Add __init__ classes to the documentation
autoclass_content = "class"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "special-members": "__call__",
    "inherited-members": True,
    "show-inheritance": True,
}

# How to show the class signature
#  mixed: signature with class name
#  separated: signature as method
autodoc_class_signature = "separated"

# alphabetical | groupwise | bysource
# How automodule + autoclass orders content.
# Right now, the current way sisl documents things
# is basically groupwise. So lets be explicit
autodoc_member_order = "groupwise"

# Do not evaluate things that are defaulted in arguments.
# Show them *as-is*.
autodoc_preserve_defaults = True

# Show type-hints in only the description, in this way the
# signature is readable and the argument order can easily
# be inferred.
autodoc_typehints = "description"

# typehints only shows the minimal class, instead
# of full module paths
# The linkage is still problematic, and a known issue:
#  https://github.com/sphinx-doc/sphinx/issues/10455
# autodoc will likely get a rewrite. Until then..
autodoc_typehints_format = "short"

# Automatically create the autodoc_type_aliases
# This is handy for commonly used terminologies.
# It currently puts everything into a `<>` which
# is sub-optimal (i.e. one cannot do "`numpy.ndarray` or `any`")
# Perhaps just a small tweak and it works.
autodoc_type_aliases = {
    # general terms
    "array-like": "~numpy.ndarray",
    "array_like": "~numpy.ndarray",
    "int-like": "int or ~numpy.ndarray",
    "float-like": "float or ~numpy.ndarray",
    "sequence": "sequence",
    "np.ndarray": "~numpy.ndarray",
    "ndarray": "~numpy.ndarray",
}
_type_aliases_skip = set()


def has_under(name: str):
    return name.startswith("_")


# Retrive all typings
try:
    from numpy.typing import __all__ as numpy_types
except ImportError:
    numpy_types = []
try:
    from sisl.typing import __all__ as sisl_types
except ImportError:
    sisl_types = []

for name in numpy_types:
    if name in _type_aliases_skip:
        continue
    autodoc_type_aliases[f"npt.{name}"] = f"~numpy.typing.{name}"


for name in sisl_types:
    if name in _type_aliases_skip:
        continue

    # sisl typing should be last, in this way we ensure
    # that sisl typing is always preferred
    autodoc_type_aliases[name] = f"~sisl.typing.{name}"

# just for ease...
napoleon_type_aliases = autodoc_type_aliases

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = [
    "template.rst",
    "build",
    "_build",
    "**/setupegg.py",
    "**/setup.rst",
    "**/tests",
    "**.ipynb_checkpoints",
]
exclude_patterns.append("**/GUI with Python Demo.ipynb")
exclude_patterns.append("**/Building a plot class.ipynb")
for _venv in pathlib.Path(".").glob("*venv*"):
    exclude_patterns.append(str(_venv.name))

if skip_notebook:
    # Just exclude *ALL* notebooks to speed-up the documentation
    # creation.
    exclude_patterns.append("**/*.ipynb")

remove_from_toctrees = ["generated/*"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "autolink"

# prefer to use the smallest name, always
python_use_unqualified_type_names = True

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = [
    "sisl.",
    "sisl.geom",
    "sisl.physics",
    "sisl.viz",
    "sisl.unit",
    "sisl.typing",
    "sisl.shape",
]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

html_theme = "sphinx_rtd_theme"
# html_theme = "furo"

if html_theme == "furo":
    html_theme_options = {
        "source_repository": "https://github.com/zerothi/sisl/",
        "source_branch": "main",
        "source_directory": "docs/",
    }

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# We do not need to put in the version.
# It is located elsewhere
html_title = "sisl"

# A shorter title for the navigation bar.  Default is the same as html_title.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
if os.path.exists("_static"):
    html_static_path = ["_static"]
else:
    html_static_path = []

# Add any extra style files that we need
html_css_files = [
    "css/custom_styles.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css",
]

# If false, no index is generated.
html_use_modindex = True
html_use_index = True


# Redirects of moved pages
rediraffe_redirects = {
    "contribute.rst": "dev/index.rst",
    "visualization/viz_module/index.rst": "visualization/index.rst",
}

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    "papersize": "a4paper",
    # The font size ('10pt', '11pt' or '12pt').
    "pointsize": "11pt",
    # Additional stuff for the LaTeX preamble.
    "preamble": r"",
    # Latex figure (float) alignment
    "figure_align": "!htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index", "sisl.tex", project, author, "manual"),
]

#####
# Custom sisl documentation stuff down here
#####

# Plot directives for matplotlib
plot_include_source = True
plot_formats = [("png", 90)]
plot_pre_code = """\
import numpy as np
import matplotlib.pyplot as plt
import sisl as si"""


# Define header content
header = f"""\
.. currentmodule:: sisl

.. ipython:: python
   :suppress:

   import numpy as np
   import sisl as si
   import matplotlib.pyplot as plt

   np.random.seed(123987)
   np.set_printoptions(precision=4, suppress=True)
"""

# IPython executables
ipython_execlines = [
    "import numpy as np",
    "import sisl as si",
    "import matplotlib.pyplot as plt",
]

html_context = {
    "github_user": "zerothi",
    "github_repo": "sisl",
    "github_version": "main",
    "doc_path": "docs",
    "header": header,
}

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
# Python, numpy, scipy and matplotlib specify https as the default objects.inv
# directory. So please retain these links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
    "skimage": ("https://scikit-image.org/docs/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
}


# ---------------------
# BibTeX information
# ---------------------
bibtex_bibfiles = ["references.bib", "sisl_uses.bib"]
bibtex_default_style = "plain"
bibtex_tooltips = True


# Allow a year-month-author sorting
import calendar

from pybtex.style.formatting.plain import Style as PlainStyle
from pybtex.style.sorting.author_year_title import SortingStyle as AYTSortingStyle


class YearMonthAuthorSortStyle(AYTSortingStyle):
    def sorting_key(self, entry):
        ayt = super().sorting_key(entry)

        year = self._year_number(entry)
        month = self._month_number(entry)

        return (-year, -month, ayt[0], ayt[2])

    def _year_number(self, entry):
        year = entry.fields.get("year", 0)
        try:
            return int(year)
        except ValueError:
            pass
        return 0

    def _month_number(self, entry):
        month = entry.fields.get("month", "")
        for ext in ("abbr", "name"):
            lst = getattr(calendar, f"month_{ext}")[:]
            if month in lst:
                return lst.index(month)
        return 0


class RevYearPlain(PlainStyle):
    default_sorting_style = "sort_rev_year"


import pybtex

pybtex.plugin.register_plugin(
    "pybtex.style.sorting", "sort_rev_year", YearMonthAuthorSortStyle
)
pybtex.plugin.register_plugin("pybtex.style.formatting", "rev_year", RevYearPlain)

# Tell nbsphinx to wait, at least X seconds for each cell
nbsphinx_timeout = 30

# Insert a link to download the IPython notebook
nbsphinx_prolog = r"""
{% set docname = "docs/" + env.doc2path(env.docname, base=None) %}

.. raw:: html

    <div align="right">
    <a href="https://raw.githubusercontent.com/zerothi/sisl/main/{{ docname }}">
        <img alt="ipynb download badge"
            src="https://img.shields.io/badge/download-ipynb-blue.svg"
            style="vertical-align:text-bottom">
    </a>
    &nbsp;
    <a href="https://mybinder.org/v2/gh/zerothi/sisl/main?filepath={{ docname|e }}">
       <img alt="Binder badge"
            src="https://mybinder.org/badge_logo.svg"
            style="vertical-align:text-bottom">
    </a>
    </div>

"""

nbsphinx_thumbnails = {}


def sisl_method2class(meth):
    # Method to retrieve the class from a method (bounded and unbounded)
    # See stackoverflow.com/questions/3589311
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls

    if inspect.isfunction(meth):
        cls = getattr(
            inspect.getmodule(meth),
            meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0],
        )
        if isinstance(cls, type):
            return cls
    return None  # not required since None would have been implicitly returned anyway


autosummary_context = {
    "sisl_dispatch_attributes": [
        "plot",
        "apply",
        "to",
        "new",
    ],
    "sisl_skip_methods": [
        "ArgumentParser",
        "ArgumentParser_out",
        "is_keys",
        "key2case",
        "keys2case",
        "line_has_key",
        "line_has_keys",
        "readline",
        "step_to",
        "step_either",
        "isDataset",
        "isDimension",
        "isGroup",
        "isRoot",
        "isVariable",
        "InfoAttr",
    ],
}


# Run hacks to ensure the documentation shows proper
# documentation.
def assign_nested_attribute(cls: object, attribute_path: str, attribute: object):
    """Sets a nested attribute to a class with a placeholder name.

    It takes `cls` and sets the full `attribute_path` (with possible `.` in it)
    to `attribute`.

    It then also does this recursively for the objects located in the nested attribute.
    """

    # This sets the *full* attribute to the class
    setattr(cls, attribute_path, attribute)
    _log.info("adding %s attribute to class %s" % (attribute_path, cls.__name__))
    attribute_paths = attribute_path.split(".")

    if len(attribute_paths) > 1:
        attribute_cls = getattr(cls, attribute_paths[0])
        _log.info(
            "adding %s attribute to class %s"
            % (".".join(attribute_paths[1:]), attribute_cls.__name__)
        )
        setattr(attribute_cls, ".".join(attribute_paths[1:]), attribute)


def assign_nested_method(
    cls: object, method_path: str, method, signature_add_self: bool = False
):
    """Takes a nested method, wraps it to make sure is of function type and creates a nested attribute in the owner class."""

    @wraps(method)
    def wrapped_method(*args, **kwargs):
        return method(*args, **kwargs)

    if signature_add_self:
        wrapper_sig = inspect.signature(wrapped_method)
        wrapped_method.__signature__ = wrapper_sig.replace(
            parameters=[
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY),
                *wrapper_sig.parameters.values(),
            ]
        )

    # Make the method assigned as
    assign_nested_attribute(cls, method_path, wrapped_method)

    # I don't really see why this is required?
    head, tail, *_ = method_path.split(".")
    setattr(
        getattr(cls, head),
        tail,
        wrapped_method,
    )


def assign_class_dispatcher_methods(
    cls: object,
    dispatcher_name: Union[str, tuple[str, str]],
    signature_add_self: bool = False,
    as_attributes: bool = False,
):
    """Document all methods in a dispatcher class as nested methods in the owner class."""

    if isinstance(dispatcher_name, str):
        dispatcher_name = (dispatcher_name, "dispatch")

    dispatcher_name, method_name = dispatcher_name
    dispatcher = getattr(cls, dispatcher_name)

    _log.info("assign_class_dispatcher_methods found dispatcher: {dispatcher}")
    for key, method in dispatcher._dispatchs.items():
        if not isinstance(key, str):
            # TODO do not know yet what to do with object types used as extractions
            continue

        if method_name is None:
            dispatch = method
        else:
            dispatch = getattr(method, method_name)

        path = f"{dispatcher_name}.{key}"
        _log.info("assign_class_dispatcher_methods assigning attribute: {path}")
        # if dispatcher_name == "new":
        #    print(cls, dispatcher_name, path, method, dispatch, dispatch.__doc__)
        # if dispatcher_name == "to":
        #    print(cls, dispatcher_name, path, method, dispatch, dispatch.__doc__)
        if as_attributes:
            assign_nested_attribute(cls, path, dispatch)
        else:
            assign_nested_method(
                cls,
                path,
                dispatch,
                signature_add_self=signature_add_self,
            )


# My custom detailed instructions for not documenting stuff
# Run through all classes in sisl, extract attributes which
# are subclasses of the AbstractDispatcher.

_TRAVERSED = set()


def is_sisl_object(obj):
    """Check whether an object is coming from the sisl module."""
    try:
        # for objects
        return obj.__module__.startswith("sisl")
    except:
        pass
    try:
        # for modules
        return obj.__name__.startswith("sisl")
    except:
        pass
    return False


def yield_objects(module):
    global _TRAVERSED

    for name, member in inspect.getmembers(module):

        # We need to sort out things that are
        # originating from the sisl module. We don't care
        # about external modules here...
        if not is_sisl_object(member):
            continue

        if inspect.ismodule(member):
            # Never run through a module twice.
            # We will likely import modules again and again,
            # thus creating infinite loops.
            if name not in _TRAVERSED:
                _TRAVERSED.add(name)
                yield from yield_objects(member)

        elif inspect.isclass(member):
            if name not in _TRAVERSED:
                _TRAVERSED.add(name)
                yield member


def yield_types(obj: object, classes):
    """Yield any attributes/methods in `obj` which is has AbstractDispatcher as
    a baseclass."""
    for name in dir(obj):

        # False-positives could be Abstract methods etc.
        if not hasattr(obj, name):
            _log.info("skipping obj.%s due to hasattr error" % name)
            continue

        # Do not parse privates
        if name.startswith("_"):
            continue

        # get the actual attribute
        attr = getattr(obj, name)

        for istype in (issubclass, isinstance):
            # print("check", obj, attr, name, classes)
            try:
                if istype(attr, classes):
                    yield name, attr
                    break
            except:
                pass


_found_dispatch_attributes = set()
for obj in yield_objects(sisl):

    for name, attr in yield_types(obj, sisl._dispatcher.AbstractDispatcher):

        # If it is a plot, document the dispatch class itself, because it contains the right
        # documentation. Also in that case add self to the signature so that sphinx doesn't
        # hide the first argument
        if name in ["plot"]:
            dispatch_name = (name, None)
            signature_add_self = True
        else:
            dispatch_name = name
            signature_add_self = False

        # Fix the class dispatchers methods
        assign_class_dispatcher_methods(
            obj,
            dispatch_name,
            as_attributes=name in ["apply"],
            signature_add_self=signature_add_self,
        )
        # Collect all the different names where a dispatcher is associated.
        # In this way we die if we add a new one, without documenting it!
        _found_dispatch_attributes.add(name)

    for name, attr in yield_types(
        obj, (sisl.io._multiple.SileBound, sisl.io._multiple.SileBinder)
    ):

        assign_nested_attribute(obj, name, attr.__wrapped__)


if (
    len(
        diff := _found_dispatch_attributes
        - set(autosummary_context["sisl_dispatch_attributes"])
    )
    > 0
):
    raise ValueError(f"Found more sets than defined: {diff}")


def sisl_skip(app, what, name, obj, skip, options):
    global autodoc_default_options, autosummary_context
    # When adding routines here, please also add them
    # to the _templates/autosummary/class.rst file to limit
    # the documentation.
    if what == "class":
        if name in autosummary_context["sisl_skip_methods"]:
            _log.info(f"skip: {obj=} {what=} {name=}")
            return True
    # elif what == "attribute":
    #    return True

    # check for special methods (we don't want all)
    if has_under(name) and name not in autodoc_default_options.get(
        "special-members", ""
    ).split(","):
        return True

    try:
        cls = sisl_method2class(obj)
    except Exception:
        cls = None

    # Quick escape
    if cls is None:
        return skip

    # Currently inherited members will never be processed
    # Apparently they will be linked directly.
    # Now we have some things to disable the output of
    if "projncSile" in cls.__name__:
        if name in [
            "current",
            "current_parameter",
            "shot_noise",
            "noise_power",
            "fano",
            "density_matrix",
            "write_tbtav",
            "orbital_COOP",
            "atom_COOP",
            "orbital_COHP",
            "atom_COHP",
        ]:
            _log.info(f"skip: {obj=} {what=} {name=}")
            return True
    if cls.__name__.endswith("SilePHtrans"):
        if name in [
            "current",
            "atom_current",
            "bond_current",
            "vector_current",
            "orbital_current",
            "chemical_potential",
            "electron_temperature",
            "kT",
            "shot_noise",
            "noise_power",
        ]:
            _log.info(f"skip: {obj=} {what=} {name=}")
            return True
    return skip


from docutils.parsers.rst import directives

#######
# @pfebrer's suggestion for overriding the shown prefix for the templates.
from sphinx.ext.autosummary import Autosummary


class RemovePrefixAutosummary(Autosummary):
    """Wrapper around the autosummary directive to allow for custom display names.

    Adds a new option `:removeprefix:` which removes a prefix from the display names.
    """

    option_spec = {**Autosummary.option_spec, "removeprefix": directives.unchanged}

    def get_items(self, *args, **kwargs):
        items = super().get_items(*args, **kwargs)

        remove_prefix = self.options.get("removeprefix")
        if remove_prefix is not None:
            items = [(item[0].removeprefix(remove_prefix), *item[1:]) for item in items]

        return items


def _setup_autodoc(app):
    """Patch and fix autodoc so we get the correct formatting of the environment"""
    from sphinx.ext import autodoc, autosummary
    from sphinx.locale import _
    from sphinx.util import typing

    # These subsequent class and methods originate from mpi4py
    # which is released under BSD-3 clause.
    # All credits must go to mpi4py developers for their contribution!
    def istypealias(obj, name):
        if isinstance(obj, type):
            return name != getattr(obj, "__name__", None)
        return obj in (typing.Any,)

    def istypevar(obj):
        return isinstance(obj, typing.TypeVar)

    class TypeDocumenter(autodoc.DataDocumenter):
        objtype = "type"
        directivetype = "data"
        priority = autodoc.ClassDocumenter.priority + 1

        @classmethod
        def can_document_member(cls, member, membername, _isattr, parent):
            return (
                isinstance(parent, autodoc.ModuleDocumenter)
                and parent.name == "sisl.typing"
                and (istypevar(member) or istypealias(member, membername))
            )

        def add_directive_header(self, sig):
            if istypevar(self.object):
                obj = self.object
                if not self.options.annotation:
                    self.options.annotation = f' = TypeVar("{obj.__name__}")'
            super().add_directive_header(sig)

        def update_content(self, more_content):
            obj = self.object
            if istypevar(obj):
                if obj.__covariant__:
                    kind = _("Covariant")
                elif obj.__contravariant__:
                    kind = _("Contravariant")
                else:
                    kind = _("Invariant")
                content = f"{kind} :class:`~typing.TypeVar`."
                more_content.append(content, "")
                more_content.append("", "")
            if istypealias(obj, self.name):
                content = _("alias of %s") % typing.restify(obj)
                more_content.append(content, "")
                more_content.append("", "")
            super().update_content(more_content)

        def get_doc(self, *args, **kwargs):
            obj = self.object
            if istypevar(obj):
                if obj.__doc__ == typing.TypeVar.__doc__:
                    return []
            return super().get_doc(*args, **kwargs)

    # Ensure the data-type gets parsed as a role.
    # This will make the Type definitions print like a class
    # which gives more space than the simpler table with very
    # limited entry-space.
    from sphinx.domains.python import PythonDomain

    PythonDomain.object_types["data"].roles += ("class",)

    app.add_autodocumenter(TypeDocumenter)


def setup(app):
    # Setup autodoc skipping
    _setup_autodoc(app)

    app.connect("autodoc-skip-member", sisl_skip)
    app.add_directive("autosummary", RemovePrefixAutosummary, override=True)
