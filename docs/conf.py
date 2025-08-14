# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Property Skipping & Table Injection ------------------------------------

import inspect
import os


def skip_property(app, what, name, obj, skip, options):
    if isinstance(obj, property):
        return True
    return skip


def inject_property_table(app, what, name, obj, options, lines):
    if what != "class":
        return

    props = []
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(obj, attr_name)
            if isinstance(attr, property):
                doc = inspect.getdoc(attr) or ""
                first_line = doc.strip().split("\n")[0]
                props.append((attr_name, first_line))
        except Exception:
            continue

    if not props:
        return

    output_dir = os.path.join("api", "_generated")
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)

    output_path = os.path.join(output_dir, f"{name}_properties.rst")
    

    with open(output_path, "w") as f:
        f.write(".. rubric:: Key Properties\n\n")
        f.write(".. list-table::\n")
        f.write("   :header-rows: 1\n")
        f.write("   :widths: 20 80\n\n")
        f.write("   * - **Property**\n")
        f.write("     - **Description**\n")
        for prop_name, desc in props:
            f.write(f"   * - ``{prop_name}``\n")
            f.write(f"     - {desc}\n")

def add_def_to_signature(app, what, name, obj, options, signature, return_annotation):
    if what == "method" and signature:
        return f"def {name}{signature}", return_annotation
    return None

def skip_member(app, what, name, obj, skip, options):
    # Skip all @property methods
    if isinstance(obj, property):
        return True

    # Skip special methods you don?t want
    if name in {"__new__", "__init_subclass__", "__class_getitem__"}:
        return True

    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
    app.connect("autodoc-skip-member", skip_property)
    app.connect("autodoc-process-docstring", inject_property_table)
    app.connect("autodoc-process-signature", add_def_to_signature)




# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ezphot'
copyright = '2025, Hyeonho Choi'
author = 'Hyeonho Choi'
release = '0.0.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
autodoc_member_order = 'bysource'
add_module_names = False
autodoc_class_signature = 'separated'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
