# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import subprocess

from rocm_docs import ROCmDocs


get_name = r'sed -n -e "s/^project(\([A-Za-z-]\+\).*/\1/p" ../CMakeLists.txt'
get_version = r'sed -n -e "s/^project(.* \([0-9\.]\{1,\}\).*/\1/p" ../CMakeLists.txt'
name = subprocess.getoutput(get_name)
version = subprocess.getoutput(get_version)
if len(version) > 0:
    name = f"{name} {version} Documentation"

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(name)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/docBin/xml")
docs_core.enable_api_reference()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
