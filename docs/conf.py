# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

from rocm_docs import ROCmDocs


file = open("../CMakeLists.txt")
text = file.read()
name_matches = re.findall("set \( ROCPROFILER_NAME.*", text)
name = re.findall(r'"([^"]*)"', name_matches[0])[0]
version_matches = re.findall("get_version.*", text)
version = re.findall(r'"([^"]*)"', version_matches[0])[0]
file.close()
if len(version) > 0:
    name = f"{name} {version} Documentation"

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(name)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/docBin/xml")
docs_core.enable_api_reference()
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
