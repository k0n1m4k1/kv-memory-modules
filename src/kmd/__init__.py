"""kmd — compile Markdown agent memories into relocatable KV-cache modules.

Modules:
    llamalib  ctypes bindings over the llama.cpp shared library + repo paths
    mdc       the .kmd compiler/linker CLI (`mdc` console script)
    hyblib    hybrid-model (attention + recurrent) linker machinery
    linker    original Phase-B linking harness (kept for reproducibility)
"""
