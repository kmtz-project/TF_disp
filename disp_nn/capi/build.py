from cffi import FFI
ffibuilder = FFI()

ffibuilder.cdef("void sgbm(float *, int, int);")

ffibuilder.set_source("_vision",  # name of the output C extension
"""
    #include "vision.h"',
""",
    sources=['vision.c'])  

if __name__ == "__main__":
    ffibuilder.compile()