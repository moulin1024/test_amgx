# Try to find AMGX library and headers

# Hints: allow AMGX_ROOT as CMake var or environment variable
set(_AMGX_ROOT_HINTS)
if(AMGX_ROOT)
    list(APPEND _AMGX_ROOT_HINTS "${AMGX_ROOT}")
endif()
if(DEFINED ENV{AMGX_ROOT})
    list(APPEND _AMGX_ROOT_HINTS "$ENV{AMGX_ROOT}")
endif()

find_path(AMGX_INCLUDE_DIR
    NAMES amgx_c.h
    HINTS ${_AMGX_ROOT_HINTS}
    PATH_SUFFIXES include Include
)

find_library(AMGX_LIBRARY
    NAMES amgxsh amgx libamgxsh libamgx
    HINTS ${_AMGX_ROOT_HINTS}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMGX
    REQUIRED_VARS AMGX_INCLUDE_DIR AMGX_LIBRARY
    FAIL_MESSAGE "AMGX not found. Set AMGX_ROOT to installation path"
)

if(AMGX_FOUND AND NOT TARGET AMGX::AMGX)
    add_library(AMGX::AMGX UNKNOWN IMPORTED)
    set_target_properties(AMGX::AMGX PROPERTIES
        IMPORTED_LOCATION "${AMGX_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${AMGX_INCLUDE_DIR}"
    )
endif()

set(AMGX_INCLUDE_DIRS "${AMGX_INCLUDE_DIR}")
set(AMGX_LIBRARIES "${AMGX_LIBRARY}")

mark_as_advanced(AMGX_INCLUDE_DIR AMGX_LIBRARY)


