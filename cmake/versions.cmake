# cmake/versions.cmake — reads versions.env into CMake cache variables.
#
# Must be included before CMakeLists.txt uses FetchContent or any version-pinned dep:
# that file reads these declarations to interpolate the versions set here.
#
# Every variable becomes a CACHE STRING, so -D on the command line wins:
#   cmake -S . -B build -DNEURIPLO_TASKS_VERSION=v0.9.0
#
# To add a pin: put it in versions.env, then expose it below with
# tritonic_declare_version(<NAME> "<docstring>"). Names match versions.env exactly
# so `grep -rn NEURIPLO_TASKS_VERSION` finds the pin and all of its consumers.

include_guard(GLOBAL)

set(TRITONIC_VERSIONS_ENV "${CMAKE_CURRENT_LIST_DIR}/../versions.env" CACHE FILEPATH
    "Path to the versions.env single source of truth")

if(NOT EXISTS "${TRITONIC_VERSIONS_ENV}")
    message(FATAL_ERROR "versions.env not found at ${TRITONIC_VERSIONS_ENV}")
endif()

# Re-run CMake when the pins change.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${TRITONIC_VERSIONS_ENV}")

# Parse KEY=VALUE lines into _TRITONIC_ENV_<KEY> in the current scope. Comments,
# blank lines and anything without a shell-identifier key are skipped.
function(_tritonic_read_versions_env path)
    file(STRINGS "${path}" _lines)
    foreach(_line IN LISTS _lines)
        string(STRIP "${_line}" _line)
        if(_line STREQUAL "" OR _line MATCHES "^#")
            continue()
        endif()
        if(NOT _line MATCHES "^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
            continue()
        endif()
        set(_key "${CMAKE_MATCH_1}")
        set(_value "${CMAKE_MATCH_2}")
        # Strip one layer of surrounding quotes, as a shell would.
        string(REGEX REPLACE "^\"(.*)\"$" "\\1" _value "${_value}")
        string(REGEX REPLACE "^'(.*)'$" "\\1" _value "${_value}")
        set("_TRITONIC_ENV_${_key}" "${_value}" PARENT_SCOPE)
    endforeach()
endfunction()

# Publish one parsed key as a cache variable. Fails loudly rather than silently
# expanding to the empty string, which would produce a plausible-looking but
# wrong download URL.
#
# A plain `set(... CACHE STRING ...)` will not do here: it never overwrites an
# existing cache entry, so editing versions.env in an *existing* build tree would
# reconfigure (CMAKE_CONFIGURE_DEPENDS sees the file change) and still build the
# old version — the source of truth defeated, silently, for everyone who does not
# wipe their build directory.
#
# So each pin also records the versions.env value it was configured from, in an
# INTERNAL stamp. On reconfigure:
#   cache == stamp   the value is still the file-derived default and the file has
#                    moved on -> adopt the new one.
#   cache != stamp   somebody passed -D<name>=... -> never clobber it.
# On a first configure there is no stamp, and the non-FORCE set already does the
# right thing: honour -D if present, otherwise take the file value.
macro(tritonic_declare_version name doc)
    if(NOT DEFINED _TRITONIC_ENV_${name})
        message(FATAL_ERROR "${name} is missing from ${TRITONIC_VERSIONS_ENV}")
    endif()
    set(_tritonic_default "${_TRITONIC_ENV_${name}}")

    if(DEFINED _TRITONIC_STAMP_${name} AND DEFINED ${name}
       AND "${${name}}" STREQUAL "${_TRITONIC_STAMP_${name}}"
       AND NOT "${${name}}" STREQUAL "${_tritonic_default}")
        set(${name} "${_tritonic_default}" CACHE STRING "${doc}" FORCE)
        message(STATUS "  ${name}: ${_TRITONIC_STAMP_${name}} -> ${_tritonic_default} (versions.env changed)")
    else()
        set(${name} "${_tritonic_default}" CACHE STRING "${doc}")
    endif()

    set(_TRITONIC_STAMP_${name} "${_tritonic_default}" CACHE INTERNAL
        "versions.env value ${name} was last configured from")
endmacro()

_tritonic_read_versions_env("${TRITONIC_VERSIONS_ENV}")

tritonic_declare_version(NEURIPLO_TASKS_VERSION "neuriplo-tasks git tag")
tritonic_declare_version(GTEST_VERSION           "GoogleTest version")
tritonic_declare_version(NGC_CONTAINER_TAG       "NGC monthly container tag")
tritonic_declare_version(TRITON_REPO_TAG         "triton-inference-server/{common,core,backend} tag")
tritonic_declare_version(NGC_TENSORRT_VERSION    "TensorRT version shipped in NGC_CONTAINER_TAG")
tritonic_declare_version(NGC_DALI_VERSION        "NVIDIA DALI version shipped in NGC_CONTAINER_TAG")
tritonic_declare_version(NGC_CUDA_VERSION        "CUDA Toolkit series shipped in NGC_CONTAINER_TAG")
tritonic_declare_version(CUDA_ARCHITECTURES      "Default CMAKE_CUDA_ARCHITECTURES")

message(STATUS "Dependency pins from ${TRITONIC_VERSIONS_ENV}:")
message(STATUS "  neuriplo-tasks ${NEURIPLO_TASKS_VERSION} | TensorRT ${NGC_TENSORRT_VERSION} | NGC ${NGC_CONTAINER_TAG}")
