cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(interaction VERSION 5.0)
enable_language(CXX)

# CMake 3.30+ removes the legacy FindBoost module. RDKit's config supports the
# upstream Boost package config, so select that behavior without warning.
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()

#########################################################################
#### Tweak the default configuration
#########################################################################

# define an option for using native cpp code
set(TARGET_CPU_ARCHITECTURE "generic" CACHE STRING "The target cpu for architecture-specific optimizations")
set_property(CACHE TARGET_CPU_ARCHITECTURE PROPERTY STRINGS generic native )

# the idea is that we want to build the optimized version, unless the user says otherwise
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The type of build" FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

# export also compile commands to have a better integration with the ide
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# enable additional warning flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic -Wl,-z,defs -Wl,-z,now -Wl,-z,relro)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wdouble-promotion)
endif()

# enable colored compiler output (with Ninja)
if ("${CMAKE_GENERATOR}" STREQUAL "Ninja")
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    endif()
endif()

#########################################################################
#### Add external dependency
#########################################################################

# ONNX Runtime: runs the exported RTMScore neural network.
# Homebrew installs a CMake config; if it is elsewhere, pass
#   -Donnxruntime_DIR=/path/to/lib/cmake/onnxruntime
find_package(onnxruntime CONFIG REQUIRED)

# RDKit: path "b" featurizes the protein/ligand directly in C++. Homebrew ships a
# CMake config at /opt/homebrew/lib/cmake/rdkit; if elsewhere pass
#   -Drdkit_DIR=/path/to/lib/cmake/rdkit
find_package(rdkit CONFIG REQUIRED)

#########################################################################
#### Define the application sources
#########################################################################

# application headers
set(header_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND header_files
  "${header_path}/main.hpp"
  "${header_path}/featurize.hpp"
  "${header_path}/geometry.hpp"
  "${header_path}/pocket.hpp"
  "${header_path}/batch.hpp"
)

# application sources
set(source_path "${CMAKE_CURRENT_SOURCE_DIR}/src")
list(APPEND source_files
  "${source_path}/main.cpp"
  "${source_path}/featurize_ligand.cpp"
  "${source_path}/featurize_protein.cpp"
  "${source_path}/pocket.cpp"
  "${source_path}/batch.cpp"
)


#########################################################################
#### Define the compilation step
#########################################################################

# define the target and its properties
add_executable(interaction ${header_files} ${source_files})
set_target_properties(interaction
  PROPERTIES
    CXX_STANDARD 20  # RDKit 2026.03 headers require C++20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# enable link-time optimizations
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported)
if(ipo_supported)
    set_property(TARGET interaction PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# Homebrew RDKit's imported targets link keg-only dependencies by bare name.
# Keep this workaround scoped to a Homebrew RDKit installation so an activated
# Mamba environment cannot accidentally link libraries from /opt/homebrew.
if(APPLE AND rdkit_DIR MATCHES "^/opt/homebrew/")
  foreach(dep maeparser coordgen inchi)
    if(EXISTS "/opt/homebrew/opt/${dep}/lib")
      target_link_directories(interaction PUBLIC "/opt/homebrew/opt/${dep}/lib")
    endif()
  endforeach()
endif()

# link ONNX Runtime and RDKit (imported targets carry their own include dirs)
target_link_libraries(interaction PUBLIC
  onnxruntime::onnxruntime
  RDKit::FileParsers
  RDKit::SmilesParse
  RDKit::CIPLabeler
  RDKit::GraphMol
  RDKit::SubstructMatch
  RDKit::RDGeometryLib
  RDKit::RDGeneral
)
