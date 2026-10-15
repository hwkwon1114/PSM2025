cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(ShellSolver LANGUAGES C CXX)

# Suppress deprecation warnings emitted by third-party external subprojects
if(POLICY CMP0218)
    cmake_policy(SET CMP0218 OLD)
endif()
set(CMAKE_WARN_DEPRECATED OFF)
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

# Location of additional cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Create compile_commands.json for language tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Safeguards against in-source builds and sets default build type
include(safeguards)
include(FetchContent)

# Base directories
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(TEST_SRC_DIR "${PROJECT_SOURCE_DIR}/test")
set(EXT_PROJECTS_DIR "${PROJECT_SOURCE_DIR}/external")

set(SHELL_OUTPUT_ROOT "${CMAKE_SOURCE_DIR}" CACHE PATH "Build-product root; defaults preserve existing bin/lib locations")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${SHELL_OUTPUT_ROOT}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SHELL_OUTPUT_ROOT}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SHELL_OUTPUT_ROOT}/bin")

# C++ Standard: C++23 required
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Modern compiler warning flags scoped to project targets
message(STATUS "We are on a ${CMAKE_SYSTEM_NAME} system")
message(STATUS "The host processor is ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(PROJECT_CXX_WARNINGS
        -Wall
        -Wextra
        -Wcast-align
        -Wformat=2
        -Wmissing-include-dirs
        -Wno-missing-braces
    )
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
    set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -O3 -march=native -mtune=native")
endif()

# Source files with CONFIGURE_DEPENDS so adding/removing files updates build graph
file(GLOB SOURCES_SIM CONFIGURE_DEPENDS "${SRC_DIR}/simulations/*.cpp")
file(GLOB HEADERS_SIM CONFIGURE_DEPENDS "${SRC_DIR}/simulations/*.hpp")
file(GLOB SOURCES_LIB CONFIGURE_DEPENDS "${SRC_DIR}/libshell/*.cpp")
file(GLOB HEADERS_LIB CONFIGURE_DEPENDS "${SRC_DIR}/libshell/*.hpp")
file(GLOB SOURCES_HLBFGS CONFIGURE_DEPENDS "${EXT_PROJECTS_DIR}/hlbfgs/*.cpp")
file(GLOB SOURCES_TEST CONFIGURE_DEPENDS "${TEST_SRC_DIR}/testshell/*.cpp")

# Sim_Bilayer_Growth: -O2 compile option to keep memory modest on build nodes
set_source_files_properties(
    "${SRC_DIR}/simulations/Sim_Bilayer_Growth.cpp"
    PROPERTIES COMPILE_OPTIONS "-O2"
)

# ------------------------------------------------------------
# FetchContent policies and defaults
# ------------------------------------------------------------
if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()
if(POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()
set(FETCHCONTENT_QUIET ON)

# ------------------------------------------------------------
# External project: HLBFGS (compiled as static library)
# ------------------------------------------------------------
add_library(hlbfgs STATIC ${SOURCES_HLBFGS})
target_compile_definitions(hlbfgs PRIVATE HLBFGS_EIGEN)

# ------------------------------------------------------------
# Main library: libshell
# ------------------------------------------------------------
add_library(libshell STATIC ${SOURCES_LIB} ${HEADERS_LIB})
target_include_directories(libshell PUBLIC "${SRC_DIR}/libshell")
target_include_directories(libshell SYSTEM PUBLIC
    "${EXT_PROJECTS_DIR}/hlbfgs"
    "${EXT_PROJECTS_DIR}/tinyad/include"
)
target_compile_definitions(libshell PUBLIC USEHLBFGS)
target_compile_options(libshell PRIVATE ${PROJECT_CXX_WARNINGS})
target_link_libraries(libshell PUBLIC hlbfgs)

# ------------------------------------------------------------
# Main executable: shell
# ------------------------------------------------------------
add_executable(shell ${SOURCES_SIM} ${HEADERS_SIM})
target_compile_options(shell PRIVATE ${PROJECT_CXX_WARNINGS})
target_link_libraries(shell PRIVATE libshell)
add_custom_command(TARGET shell POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:shell> "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/sim_bilayer_growth"
    COMMENT "Creating symlink sim_bilayer_growth -> shell"
)

# ------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(libshell PUBLIC OpenMP::OpenMP_CXX)
endif()

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Found Eigen3 in: ${EIGEN3_INCLUDE_DIR}")
target_include_directories(libshell SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR})
target_include_directories(hlbfgs SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})

# SuiteSparse / CHOLMOD direct solver support and native hybrid solver (default ON)
option(ENABLE_CHOLMOD "Enable SuiteSparse CHOLMOD solver backend" ON)
option(ENABLE_NATIVE_HYBRID "Build the native L-BFGS/fresh-QR CHOLMOD shell hybrid solver" ON)
set(SHELL_HAVE_CHOLMOD FALSE)
if(ENABLE_CHOLMOD)
    find_package(SuiteSparse QUIET COMPONENTS CHOLMOD)
    if(SuiteSparse_FOUND)
        message(STATUS "Found SuiteSparse CHOLMOD: ${SuiteSparse_LIBRARIES}")
        set(SHELL_HAVE_CHOLMOD TRUE)
        target_compile_definitions(libshell PUBLIC NONCONVEX_WITH_CHOLMOD=1)
        target_include_directories(libshell PUBLIC ${SuiteSparse_INCLUDE_DIRS})
        target_link_libraries(libshell PUBLIC ${SuiteSparse_LIBRARIES})
    else()
        find_path(CHOLMOD_INCLUDE_DIR cholmod.h
            HINTS ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT
            PATH_SUFFIXES include suitesparse)
        find_library(CHOLMOD_LIBRARY cholmod
            HINTS ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT
            PATH_SUFFIXES lib lib64)
        if(CHOLMOD_INCLUDE_DIR AND CHOLMOD_LIBRARY)
            message(STATUS "Found CHOLMOD via hint path: ${CHOLMOD_INCLUDE_DIR}")
            set(SHELL_HAVE_CHOLMOD TRUE)
            target_compile_definitions(libshell PUBLIC NONCONVEX_WITH_CHOLMOD=1)
            target_include_directories(libshell PUBLIC ${CHOLMOD_INCLUDE_DIR})
            target_link_libraries(libshell PUBLIC ${CHOLMOD_LIBRARY})
            get_filename_component(SUITESPARSE_LIB_DIR "${CHOLMOD_LIBRARY}" DIRECTORY)
            find_library(AMD_LIB amd HINTS "${SUITESPARSE_LIB_DIR}" ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT PATH_SUFFIXES lib lib64)
            find_library(COLAMD_LIB colamd HINTS "${SUITESPARSE_LIB_DIR}" ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT PATH_SUFFIXES lib lib64)
            find_library(CAMD_LIB camd HINTS "${SUITESPARSE_LIB_DIR}" ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT PATH_SUFFIXES lib lib64)
            find_library(CCOLAMD_LIB ccolamd HINTS "${SUITESPARSE_LIB_DIR}" ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT PATH_SUFFIXES lib lib64)
            find_library(SUITESPARSE_CONFIG_LIB suitesparseconfig HINTS "${SUITESPARSE_LIB_DIR}" ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT PATH_SUFFIXES lib lib64)
            find_library(METIS_LIB metis HINTS "${SUITESPARSE_LIB_DIR}" ENV SUITESPARSE_DIR ENV SUITESPARSE_ROOT PATH_SUFFIXES lib lib64)
            if(AMD_LIB)
                message(STATUS "Found AMD: ${AMD_LIB}")
                target_link_libraries(libshell PUBLIC ${AMD_LIB})
            endif()
            if(COLAMD_LIB)
                message(STATUS "Found COLAMD: ${COLAMD_LIB}")
                target_link_libraries(libshell PUBLIC ${COLAMD_LIB})
            endif()
            if(CAMD_LIB)
                message(STATUS "Found CAMD: ${CAMD_LIB}")
                target_link_libraries(libshell PUBLIC ${CAMD_LIB})
            endif()
            if(CCOLAMD_LIB)
                message(STATUS "Found CCOLAMD: ${CCOLAMD_LIB}")
                target_link_libraries(libshell PUBLIC ${CCOLAMD_LIB})
            endif()
            if(SUITESPARSE_CONFIG_LIB)
                message(STATUS "Found SuiteSparse_config: ${SUITESPARSE_CONFIG_LIB}")
                target_link_libraries(libshell PUBLIC ${SUITESPARSE_CONFIG_LIB})
            endif()
            if(METIS_LIB)
                message(STATUS "Found METIS: ${METIS_LIB}")
                target_link_libraries(libshell PUBLIC ${METIS_LIB})
            endif()
        else()
            message(STATUS "SuiteSparse CHOLMOD not found; falling back to Eigen direct solvers.")
        endif()
    endif()
endif()

if(ENABLE_NATIVE_HYBRID)
    if(NOT SHELL_HAVE_CHOLMOD)
        message(FATAL_ERROR "ENABLE_NATIVE_HYBRID requires ENABLE_CHOLMOD=ON and an available CHOLMOD backend")
    endif()
    find_package(nlohmann_json CONFIG REQUIRED)
    target_link_libraries(libshell PUBLIC nlohmann_json::nlohmann_json)
    target_compile_definitions(libshell PUBLIC SHELL_WITH_NATIVE_HYBRID=1)
endif()

# TRIANGLE (download directly from LIBIGL and create library)
FetchContent_Declare(
    triangle
    GIT_REPOSITORY https://github.com/libigl/triangle
    GIT_TAG master
)
FetchContent_MakeAvailable(triangle)
message(STATUS "Triangle available in: ${triangle_SOURCE_DIR}")
target_compile_definitions(triangle PRIVATE TRILIBRARY)
target_compile_definitions(libshell PUBLIC USETRILIBRARY)
target_include_directories(libshell SYSTEM PUBLIC ${triangle_SOURCE_DIR})
target_link_libraries(libshell PUBLIC triangle)

# LIBIGL
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG main
)
FetchContent_MakeAvailable(libigl)
message(STATUS "LIBIGL available in: ${libigl_SOURCE_DIR}")
target_include_directories(libshell SYSTEM PUBLIC ${libigl_SOURCE_DIR}/include)

# TBB
set(TBB_USE_DEBUG_BUILD FALSE)
find_package(TBB REQUIRED)
target_include_directories(libshell SYSTEM PUBLIC ${TBB_INCLUDE_DIRS})
target_link_libraries(libshell PUBLIC TBB::tbb TBB::tbbmalloc)

# VTK (OPTIONAL BUT STRONGLY RECOMMENDED FOR IO)
find_package(VTK REQUIRED COMPONENTS CommonCore CommonDataModel RenderingCore FiltersCore FiltersSources IOXML)
if(VTK_FOUND)
    message(STATUS "Found VTK in: ${VTK_DIR}")
    target_compile_definitions(libshell PUBLIC USEVTK)
    if(VTK_MAJOR_VERSION VERSION_LESS "8.9.0")
        include(${VTK_USE_FILE})
        target_link_libraries(libshell PUBLIC ${VTK_LIBRARIES})
    else()
        message(STATUS "This is a recent version of VTK, importing CMake targets from VTK build.")
        target_link_libraries(libshell PUBLIC
            VTK::CommonCore
            VTK::CommonDataModel
            VTK::RenderingCore
            VTK::FiltersCore
            VTK::FiltersSources
            VTK::IOXML
        )
    endif()
else()
    message(STATUS "Did not find VTK - compiling without")
endif()

# GSL (OPTIONAL BUT SOME TESTS WONT WORK WITHOUT IT)
find_package(GSL)
if(GSL_FOUND)
    message(STATUS "Found GSL in: ${GSL_INCLUDE_DIRS}")
    target_include_directories(libshell SYSTEM PUBLIC ${GSL_INCLUDE_DIRS})
    target_link_libraries(libshell PUBLIC ${GSL_LIBRARIES})
    target_compile_definitions(libshell PUBLIC USEGSL)
else()
    message(STATUS "Did not find GSL - compiling without")
endif()

# ------------------------------------------------------------
# Testing: GoogleTest
# ------------------------------------------------------------
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(testshell ${SOURCES_TEST})
target_compile_options(testshell PRIVATE ${PROJECT_CXX_WARNINGS})
target_include_directories(testshell PRIVATE "${SRC_DIR}/simulations")
target_link_libraries(testshell PRIVATE gtest gmock_main libshell)
add_test(NAME testshell COMMAND testshell)

# Focused opt-in correctness fixture; excluded from the normal build. It cannot
# dispatch a physical forward/campaign and avoids compiling all legacy tests.
if(ENABLE_NATIVE_HYBRID)
    add_executable(native_hybrid_fixtures EXCLUDE_FROM_ALL test/testshell/Test_NativeHybridProduction.cpp)
    target_compile_options(native_hybrid_fixtures PRIVATE -O1 ${PROJECT_CXX_WARNINGS})
    target_link_libraries(native_hybrid_fixtures PRIVATE gtest_main libshell)
endif()
