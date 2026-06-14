CMAKE_MINIMUM_REQUIRED(VERSION 3.9 FATAL_ERROR)

# Name the plugin project.
# TODO: Set the project title to the name of your Python package.
project(hoomd-heat-transfer LANGUAGES C CXX)

# Find the installed HOOMD.
find_package(HOOMD 6.0.0 REQUIRED)

if(HOOMD_VERSION VERSION_GREATER_EQUAL 7.0.0)
    message(FATAL_ERROR "The support/hoomd-6.0 branch supports HOOMD-blue 6.x. Use a matching support branch for HOOMD ${HOOMD_VERSION}.")
endif()

message(STATUS "Found HOOMD ${HOOMD_VERSION}: ${HOOMD_INSTALL_PREFIX}/${PYTHON_SITE_INSTALL_DIR}")

# Force installation to the HOOMD installation location.
set(CMAKE_INSTALL_PREFIX ${HOOMD_INSTALL_PREFIX} CACHE PATH "Installation prefix" FORCE)

# Enable compiler warnings on gcc and clang.
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wconversion -Wno-sign-conversion -Wno-unknown-pragmas -Wno-deprecated-declarations -Wno-unused-result")
endif()

# Add the component's source directory.
add_subdirectory(src)
