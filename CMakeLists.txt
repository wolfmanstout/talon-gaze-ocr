cmake_minimum_required(VERSION 3.12.0)

cmake_policy(SET CMP0054 NEW)
set(SKBUILD_LINK_LIBRARIES_KEYWORD PRIVATE)

set(THREADS_PREFER_PTHREAD_FLAG ON)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum OS X deployment version")
endif()

project(jarowinkler LANGUAGES C CXX)

find_package(PythonExtensions REQUIRED)
find_package(Python COMPONENTS Interpreter Development)
include(FetchContent)

set(JW_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

find_package(jaro_winkler 1.0.2 QUIET)
if (jaro_winkler_FOUND)
    message("Using system supplied version of jaro_winkler")
else()
    message("Using packaged version of jaro_winkler")
    add_subdirectory(extern/jarowinkler-cpp)
endif()

add_subdirectory(src/jarowinkler)
