cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(PoissonSurfaceReconstruction LANGUAGES CXX CUDA)

# Language options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cuda
find_package(CUDA 11.6 REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
set(CUDA_ARCH "52;60;70;80;86" CACHE STRING "Architecture(s) for which to generate CUDA PTX code")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS "52" "60" "70" "80" "86")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CUDA_ARCH}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

# GLFW package
find_package(glfw3 REQUIRED)

if(MSVC)
    option( MSVC_USE_DYNAMIC_CRT  "Use static C Runtime with MSVC, /MD instead of /MT" ON)

    # Make sure there are no erroneous C Runtime flags
    list(APPEND FLAG_VARS
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
    )

    if(MSVC_USE_DYNAMIC_CRT)
        foreach(FLAG_VAR ${FLAG_VARS})
            string(REGEX REPLACE "/MT" "/MD" NEW_FLAGS "${${FLAG_VAR}}")
            set(${FLAG_VAR} "${NEW_FLAGS}" CACHE STRING "" FORCE)
        endforeach()
    else()
        foreach(FLAG_VAR ${FLAG_VARS})
            string(REGEX REPLACE "/MD" "/MT" NEW_FLAGS "${${FLAG_VAR}}")
            set(${FLAG_VAR} "${NEW_FLAGS}" CACHE STRING "" FORCE)
        endforeach()
    endif()
endif(MSVC)

# The library and compiler flags on Ubuntu
if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -G")
endif(UNIX)

INCLUDE_DIRECTORIES(${CUB_DIR})
INCLUDE_DIRECTORIES(${EIGEN_DIR})
INCLUDE_DIRECTORIES(${BOOST_DIR})
INCLUDE_DIRECTORIES(${GLAD_DIR})

# PCL
find_package(PCL REQUIRED)
INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})

# Include
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/base)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/core)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/math)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/mesh)
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/render)

# Files
file(GLOB_RECURSE BASE "base/*.cpp" "base/*.hpp" "base/*.h" "base/*.cuh" "base/*.cu")
file(GLOB_RECURSE CORE "core/*.h" "core/*.cu" "core/*.cuh")
file(GLOB_RECURSE MATH "math/*.cpp" "math/*.h" "math/*.cu") 
file(GLOB_RECURSE MESH "mesh/*.cpp" "mesh/*.h" "mesh/*.cu" "mesh/*.cuh") 
file(GLOB_RECURSE RENDER "render/*.cpp" "render/*.h") 
file(GLOB_RECURSE OTHER "*.cpp") 

# Message
message(STATUS "BASE files: ${BASE}")
message(STATUS "CORE files: ${CORE}")
message(STATUS "MATH files: ${MATH}")
message(STATUS "MESH files: ${MESH}")
message(STATUS "RENDER files: ${RENDER}")
message(STATUS "OTHER files: ${OTHER}")

add_executable(PoissonSurfaceReconstruction ${BASE} ${CORE} ${MATH} ${MESH} ${RENDER} ${OTHER})

set_target_properties(PoissonSurfaceReconstruction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(PoissonSurfaceReconstruction ${CUDA_LIBRARIES} ${CUDA_cusparse_LIBRARY} ${CUDA_cusolver_LIBRARY} ${CUDA_cublas_LIBRARY})
