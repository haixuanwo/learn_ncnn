
cmake_minimum_required(VERSION 3.5)

project(jh_tof_camera VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找软件包
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 指定头文件路径
include_directories(3rd_party/ncnn/include)
include_directories(src)
include_directories(model_include)

# 链接库路径
link_directories(3rd_party/ncnn/lib)

# 定义源文件，在src目录下所有.cpp和.h文件
file(GLOB_RECURSE SOURCES src/*.cpp src/*.h)

set(PROJECT_SOURCES
    ${SOURCES}
    main.cpp
)

add_executable(jh_tof_camera
    ${PROJECT_SOURCES}
)

target_link_libraries(jh_tof_camera
    ${OpenMP_CXX_FLAGS}
    ncnn
)
