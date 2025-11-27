cmake_minimum_required(VERSION 3.15)
project(lab4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# vcpkg автоматически найдет OpenBLAS
find_package(OpenBLAS REQUIRED)

add_executable(lab4 src/main.cpp)

target_link_libraries(lab4 PRIVATE OpenBLAS::OpenBLAS)

if(MINGW)
    target_link_options(lab4 PRIVATE
            "-static"
            "-static-libgcc"
            "-static-libstdc++"
    )
endif()
