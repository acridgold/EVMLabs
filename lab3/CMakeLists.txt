cmake_minimum_required(VERSION 3.15)
project(lab3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/NSU/MyLabs/EVMandPU/evm_lab_works/lab3/opencv/opencv/build")

find_package(OpenCV REQUIRED)

add_executable(lab3 src/main.cpp)

target_include_directories(lab3 PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(lab3 PRIVATE ${OpenCV_LIBS})

if(MINGW)
    target_link_options(lab3 PRIVATE "-static" "-static-libgcc" "-static-libstdc++")
endif()

if(APPLE)
    target_link_libraries(lab3 PRIVATE "-framework IOKit")
    target_link_libraries(lab3 PRIVATE "-framework Cocoa")
    target_link_libraries(lab3 PRIVATE "-framework OpenGL")
endif()
