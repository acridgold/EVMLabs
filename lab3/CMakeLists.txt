cmake_minimum_required(VERSION 3.15)
project(lab3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/NSU/MyLabs/EVMandPU/evm_lab_works/lab3/opencv/build-static")
set(OpenCV_STATIC ON)
set(BUILD_SHARED_LIBS OFF)

find_package(OpenCV REQUIRED COMPONENTS world)

add_executable(lab3 src/main.cpp)

target_include_directories(lab3 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lab3 PRIVATE ${OpenCV_LIBS})

if(MSVC)
    set_property(TARGET lab3 PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    target_link_libraries(lab3 PRIVATE
            ws2_32
            comctl32
            gdi32
            ole32
            oleaut32
            uuid
            vfw32
    )
endif()

if(MINGW)
    target_link_options(lab3 PRIVATE
            "-static"
            "-static-libgcc"
            "-static-libstdc++"
    )
endif()

if(APPLE)
    target_link_libraries(lab3 PRIVATE
            "-framework IOKit"
            "-framework Cocoa"
            "-framework OpenGL"
    )
endif()
