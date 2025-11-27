cmake_minimum_required(VERSION 3.15)
project(lab4)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lab4 src/main.cpp)

if(MSVC)
    set_property(TARGET lab4 PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    target_link_libraries(lab4 PRIVATE
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
    target_link_options(lab4 PRIVATE
            "-static"
            "-static-libgcc"
            "-static-libstdc++"
    )
endif()

if(APPLE)
    target_link_libraries(lab4 PRIVATE
            "-framework IOKit"
            "-framework Cocoa"
            "-framework OpenGL"
    )
endif()
