cmake_minimum_required(VERSION 3.15)
project(lab5)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(lab5 src/main.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MINGW)
    target_compile_options(lab5 PRIVATE -O1)
endif()

add_custom_target(show_flags ALL
        COMMAND ${CMAKE_CXX_COMPILER} -v
        COMMENT "Show compiler"
)

message(STATUS "CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")

if(MSVC)
    set_property(TARGET lab5 PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    target_link_libraries(lab5 PRIVATE
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
    target_link_options(lab5 PRIVATE
            "-static"
            "-static-libgcc"
            "-static-libstdc++"
    )
endif()

if(APPLE)
    target_link_libraries(lab5 PRIVATE
            "-framework IOKit"
            "-framework Cocoa"
            "-framework OpenGL"
    )
endif()
