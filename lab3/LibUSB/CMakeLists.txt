cmake_minimum_required(VERSION 3.10)
project(USBDevices)

# Используем PkgConfig для поиска libusb
find_package(PkgConfig REQUIRED)
pkg_check_modules(libusb REQUIRED IMPORTED_TARGET libusb-1.0)

# Создание исполняемого файла
add_executable(usb_devices usb_devices.cpp)

# Связывание с libusb через PkgConfig
target_link_libraries(usb_devices PRIVATE PkgConfig::libusb)

# Установка стандарта C++
set_property(TARGET usb_devices PROPERTY CXX_STANDARD 11)
