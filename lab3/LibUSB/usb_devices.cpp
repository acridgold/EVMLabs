#include <iostream>
#include <libusb.h>
#include <iomanip>

using namespace std;

const char* getDeviceClass(uint8_t class_code) {
    switch(class_code) {
        case 0x00: return "Device";
        case 0x01: return "Audio";
        case 0x02: return "Communications";
        case 0x03: return "HID (Human Interface Device)";
        case 0x05: return "Physical";
        case 0x06: return "Image";
        case 0x07: return "Printer";
        case 0x08: return "Mass Storage";
        case 0x09: return "Hub";
        case 0x0A: return "CDC-Data";
        case 0x0B: return "Smart Card";
        case 0x0D: return "Content Security";
        case 0x0E: return "Video";
        case 0x0F: return "Personal Healthcare";
        case 0x10: return "Audio/Video";
        case 0xEF: return "Miscellaneous";
        case 0xFE: return "Application Specific";
        case 0xFF: return "Vendor Specific";
        default: return "Unknown";
    }
}

string getSerialNumber(libusb_device_handle *dev, uint8_t serial_index) {
    if (serial_index == 0) {
        return "N/A";
    }

    unsigned char serial_number[256];
    int ret = libusb_get_string_descriptor_ascii(dev, serial_index, serial_number, sizeof(serial_number));

    if (ret < 0) {
        return "Error reading serial";
    }

    return string(reinterpret_cast<char*>(serial_number));
}

int main() {
    libusb_device **devs = NULL;
    libusb_context *ctx = NULL;
    int r = libusb_init(&ctx);

    if (r < 0) {
        cerr << "Error initializing libusb: " << libusb_error_name(r) << endl;
        return -1;
    }

    // Логирование
    libusb_set_option(ctx, LIBUSB_OPTION_LOG_LEVEL, LIBUSB_LOG_LEVEL_WARNING);

    // Список всех USB-устройств
    ssize_t cnt = libusb_get_device_list(ctx, &devs);

    if (cnt < 0) {
        cerr << "Error getting device list: " << libusb_error_name(cnt) << endl;
        libusb_exit(ctx);
        return -1;
    }

    cout << "Found " << cnt << " USB device(s)" << endl;
    cout << "========================================" << endl;
    cout << endl;

    for (ssize_t i = 0; i < cnt; i++) {
        libusb_device *dev = devs[i];
        libusb_device_descriptor desc{};

        // Дескриптор
        r = libusb_get_device_descriptor(dev, &desc);
        if (r < 0) {
            cerr << "Error getting device descriptor: " << libusb_error_name(r) << endl;
            continue;
        }

        cout << "Device " << (int)i + 1 << ":" << endl;

        // Вывод класса
        cout << " -- Class: " << getDeviceClass(desc.bDeviceClass)
             << " (0x" << hex << setfill('0') << setw(2) << (int)desc.bDeviceClass << ")" << dec << endl;

        // Вывод VID и PID
        cout << " ---- Vendor ID: 0x" << hex << setfill('0') << setw(4) << desc.idVendor << dec << endl;
        cout << " ------ Product ID: 0x" << hex << setfill('0') << setw(4) << desc.idProduct << dec << endl;

        libusb_device_handle *handle = NULL;
        r = libusb_open(dev, &handle);

        if (r == 0 && handle != NULL) {
            // Получение и вывод серийного номера
            string serial = getSerialNumber(handle, desc.iSerialNumber);
            cout << " -------- Serial Number: " << serial << endl;

            libusb_close(handle);
        } else {
            cout << "  Serial Number: Could not open device (may require root/sudo)" << endl;
        }

        cout << endl;
    }

    libusb_free_device_list(devs, 1);
    libusb_exit(ctx);

    cout << "Done!" << endl;
    return 0;
}
