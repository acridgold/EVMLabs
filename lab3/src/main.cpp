#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру" << std::endl;
        return -1;
    }

    // Настройки для повышения FPS
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 60);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G')); // MJPEG кодек для ускорения

    cv::Mat frame, flipped;

    // Переменные для подсчета FPS
    int frameCount = 0;
    double fps = 0.0;
    auto start = std::chrono::high_resolution_clock::now();

    while (true) {
        cap >> frame;  // Считываем кадр из камеры
        if (frame.empty()) break;

        // Зеркальное отражение по горизонтали
        cv::flip(frame, flipped, 1);

        // Подсчет FPS
        frameCount++;
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (elapsed.count() >= 1.0) {
            fps = frameCount / elapsed.count();
            frameCount = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        // Отображение FPS на экране
        std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(flipped, fpsText, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Камера", flipped);

        // Выход из цикла по нажатию 'q' или ESC
        char c = (char)cv::waitKey(1);
        if (c == 27 || c == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
