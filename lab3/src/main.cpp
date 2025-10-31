#include <opencv2/opencv.hpp>

int main() {
    // Открываем первую камеру по умолчанию (id = 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: не удалось открыть камеру" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;  // Считываем кадр из камеры
        if (frame.empty()) break;

        cv::imshow("Камера", frame);  // Показываем кадр

        // Выход из цикла по нажатию 'q' или ESC
        char c = (char)cv::waitKey(30);
        if (c == 27 || c == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
