#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

int selectedFilter = -1;
int frameCount = 0;
cv::Mat currentFrame;

double inputTime = 0, processTime = 0, displayTime = 0;
double inputPercent = 0, processPercent = 0;

cv::Mat applyFilter(const cv::Mat& frame, int type)
{
    cv::Mat result = frame.clone();

    switch (type)
    {
    case 0: // Original
        return result;

    case 1: // Edges
        {
            cv::cvtColor(result, result, cv::COLOR_BGR2GRAY);
            cv::Canny(result, result, 100, 200);
            cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
            return result;
        }

    case 2: // Sepia
        {
            for (int y = 0; y < result.rows; y++)
            {
                for (int x = 0; x < result.cols; x++)
                {
                    cv::Vec3b pixel = result.at<cv::Vec3b>(y, x);
                    int b = pixel[0], g = pixel[1], r = pixel[2];
                    int gray = (int)(0.2126 * r + 0.7152 * g + 0.0722 * b);

                    // Изменение цвета по синусоиде
                    float t = frameCount * 0.1f;
                    float redMult = 1.2f + 0.5f * sin(t);
                    float greenMult = 1.0f + 0.2f * sin(t + 2.09f);
                    float blueMult = 0.8f + 0.2f * sin(t + 4.19f);

                    result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        cv::saturate_cast<uchar>(gray * blueMult),
                        cv::saturate_cast<uchar>(gray * greenMult),
                        cv::saturate_cast<uchar>(gray * redMult)
                    );
                }
            }

            frameCount++;
            return result;
        }

    default: return result;
    }
}

void mouseCallback(int event, int x, int, int, void*)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        int col = x / (currentFrame.cols / 3);
        if (col < 3) selectedFilter = col;
    }
    if (event == cv::EVENT_RBUTTONDOWN)
    {
        selectedFilter = -1;
    }
}

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1200);

    std::vector<std::string> names = {"Original", "Edges", "Sepia"};

    cv::namedWindow("Filters", cv::WINDOW_NORMAL);
    cv::resizeWindow("Filters", 1920, 1200);
    cv::setMouseCallback("Filters", mouseCallback);

    cv::Mat frame, flipped;

    double fps = 0;
    time_t prevTime = time(nullptr);

    while (true)
    {
        auto t_input_start = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto t_input_end = std::chrono::high_resolution_clock::now();

        if (frame.empty())
        {
            break;
        }

        auto t_proc_start = std::chrono::high_resolution_clock::now();

        // Flipping increases time x3.8 times
        cv::flip(frame, flipped, 1);

        // Resize does the same
        if (flipped.cols != 1920 || flipped.rows != 1200)
        {
            cv::resize(flipped, flipped, cv::Size(1920, 1200));
        }

        // Clear memory copying needs much system time
        currentFrame = flipped.clone();
        frameCount++;

        if (time(nullptr) - prevTime >= 1)
        {
            fps = frameCount;
            frameCount = 0;
            prevTime = time(nullptr);
        }

        auto t_proc_end = std::chrono::high_resolution_clock::now();
        inputTime = std::chrono::duration<double, std::milli>(t_input_end - t_input_start).count();
        processTime = std::chrono::duration<double, std::milli>(t_proc_end - t_proc_start).count();
        double totalTime = inputTime + processTime;
        if (totalTime > 0)
        {
            inputPercent = (inputTime / totalTime) * 100;
            processPercent = (processTime / totalTime) * 100;
        }

        if (selectedFilter >= 0 && selectedFilter < 3)
        {
            cv::Mat filtered = applyFilter(flipped, selectedFilter);
            cv::putText(filtered, "FPS: " + std::to_string((int)fps),
                        cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5,
                        cv::Scalar(0, 255, 0), 3);
            cv::putText(filtered, "Input: " + std::to_string((int)inputTime) + "ms (" + std::to_string((int)inputPercent) + "%)",
                        cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(255, 0, 0), 2);
            cv::putText(filtered, "Process: " + std::to_string((int)processTime) + "ms (" + std::to_string((int)processPercent) + "%)",
                        cv::Point(20, 140), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 255, 255), 2);
            cv::putText(filtered, names[selectedFilter],
                        cv::Point(20, 200), cv::FONT_HERSHEY_SIMPLEX, 2,
                        cv::Scalar(0, 255, 0), 3);
            cv::putText(filtered, "Right-click to return",
                        cv::Point(20, filtered.rows - 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);

            auto t_display_start = std::chrono::high_resolution_clock::now();
            cv::imshow("Filters", filtered);
            auto t_display_end = std::chrono::high_resolution_clock::now();
            displayTime = std::chrono::duration<double, std::milli>(t_display_end - t_display_start).count();
        }
        else
        {
            cv::Mat grid = flipped.clone();
            int width = grid.cols / 3;

            // Vertical lines
            for (int i = 1; i < 3; i++)
            {
                cv::line(grid, cv::Point(i * width, 0),
                         cv::Point(i * width, grid.rows),
                         cv::Scalar(255, 255, 255), 3);
            }

            // Apply filters to each column
            for (int i = 0; i < 3; i++)
            {
                int x1 = i * width;
                cv::Mat cell = grid(cv::Rect(x1, 0, width, grid.rows)).clone();
                applyFilter(cell, i).copyTo(grid(cv::Rect(x1, 0, width, grid.rows)));
                cv::putText(grid, names[i], cv::Point(x1 + 20, 60),
                            cv::FONT_HERSHEY_SIMPLEX, 1.5,
                            cv::Scalar(0, 255, 0), 3);
            }

            cv::putText(grid, "FPS: " + std::to_string((int)fps),
                        cv::Point(20, grid.rows - 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 3);
            cv::putText(grid, "Input: " + std::to_string((int)inputTime) + "ms (" + std::to_string((int)inputPercent) + "%)",
                        cv::Point(20, grid.rows - 70),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);
            cv::putText(grid, "Process: " + std::to_string((int)processTime) + "ms (" + std::to_string((int)processPercent) + "%)",
                        cv::Point(20, grid.rows - 110),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

            auto t_display_start = std::chrono::high_resolution_clock::now();
            cv::imshow("Filters", grid);
            auto t_display_end = std::chrono::high_resolution_clock::now();
            displayTime = std::chrono::duration<double, std::milli>(t_display_end - t_display_start).count();
        }

        if ((char)cv::waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
