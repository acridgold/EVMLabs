#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

using namespace cv;

int selectedFilter = -1;
int frameCount = 0;
Mat currentFrame;

double inputTime = 0, processTime = 0, displayTime = 0;
double inputPercent = 0, processPercent = 0, displayPercent = 0;

Mat applyFilter(const Mat& frame, int type)
{
    Mat result = frame.clone();

    switch (type)
    {
    case 0: // Original
        return result;

    case 1: // Edges
        {
            cvtColor(result, result, COLOR_BGR2GRAY);
            Canny(result, result, 100, 200);
            cvtColor(result, result, COLOR_GRAY2BGR);
            return result;
        }

    case 2: // Sepia
        {
            for (int y = 0; y < result.rows; y++)
            {
                for (int x = 0; x < result.cols; x++)
                {
                    Vec3b pixel = result.at<Vec3b>(y, x);
                    int b = pixel[0], g = pixel[1], r = pixel[2];
                    int gray = (int)(0.2126 * r + 0.7152 * g + 0.0722 * b);

                    // Изменение цвета по синусоиде
                    float t = frameCount * 0.1f;
                    float redMult = 1.2f + 0.5f * sin(t);
                    float greenMult = 1.0f + 0.2f * sin(t + 2.09f);
                    float blueMult = 0.8f + 0.2f * sin(t + 4.19f);

                    result.at<Vec3b>(y, x) = Vec3b(
                        saturate_cast<uchar>(gray * blueMult),
                        saturate_cast<uchar>(gray * greenMult),
                        saturate_cast<uchar>(gray * redMult)
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
    if (event == EVENT_LBUTTONDOWN)
    {
        int col = x / (currentFrame.cols / 3);
        if (col < 3) selectedFilter = col;
    }
    if (event == EVENT_RBUTTONDOWN)
    {
        selectedFilter = -1;
    }
}

int main()
{
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CAP_PROP_FRAME_HEIGHT, 1200);

    std::vector<std::string> names = {"Original", "Edges", "Sepia"};

    namedWindow("Filters", WINDOW_NORMAL);
    resizeWindow("Filters", 1920, 1200);
    setMouseCallback("Filters", mouseCallback);

    Mat frame, flipped;

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
        flip(frame, flipped, 1);

        // Resize does the same
        resize(flipped, flipped, Size(1920, 1200));

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

        if (selectedFilter >= 0 && selectedFilter < 3)
        {
            Mat filtered = applyFilter(flipped, selectedFilter);

            auto t_display_start = std::chrono::high_resolution_clock::now();

            putText(filtered, "FPS: " + std::to_string((int)fps),
                        Point(20, 50), FONT_HERSHEY_SIMPLEX, 1.5,
                        Scalar(0, 255, 0), 3);
            putText(filtered, "Input: " + std::to_string((int)inputTime) + "ms (" + std::to_string((int)inputPercent) + "%)",
                        Point(20, 100), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(255, 0, 0), 2);
            putText(filtered, "Process: " + std::to_string((int)processTime) + "ms (" + std::to_string((int)processPercent) + "%)",
                        Point(20, 140), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(0, 255, 255), 2);
            putText(filtered, "Display: " + std::to_string((int)displayTime) + "ms (" + std::to_string((int)displayPercent) + "%)",
                        Point(20, 180), FONT_HERSHEY_SIMPLEX, 0.8,
                        Scalar(255, 255, 0), 2);
            putText(filtered, names[selectedFilter],
                        Point(20, 240), FONT_HERSHEY_SIMPLEX, 2,
                        Scalar(0, 255, 0), 3);
            putText(filtered, "Right-click to return",
                        Point(20, filtered.rows - 30),
                        FONT_HERSHEY_SIMPLEX, 1.2, Scalar(0, 255, 0), 2);

            imshow("Filters", filtered);
            auto t_display_end = std::chrono::high_resolution_clock::now();
            displayTime = std::chrono::duration<double, std::milli>(t_display_end - t_display_start).count();
        }
        else
        {
            Mat grid = flipped.clone();
            int width = grid.cols / 3;

            // Vertical lines
            for (int i = 1; i < 3; i++)
            {
                line(grid, Point(i * width, 0),
                         Point(i * width, grid.rows),
                         Scalar(255, 255, 255), 3);
            }

            // Apply filters to each column
            for (int i = 0; i < 3; i++)
            {
                int x1 = i * width;
                Mat cell = grid(Rect(x1, 0, width, grid.rows)).clone();
                applyFilter(cell, i).copyTo(grid(Rect(x1, 0, width, grid.rows)));
                putText(grid, names[i], Point(x1 + 20, 60),
                            FONT_HERSHEY_SIMPLEX, 1.5,
                            Scalar(0, 255, 0), 3);
            }

            auto t_display_start = std::chrono::high_resolution_clock::now();

            putText(grid, "FPS: " + std::to_string((int)fps),
                        Point(20, grid.rows - 30),
                        FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 0), 3);
            putText(grid, "Input: " + std::to_string((int)inputTime) + "ms (" + std::to_string((int)inputPercent) + "%)",
                        Point(20, grid.rows - 70),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
            putText(grid, "Process: " + std::to_string((int)processTime) + "ms (" + std::to_string((int)processPercent) + "%)",
                        Point(20, grid.rows - 110),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 255), 2);
            putText(grid, "Display: " + std::to_string((int)displayTime) + "ms (" + std::to_string((int)displayPercent) + "%)",
                        Point(20, grid.rows - 150),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 0), 2);

            imshow("Filters", grid);
            auto t_display_end = std::chrono::high_resolution_clock::now();
            displayTime = std::chrono::duration<double, std::milli>(t_display_end - t_display_start).count();
        }

        // Пересчёт процентов с учётом displayTime
        double totalTime = inputTime + processTime + displayTime;
        if (totalTime > 0)
        {
            inputPercent = (inputTime / totalTime) * 100;
            processPercent = (processTime / totalTime) * 100;
            displayPercent = (displayTime / totalTime) * 100;
        }

        if ((char)waitKey(1) == 27) {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
