#include <opencv2/opencv.hpp>
#include <vector>

int selectedFilter = -1;
cv::Mat currentFrame;

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
                    result.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        cv::saturate_cast<uchar>(gray * 0.8),
                        cv::saturate_cast<uchar>(gray),
                        cv::saturate_cast<uchar>(gray * 1.2)
                    );
                }
            }
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
    int frameCount = 0;
    double fps = 0;
    time_t prevTime = time(nullptr);

    while (true)
    {
        cap >> frame;
        if (frame.empty())
        {
            break;
        }

        cv::flip(frame, flipped, 1);
        if (flipped.cols != 1920 || flipped.rows != 1200)
        {
            cv::resize(flipped, flipped, cv::Size(1920, 1200));
        }

        currentFrame = flipped.clone();
        frameCount++;

        if (time(nullptr) - prevTime >= 1)
        {
            fps = frameCount;
            frameCount = 0;
            prevTime = time(nullptr);
        }

        if (selectedFilter >= 0 && selectedFilter < 3)
        {
            cv::Mat filtered = applyFilter(flipped, selectedFilter);
            cv::putText(filtered, "FPS: " + std::to_string((int)fps),
                        cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.5,
                        cv::Scalar(0, 255, 0), 3);
            cv::putText(filtered, names[selectedFilter],
                        cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 2,
                        cv::Scalar(0, 255, 0), 3);
            cv::putText(filtered, "Right-click to return",
                        cv::Point(20, filtered.rows - 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Filters", filtered);
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
            cv::imshow("Filters", grid);
        }

        if ((char)cv::waitKey(1) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
