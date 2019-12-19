#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include<stdlib.h>
using namespace cv;


int add_labels(Mat src, std::vector<int>& labels)
{
    int A, B, C;
    int lable = 1;
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; ++x)
        {
            Vec3b bgr = src.at<Vec3b>(y, x);
            labels[x + y * src.cols] = ((bgr[0] + bgr[1] + bgr[2]) == 0);
        }
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            int kn = x - 1;
            if (kn <= 0)
            {
                kn = 1;
                B = 0;
            }
            else
                B = labels[kn + y * src.cols];
            int km = y - 1;
            if (km <= 0)
            {
                km = 1;
                C = 0;
            }
            else
                C = labels[x + km * src.cols];
            A = labels[x + y * src.cols];
            if (A == 0)
                continue;
            if (B == 0 && C == 0)
            {
                lable++;
                labels[x + y * src.cols] = lable;
            }
            else if (B != 0 && C == 0)
                labels[x + y * src.cols] = B;
            else if (B == 0 && C != 0)
                labels[x + y * src.cols] = C;
            else if(B == C)
                labels[x + y * src.cols] = B;
            else
            {
                labels[x + y * src.cols] = B;
                for (int y = 0; y < src.rows; y++)
                    for (int x = 0; x < src.cols; ++x)
                        if (labels[x + y * src.cols] == C)
                            labels[x + y * src.cols] = B;
            }
        }
    return lable;
}

std::vector<double> get_histogramme(Mat src)
{
    std::vector<double> hisogramme(766);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            Vec3b bgr = src.at<Vec3b>(y, x);
            hisogramme[bgr[0] + bgr[1] + bgr[2]]++;
        }
    int sz = src.rows * src.cols;
    for (auto it = hisogramme.begin(); it != hisogramme.end(); it++)
        *it /= sz;
    return hisogramme;
}

double get_relative_frequency(int k, std::vector<double> vec)
{
    double res = 0;
    for (int i = 0; i < k; i++)
        res += vec[i];
    return res;
}

double get_mathematical_expectation_1(int k, std::vector<double> vec)
{
    double res = 0;
    double u = get_relative_frequency(k, vec);
    for (int i = 0; i < k; i++)
        res += i * vec[i] / u;
    return res;
}

double get_mathematical_expectation_2(int k, std::vector<double> vec)
{
    double res = 0;
    double u = 1 - get_relative_frequency(k, vec);
    for (int i = k; i < vec.size(); i++)
        res += i * vec[i] / u;
    return res;
}

int Otsu_method_threshold(Mat img)
{
    int max = 0;
    double dispertion;
    double max_dispertion = 0;
    std::vector<double> hist = get_histogramme(img);
    for (int i = 0; i < 756; i++)
    {
        double frequency = get_relative_frequency(i, hist);
        double delta_mexpectation = get_mathematical_expectation_2(i, hist) - get_mathematical_expectation_1(i, hist);
        dispertion = frequency * (1 - frequency) * delta_mexpectation * delta_mexpectation;
        if (dispertion > max_dispertion)
        {
            max_dispertion = dispertion;
            max = i;
        }
    }
    return max;
}

Mat Otsu_method_binarization(Mat src)
{
    Mat res;
    src.copyTo(res);
    int threshold = Otsu_method_threshold(src);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            Vec3b bgr = src.at<Vec3b>(y, x);
            float val = bgr[0] + bgr[1] + bgr[2];
            if (val > threshold)
            {
                res.at<Vec3b>(y, x)[0] = 255;
                res.at<Vec3b>(y, x)[1] = 255;
                res.at<Vec3b>(y, x)[2] = 255;
            }
            else
            {
                res.at<Vec3b>(y, x)[0] = 0;
                res.at<Vec3b>(y, x)[1] = 0;
                res.at<Vec3b>(y, x)[2] = 0;
            }
        }
    return res;
}

Mat get_areas(Mat src)
{
    Mat res;
    src.copyTo(res);
    std::vector<int> labels(src.cols * src.rows);
    std::list<int> unic;
    add_labels(src, labels);
    for (auto it = labels.begin(); it != labels.end(); it++)
    {
        if (std::find(unic.begin(), unic.end(), *it) == unic.end())
            unic.push_back(*it);
    }
    std::cout << unic.size() - 1 << std::endl;
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            res.at<Vec3b>(y, x)[0] = 255 - (labels[x + y * src.cols] * 10) % 255;
            res.at<Vec3b>(y, x)[1] = 255 - (labels[x + y * src.cols] * 30) % 255;
            res.at<Vec3b>(y, x)[2] = 255 - (labels[x + y * src.cols] * 20) % 255;
        }
    return res;
}

Mat PhotoshopGray(Mat src)
{
    Mat res;
    src.copyTo(res);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            Vec3b bgr = src.at<Vec3b>(y, x);
            float gray = bgr[0] * 0.11f + bgr[1] * 0.59f + bgr[2] * 0.3f;
            res.at<Vec3b>(y, x)[0] = gray;
            res.at<Vec3b>(y, x)[1] = gray;
            res.at<Vec3b>(y, x)[2] = gray;
        }
    return res;
}

int main(int argc, char** argv)
{
    srand(time(NULL));
    Mat img = imread("C:\\Users\\dimen\\Pictures\\nums2.jpg");
    Mat grey = PhotoshopGray(img);
    Mat Otsu_img = Otsu_method_binarization(grey);
    Mat areas = get_areas(Otsu_img);
    imshow("Original", img);
    imshow("Otsu method", Otsu_img);
    imshow("test", areas);
    waitKey();
    return 0;
}
