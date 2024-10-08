/*
 * @Author: Clark
 * @Email: haixuanwoTxh@gmail.com
 * @Date: 2024-09-20 16:02:06
 * @LastEditors: Clark
 * @LastEditTime: 2024-09-24 15:37:13
 * @Description: file content
 */

#include <iostream>
#include "HandAlg.h"
#include <string>
#include "misc.h"
#include "ncnn/mat.h"
#include "ncnn/simpleocv.h"

#define WIDTH 640
#define HEIGHT 480
#define GRAY_SIZE 307200

static void* m_pHandAlg_Handle = nullptr;

bool gesture_recognition(const std::string &path)
{
    std::vector<uint8_t> data(GRAY_SIZE);
    File file;

    if (!file.open(path))
    {
        printf("open file failed\n");
        return false;
    }

    if (!file.read(data.data(), GRAY_SIZE))
    {
        printf("read file failed\n");
        return false;
    }

    // 手掌检测
    // uint8_t data[480 * 640] = {0};
    // cv::Mat nirImg = cv::Mat(480, 640, CV_8UC1, data.data()).clone();
    // cv::cvtColor(nirImg, nirImg, cv::COLOR_GRAY2BGR);
    // cv::Mat src_det = nirImg.clone();
    // cv::Mat src_rec = nirImg.clone();
    // cv::Mat src_landms = nirImg.clone();

    std::vector<bbox> boxes;     //// 输出坐标,恢复至原图上的
    boxes.clear();

    ncnn::Mat bgr = ncnn::Mat::from_pixels_resize(data.data(), ncnn::Mat::PIXEL_GRAY2BGR, WIDTH, HEIGHT, WIDTH, HEIGHT);
    ncnn::Mat bgr1 = bgr.clone();
    ncnn::Mat bgr2 = bgr.clone();

    // 手掌检测
    hand_det(m_pHandAlg_Handle, bgr, boxes);
    if (boxes.size() <= 0)
    {
        printf("no hand detected\n");
        return false;
    }
    printf("hand boxes num: %ld\n", boxes.size());

    // 手掌关键点检测
    std::vector<float> points;
    std::vector<float> angles;
    hand_landms(m_pHandAlg_Handle, bgr1, boxes, points, angles);
    std::cout << "hand boxes num: " << boxes.size() << " points num: " << points.size() << " angles num: " << angles.size() << std::endl;

    // 静态手势分类
    Rec_Info rec_info;
    std::vector<float> features;
    hand_rec(m_pHandAlg_Handle, bgr2, boxes, features);
    featureMatch(features, rec_info);
    std::cout << "JH --- rec_info.id: " << rec_info.id << " rec_info.dis: " << rec_info.dis << std::endl;

    return true;
}

int main()
{
    std::vector<std::string> handsFile = {"data/nohand.raw", "data/stop.raw", "data/v.raw"};
    initHandAlg(&m_pHandAlg_Handle, "model");
    if (!m_pHandAlg_Handle)
    {
        std::cout << "JH --- init hand alg failed" << std::endl;
    }
    std::cout << "JH --- init hand alg ok" << std::endl;

    while (1)
    {
        for (auto &file : handsFile)
        {
            gesture_recognition(file);
        }
    }

    return 0;
}
