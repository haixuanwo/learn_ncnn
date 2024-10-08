/*
 * @Author: Clark
 * @Email: haixuanwoTxh@gmail.com
 * @Date: 2024-06-13 11:34:10
 * @LastEditors: Clark
 * @LastEditTime: 2024-07-03 13:44:23
 * @Description: file content
 */

#include "misc.h"
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip> // 包含处理时间的头文件
#include <ctime>

static bool isRgbdCamera = false;

/**
 * @brief 是否为RGBD相机
 */
bool is_rgbd_camera()
{
    return isRgbdCamera;
}

/**
 * @brief 设置RGBD相机标志
 */
void set_rgbd_camera(bool flag)
{
    isRgbdCamera = flag;
}

std::string get_now_time_str()
{
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();
    // 转换为time_t类型以便转换为tm结构体
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    // 转换为本地时间
    std::tm* now_tm = std::localtime(&now_c);
    // 使用put_time来格式化时间
    std::stringstream ss;
    ss << std::put_time(now_tm, "%Y-%m-%d_%H_%M_%S");

    return std::string(ss.str());
}
