/*
 * @Author: Clark
 * @Email: haixuanwoTxh@gmail.com
 * @Date: 2024-06-13 11:34:10
 * @LastEditors: Clark
 * @LastEditTime: 2024-09-23 19:17:25
 * @Description: file content
 */
#pragma once

#include <string>
#include <chrono>
#include <stdint.h>
#include <iostream>

#define IMAGE_PATH "image"
#define POINT_CLOUD_WIDTH 640
#define POINT_CLOUD_HEIGHT 480
#define POINT_CLOUD_COUNT 307200    // 640*480

enum pos{
    X,
    Y,
    Z,
};

enum color{
    R,
    G,
    B,
    A,
};

/**
 * @brief 点云点信息
 */
struct VertexInfo
{
    float pos[3] = {0.0,0.0,0.0};           // X,Y,Z
    float normal[3] = {0.0,1.0,0.0};        // 法向量
    float color[4] = {0.0, 255.0f, 0.0, 1.0}; // R,G,B,A
};

/**
 * @brief 显示模式
 */
typedef enum {
    DISPLAY_MODE_FOUR,              // 四分屏显示模式
    DISPLAY_MODE_POINT_CLOUD,       // 点云显示模式
    DISPLAY_MODE_POINT_CLOUD_RGB,   // 点云显示模式
    DISPLAY_MODE_DEPTH,             // 深度显示模式
    DISPLAY_MODE_IR,                // 灰度显示模式
    DISPLAY_MODE_RGB,               // RGB显示模式
}DisplayMode;

/**
 * @brief 是否为RGBD相机
 */
bool is_rgbd_camera();

/**
 * @brief 设置RGBD相机标志
 */
void set_rgbd_camera(bool flag);

std::string get_now_time_str();

class PerformanceTime
{
public:
    uint64_t get_now_time_ms()
    {
        auto now = std::chrono::high_resolution_clock::now();

        // 转换为时间戳（纳秒）
        auto duration = now.time_since_epoch();

        // 将纳秒转换为微妙数（1微妙 = 1000纳秒）
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

        return microseconds/1000;
    }
};


class TestFps
{
public:
    TestFps() : m_count(0)
    {
        m_start = get_now_time_ms();
    }

    void update(const std::string &info)
    {
        uint64_t now = get_now_time_ms();
        uint64_t diff = now - m_start;
        if (diff >= 3000)
        {
            std::cout << info << " fps: " << (m_count*1000)/diff << std::endl;
            m_count = 0;
            m_start = now;
        }

        m_count++;
    }

private:
    uint64_t get_now_time_ms()
    {
        auto now = std::chrono::high_resolution_clock::now();

        // 转换为时间戳（纳秒）
        auto duration = now.time_since_epoch();

        // 将纳秒转换为微妙数（1微妙 = 1000纳秒）
        auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

        return microseconds/1000;
    }

private:
    uint64_t m_count;
    uint64_t m_start;
};

class File
{
public:
    File() : m_file(nullptr)
    {

    }

    ~File()
    {
        if (m_file)
        {
            fclose(m_file);
        }
    }

    bool open(const std::string& path)
    {
        m_file = fopen(path.c_str(), "r+");
        if (m_file == nullptr)
        {
            m_file = fopen(path.c_str(), "wb");
            if (m_file == nullptr)
            {
                std::cerr << "Failed to open file: " << path << std::endl;
                return false;
            }
        }
        return true;
    }

    bool write(const void* data, size_t size)
    {
        if (fwrite(data, size, 1, m_file) != 1)
        {
            std::cerr << "Failed to write data to file" << std::endl;
            return false;
        }
        return true;
    }

    bool read(void* data, size_t size)
    {
        if (fread(data, size, 1, m_file) != 1)
        {
            std::cerr << "Failed to read data from file" << std::endl;
            return false;
        }
        return true;
    }

    void close()
    {
        fclose(m_file);
    }

    static bool save_data_to_file(const std::string& path, const void* data, size_t size)
    {
        File file;
        if (!file.open(path))
        {
            return false;
        }

        if (!file.write(data, size))
        {
            return false;
        }

        file.close();
        return true;
    }

    // static bool save_data_to_file(QImage &img, const std::string& suffix)
    // {
    //     std::string filename = IMAGE_PATH;
    //     filename += "/";
    //     filename += get_now_time_str();
    //     filename += suffix;

    //     QImageWriter writer;
    //     writer.setFileName(filename.c_str());
    //     writer.write(img);
    //     return true;
    // }
private:
    FILE* m_file;
};
