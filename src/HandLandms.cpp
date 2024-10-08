#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <ncnn/gpu.h>
#include "ir231110_squeezenet1_1-size-256-loss-wing_loss-model_epoch-1900_fp16.id.h"
#include "HandAlg.h"
#include "HandCommon.h"

float cal_angle(std::vector<float> points, int a, int b, int c)
{
    float p[2] = {points[b*2+0] - points[a*2+0], points[b*2+1] - points[a*2+1]};
    float q[2] = {points[b*2+0] - points[c*2+0], points[b*2+1] - points[c*2+1]};
    float p_ = sqrt(pow(points[b*2+0] - points[a*2+0], 2) + pow(points[b*2+1] - points[a*2+1], 2));
    float q_ = sqrt(pow(points[b*2+0] - points[c*2+0], 2) + pow(points[b*2+1] - points[c*2+1], 2));
    float theta = ((points[b*2+0] - points[a*2+0])*(points[b*2+0] - points[c*2+0]) + (points[b*2+1] - points[a*2+1])*(points[b*2+1] - points[c*2+1]))/(p_*q_);
    return acos(theta)*180/3.14159;

}

int palm_angle(std::vector<float> points, std::vector<float> &angles)
{
    float angle = cal_angle(points, 1, 2, 4);
    angles.push_back(angle);
    angles.push_back(cal_angle(points, 5, 6, 8));
    angles.push_back(cal_angle(points, 9, 10, 12));
    angles.push_back(cal_angle(points, 13, 14, 16));
    angles.push_back(cal_angle(points, 17, 18, 20));
    return 0;
}

int hand_landms_work(hand_landms_handle_t* hand_landms, ncnn::Mat &bgr, std::vector<bbox> boxes, std::vector<float> &points, std::vector<float> &angles, int cpuCount)
{
    // crop img
    int w, h, w_, x1, y1, x2, y2;
    w = boxes[0].x2 - boxes[0].x1;
    h = boxes[0].y2 - boxes[0].y1;
//    w_ = max(w, h) * 1.1;
    w_ = std::max(w, h) * 1.1;
    x1 = (std::max)(int(boxes[0].x1 + w/2 - w_/2), 0);
    x2 = (std::min)(int(boxes[0].x1 + w/2 + w_/2), bgr.w);
    y1 = (std::max)(int(boxes[0].y1 + h/2 - w_/2), 0);
    y2 = (std::min)(int(boxes[0].y1 + h/2 + w_/2), bgr.h);

    ncnn::Mat cutBgr;
    copy_cut_border(bgr, cutBgr, y1, y1, x1, x1);
    ncnn::Mat in;
    resize_nearest(cutBgr, in, 256, 256);

    // inference
    const float mean_vals[3] = {127.0f, 127.0f, 127.0f};
    const float norm_vals[3] = {1.0 / (255.0f), 1.0 / (255.0f), 1.0 / (255.0f)};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = hand_landms->hand_landms_net->create_extractor();
    ex.set_light_mode(true);
    // ex.set_num_threads(cpuCount);

    //double start_rec = ncnn::get_current_time();
    ex.input(ir231110_squeezenet1_1_size_256_loss_wing_loss_model_epoch_1900_fp16_param_id::BLOB_input0, in);
    ncnn::Mat out;
    ex.extract(ir231110_squeezenet1_1_size_256_loss_wing_loss_model_epoch_1900_fp16_param_id::BLOB_output0, out);
    //double end_rec = ncnn::get_current_time();
    //std::cout<<"Time--hand_landms: "<<double(end_rec-start_rec)<<std::endl;    // 7 ms

    out = out.reshape(out.w, out.h, out.c);     //(w,h,c) = (512, 1, 1)
    for (int i=0; i<out.w; ++i)
    {
        if (i%2==0)
        {
            points.push_back(out[i]*(x2-x1) + x1);     // 在原图bgr上的 x
        }
        else
        {
            points.push_back(out[i]*(y2-y1) + y1);    //在原图bgr上的 y
        }
    }
    palm_angle(points, angles);

	return 0;
}
