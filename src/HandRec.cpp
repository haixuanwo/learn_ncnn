/*
 * @Author: Clark
 * @Email: haixuanwoTxh@gmail.com
 * @Date: 2024-10-08 16:29:33
 * @LastEditors: Clark
 * @LastEditTime: 2024-10-08 16:37:42
 * @Description: file content
 */
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <ncnn/gpu.h>
#include "ir20230815_0810mv3_shuff_0.5m_backbone_fintune_fp16.id.h"

#include "HandAlg.h"
#include "HandCommon.h"
#include "ncnn/mat.h"

#if 0
void pretty_print(const ncnn::Mat& m)
{
    // 打印 ncnn::Mat 数值
    for (int q=0; q<m.c; q++)
    {
        //cout<<"111111111--channel: "<<q<<endl;
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
    }
}
#endif

int hand_rec_work(hand_rec_handle_t* hand_rec, ncnn::Mat &bgr, std::vector<bbox> boxes, int cpuCount, std::vector<float> &features)
{
    // 裁剪图片
    ncnn::Mat cutBgr;
    ncnn::copy_cut_border(bgr, cutBgr, boxes[0].y1, boxes[0].y1, boxes[0].x1, boxes[0].x1);

    int diffLen = (cutBgr.h - cutBgr.w)/2;
    ncnn::Mat blackBgr;
    if (diffLen > 0)
    {
        ncnn::copy_make_border(cutBgr, blackBgr, 0, 0, diffLen, diffLen, ncnn::BORDER_CONSTANT, 0);
    }
    else
    {
        diffLen = -diffLen;
        ncnn::copy_make_border(cutBgr, blackBgr, diffLen, diffLen, 0, 0, ncnn::BORDER_CONSTANT, 0);
    }

    ncnn::Mat in;
    resize_nearest(blackBgr, in, 80, 80);

    // inference
    const float mean_vals[3] = {255*0.343f, 255*0.343f, 255*0.343f};
    const float norm_vals[3] = {1.0 / (255*0.204f), 1.0 / (255*0.204f), 1.0 / (255*0.204f)};
    in.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = hand_rec->hand_rec_net->create_extractor();
    ex.set_light_mode(true);
    // ex.set_num_threads(cpuCount);

    ex.input(ir20230815_0810mv3_shuff_0_5m_backbone_fintune_fp16_param_id::BLOB_input0, in);
    ncnn::Mat out;
    ex.extract(ir20230815_0810mv3_shuff_0_5m_backbone_fintune_fp16_param_id::BLOB_output0, out);
    out = out.reshape(out.w, out.h, out.c);     //(w,h,c) = (512, 1, 1)

	// 获取 L2归一化后的 features
	features.clear();
	float fenmu = 0.f;
	float fenmu_sq = 0.f;
	float fenzi = 0.f;
	for (int i = 0; i < out.w; ++i)
	{
		fenmu += std::pow(out[i], 2);
	}
	fenmu_sq = std::sqrt(fenmu + 1e-9);
	for (int i = 0; i < out.w; ++i)
	{
		features.push_back(out[i] / fenmu_sq);
	}

	return 0;
}
