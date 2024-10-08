#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include "ir20231031_ir3_slim300_epoch145_fp16.id.h"
#include "HandAlg.h"
#include "HandCommon.h"
#include "ncnn/mat.h"

inline void Detector::Release()
{

}

Detector::Detector(bool retinaface):
        _nms(0.4),
        _threshold(0.9),
        _mean_val{104.f, 117.f, 123.f},
        _retinaface(retinaface)
{
    //Init(model_param, model_bin);
}

inline bool Detector::cmp(bbox a, bbox b) {
    if (a.s > b.s)
        return true;
    return false;
}

inline void Detector::SetDefaultParams(){
    _nms = 0.4;
    _threshold = 0.9;
    _mean_val[0] = 104;
    _mean_val[1] = 117;
    _mean_val[2] = 123;
    //Net = nullptr;

}

Detector::~Detector(){
    Release();
}

void Detector::create_anchor(std::vector<box> &anchor, int w, int h)
{

    anchor.clear();
    std::vector<std::vector<int> > feature_map(4), min_sizes(4);
    float steps[] = {8, 16, 32, 64};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }


    #if 0
    std::vector<int> minsize1 = {10, 16, 24};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {32, 48};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {64, 96};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {128, 192, 256};
    min_sizes[3] = minsize4;
    #endif

#if 1
    std::vector<int> minsize1 = {8, 12, 20};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {24, 38};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {52, 76};
    min_sizes[2] = minsize3;
    std::vector<int> minsize4 = {102, 154, 206};
    min_sizes[3] = minsize4;
#endif


    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    box axil = {cx, cy, s_kx, s_ky};
                    anchor.push_back(axil);
                }
            }
        }

    }

}

void Detector::nms(std::vector<bbox> &input_boxes, float NMS_THRESH)
{
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float   h = (std::max)(float(0), yy2 - yy1 + 1);
            float   inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

void Detector::Detect(ncnn::Net *Net_det, ncnn::Mat& in, std::vector<bbox>& boxes, int cpuCount, float scale)
{
    // ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, bgr.cols, bgr.rows);
    in.substract_mean_normalize(_mean_val, 0);
    ncnn::Extractor ex = Net_det->create_extractor();
    ex.set_light_mode(true);
    // ex.set_num_threads(cpuCount);

    ex.input(ir20231031_ir3_slim300_epoch145_fp16_param_id::BLOB_input, in);
    ncnn::Mat out, out1;

//    ex.extract("loc", out);          //loc
//    ex.extract("conf", out1);        // class

    ex.extract(ir20231031_ir3_slim300_epoch145_fp16_param_id::BLOB_loc, out);          //loc
    ex.extract(ir20231031_ir3_slim300_epoch145_fp16_param_id::BLOB_conf, out1);        // class

    std::vector<box> anchor;
    create_anchor(anchor,  in.w, in.h);

    std::vector<bbox > total_box;
    float *ptr = out.channel(0);
    float *ptr1 = out1.channel(0);

    for (int i = 0; i < anchor.size(); ++i)
    {
        if (*(ptr1+1) > (_threshold))
        {
            box tmp = anchor[i];
            box tmp1;
            bbox result;

            // loc and conf
            tmp1.cx = tmp.cx + *ptr * 0.1 * tmp.sx;
            tmp1.cy = tmp.cy + *(ptr+1) * 0.1 * tmp.sy;
            tmp1.sx = tmp.sx * exp(*(ptr+2) * 0.2);
            tmp1.sy = tmp.sy * exp(*(ptr+3) * 0.2);

            result.x1 = (tmp1.cx - tmp1.sx/2) * in.w;
            if (result.x1<0)
                result.x1 = 0;
            result.y1 = (tmp1.cy - tmp1.sy/2) * in.h;
            if (result.y1<0)
                result.y1 = 0;
            result.x2 = (tmp1.cx + tmp1.sx/2) * in.w;
            if (result.x2>in.w)
                result.x2 = in.w;
            result.y2 = (tmp1.cy + tmp1.sy/2)* in.h;
            if (result.y2>in.h)
                result.y2 = in.h;
            result.s = *(ptr1 + 1);

			result.x1 /= scale;
			result.y1 /= scale;
			result.x2 /= scale;
			result.y2 /= scale;

            total_box.push_back(result);
        }
        ptr += 4;
        ptr1 += 2;
    }

    std::sort(total_box.begin(), total_box.end(), cmp);
    nms(total_box, _nms);

    for (int j = 0; j < total_box.size(); ++j)
    {
        boxes.push_back(total_box[j]);
    }
}

int hand_det_work(hand_det_handle_t* hand_det, ncnn::Mat &img, std::vector<bbox> &boxes, int cpuCount)
{
	Detector detector(false);      //// slim or RFB: false, retinaface: true

	/***********scale the image, the max_size:240  ***********/
	const int max_side = 300;      // 240
	float long_side = (std::max)(img.w, img.h);
	float scale = max_side / long_side;

    ncnn::Mat img_scale;
    resize_nearest(img, img_scale, img.w*scale, img.h*scale);

    detector.Detect(hand_det->hand_det_net, img_scale, boxes, cpuCount, scale);
    return 0;
}
