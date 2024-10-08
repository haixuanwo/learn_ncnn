#ifndef hand_COMMON_H
#define hand_COMMON_H

#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/benchmark.h>
#include "HandAlg.h"

typedef struct
{
	ncnn::Net* hand_det_net;
}hand_det_handle_t;

typedef struct
{
	ncnn::Net* hand_rec_net;
}hand_rec_handle_t;

typedef struct
{
	ncnn::Net* hand_landms_net;
}hand_landms_handle_t;

typedef struct
{
    hand_det_handle_t* hand_det;
    hand_rec_handle_t* hand_rec;
	hand_landms_handle_t* hand_landms;
} FaceHandle;

class Detector
{
	public:
		Detector();
		Detector(bool retinaface = false);
		inline void Release();
		void nms(std::vector<bbox> &input_boxes, float NMS_THRESH);
		void Detect(ncnn::Net *Net_det, ncnn::Mat& bgr, std::vector<bbox>& boxes, int cpuCount, float scale);
		void create_anchor(std::vector<box> &anchor, int w, int h);
		inline void SetDefaultParams();
		static inline bool cmp(bbox a, bbox b);
		~Detector();

	public:
		float _nms;
		float _threshold;
		float _mean_val[3];
		bool _retinaface;
};

int hand_det_init(hand_det_handle_t* hand_det, const char* model_path);
int hand_det_work(hand_det_handle_t* hand_det, ncnn::Mat &image, std::vector<bbox> &boxes, int cpuCount);
int hand_rec_init(hand_rec_handle_t* hand_rec, const char* model_path);
int hand_rec_work(hand_rec_handle_t* hand_rec, ncnn::Mat &bgr, std::vector<bbox> boxes, int cpuCount, std::vector<float> &features);
int hand_landms_init(hand_landms_handle_t* hand_landms, const char* model_path);
int hand_landms_work(hand_landms_handle_t* hand_landms, ncnn::Mat &bgr, std::vector<bbox> boxes, std::vector<float> &points, std::vector<float> &angles, int cpuCount);

#endif
