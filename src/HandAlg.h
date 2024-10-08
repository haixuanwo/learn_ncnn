#ifndef FACE_ALG_H
#define FACE_ALG_H

#include <string>
#include <stack>

#include "ncnn/mat.h"

#if defined(_MSC_VER) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#  define DECL_EXPORT __declspec(dllexport)
#  define DECL_IMPORT __declspec(dllimport)
#else
#  define DECL_EXPORT     __attribute__((visibility("default")))
#  define DECL_IMPORT     __attribute__((visibility("default")))
#endif

#ifdef _TOF_EXPORT_
#define _HAND_API_ DECL_EXPORT
#else
#define _HAND_API_ DECL_IMPORT
#endif // _FACE_API_

#ifdef __cplusplus
extern "C" {
#endif

	struct Rec_Info {
		float dis;
		int id;
	};

	struct Point{
		float _x;
		float _y;
	};

	struct bbox{
		float x1;
		float y1;
		float x2;
		float y2;
		float s;
		Point point[5];
	};

	struct box{
		float cx;
		float cy;
		float sx;
		float sy;
	};

	_HAND_API_ int initHandAlg(void **algHandle, const char *modelPath);

	_HAND_API_ int hand_det(const void *algHandle, ncnn::Mat &bgr, std::vector<bbox> &boxes);

	_HAND_API_ int hand_rec(const void *algHandle, ncnn::Mat &bgr, std::vector<bbox> boxes, std::vector<float> &features);

	_HAND_API_ int hand_landms(const void *algHandle, ncnn::Mat &bgr, std::vector<bbox> boxes, std::vector<float> &points, std::vector<float> &angles);

	_HAND_API_ int Get_CPU_Number_Windows();

    /**
     * @brief 静态匹配手势
     * @param features
     * @param rec_info
     */
    void featureMatch(std::vector<float> &features, struct Rec_Info &rec_info);

#ifdef __cplusplus
}
#endif

#endif
