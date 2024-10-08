#include <sys/types.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <algorithm>
#include <iostream>
#include <HandAlg.h>
#include <HandCommon.h>
#include <fstream>
#include <sstream>

#include "ir20231031_ir3_slim300_epoch145_fp16.id.h"
#include "ir20231031_ir3_slim300_epoch145_fp16.mem.h"
#include "ir20230815_0810mv3_shuff_0.5m_backbone_fintune_fp16.id.h"
#include "ir20230815_0810mv3_shuff_0.5m_backbone_fintune_fp16.mem.h"
#include "ir231110_squeezenet1_1-size-256-loss-wing_loss-model_epoch-1900_fp16.id.h"
#include "ir231110_squeezenet1_1-size-256-loss-wing_loss-model_epoch-1900_fp16.mem.h"

static int cpuCount = 1;

int hand_det_init(hand_det_handle_t* hand_det, const char* model_path)
{
	std::string strpath(model_path);
	if (strpath.empty())
	{
		return -100;
	}

	hand_det->hand_det_net = new ncnn::Net();          //创建net

    #if 0
	if (0 != hand_det->hand_det_net->load_param_bin((strpath + "/gesture_det/20231031_ir3_slim300_epoch145_fp16.param").c_str()))   //读取模型
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);

	if (0 != hand_det->hand_det_net->load_model((strpath + "/gesture_det/20231031_ir3_slim300_epoch145_fp16.bin").c_str()))
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
    #else

    if (0 >= hand_det->hand_det_net->load_param(ir20231031_ir3_slim300_epoch145_fp16_param_bin))   //读取模型
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);

	if (0 >= hand_det->hand_det_net->load_model(ir20231031_ir3_slim300_epoch145_fp16_bin))
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);

    #endif

	return 0;
}

int hand_rec_init(hand_rec_handle_t* hand_rec, const char* model_path)
{
	std::string strpath(model_path);
	if (strpath.empty())
	{
		return -100;
	}

	hand_rec->hand_rec_net = new ncnn::Net();
    #if 0
	if (0 != hand_rec->hand_rec_net->load_param_bin((strpath + "/gesture_cls/20230815_0810mv3_shuff_0.5m_backbone_fintune_fp16.param").c_str()))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}

	if (0 != hand_rec->hand_rec_net->load_model((strpath + "/gesture_cls/20230815_0810mv3_shuff_0.5m_backbone_fintune_fp16.bin").c_str()))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}
    #else

    if (0 >= hand_rec->hand_rec_net->load_param(ir20230815_0810mv3_shuff_0_5m_backbone_fintune_fp16_param_bin))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}

	if (0 >= hand_rec->hand_rec_net->load_model(ir20230815_0810mv3_shuff_0_5m_backbone_fintune_fp16_bin))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}
    #endif

	return 0;
}

int hand_landms_init(hand_landms_handle_t* hand_landms, const char* model_path)
{
	std::string strpath(model_path);
	if (strpath.empty())
	{
		return -100;
	}

	hand_landms->hand_landms_net = new ncnn::Net();
    #if 0
	if (0 != hand_landms->hand_landms_net->load_param_bin((strpath + "/gesture_landms/ir231110_squeezenet1_1-size-256-loss-wing_loss-model_epoch-1900_fp16.param").c_str()))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}

	if (0 != hand_landms->hand_landms_net->load_model((strpath + "/gesture_landms/ir231110_squeezenet1_1-size-256-loss-wing_loss-model_epoch-1900_fp16.bin").c_str()))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}
    #else

    if (0 >= hand_landms->hand_landms_net->load_param(ir231110_squeezenet1_1_size_256_loss_wing_loss_model_epoch_1900_fp16_param_bin))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}

	if (0 >= hand_landms->hand_landms_net->load_model(ir231110_squeezenet1_1_size_256_loss_wing_loss_model_epoch_1900_fp16_bin))
	{
		printf("load_param failed, %s, %d\n", __FILE__, __LINE__);
	}
    #endif
	return 0;
}

int initHandAlg(void **algHandle, const char *modelPath)
{
    FaceHandle *hand_handle = (FaceHandle *)malloc(sizeof(FaceHandle));
 	if(NULL == hand_handle)
	{
		printf("*****************888%s, %d, error:%s\n", __func__, __LINE__, strerror(errno));
		return -99;
	}

    hand_handle->hand_det = (hand_det_handle_t *)malloc(sizeof(hand_det_handle_t));
    hand_handle->hand_rec = (hand_rec_handle_t *)malloc(sizeof(hand_rec_handle_t));
	hand_handle->hand_landms = (hand_landms_handle_t *)malloc(sizeof(hand_landms_handle_t));

    hand_det_init(hand_handle->hand_det, modelPath);
    hand_rec_init(hand_handle->hand_rec, modelPath);
	hand_landms_init(hand_handle->hand_landms, modelPath);
	std::cout << "HandAlg Init Successfuly!!!" << std::endl;

    *algHandle = hand_handle;
    return 0;
}

int hand_det(const void *algHandle, ncnn::Mat &bgr, std::vector<bbox> &boxes)
{
    if(!algHandle)
    {
        //printf("algHandle is empty\n");
        return -98;
    }
    FaceHandle *hand_handle = (FaceHandle *)algHandle;
    hand_det_work(hand_handle->hand_det, bgr, boxes, cpuCount);
    return 0;
}

int hand_rec(const void *algHandle, ncnn::Mat &bgr, std::vector<bbox> boxes, std::vector<float> &features)
{
    if(!algHandle)
    {
        //printf("algHandle is empty\n");
        return -98;
    }
    FaceHandle *hand_handle = (FaceHandle *)algHandle;
    hand_rec_work(hand_handle->hand_rec, bgr, boxes, cpuCount, features);

    return 0;
}


int hand_landms(const void *algHandle, ncnn::Mat &bgr, std::vector<bbox> boxes, std::vector<float> &points, std::vector<float> &angles)
{
    if(!algHandle || boxes.size() <= 0)
    {
        //printf("algHandle is empty\n");
        return -98;
    }
    FaceHandle *hand_handle = (FaceHandle *)algHandle;

    hand_landms_work(hand_handle->hand_landms, bgr, boxes, points, angles, cpuCount);

    return 0;
}

#include "misc.h"
#include "gesturebank.h"

/**
 * @brief 静态匹配手势
 * @param features
 * @param rec_info
 */
void featureMatch(std::vector<float> &features, struct Rec_Info &rec_info)
{
	std::vector<float> distance;

    static std::vector<std::vector<float>> m_pHandGestureBank;
    if (m_pHandGestureBank.empty())
    {
        std::vector<float> feature;
        for (auto &gesture : gesturebank)
        {
            feature.clear();
            for (auto &value : gesture)
            {
                feature.push_back(value);
            }
            m_pHandGestureBank.push_back(feature);
        }

        #if 0
        char buf[128] = {0};
        File gestureFile;
        gestureFile.open("gesturebank.h");

        // 加载手势识别特征
        std::string handGestureBankPath = "data/gesturebank.txt";
        std::fstream file(handGestureBankPath);
        std::string line;
        if (!file.is_open())
        {
            std::cout<<"Can't open the gesturebank !!!"<<std::endl;
        }

        snprintf(buf, sizeof(buf), "gesturebank[][256] = {", NULL);
        gestureFile.write(buf, strlen(buf));

        while (getline(file, line))
        {
            std::stringstream ss(line);
            std::vector<float> handgesture;
            handgesture.clear();
            float x;

            snprintf(buf, sizeof(buf), "{", NULL);
            gestureFile.write(buf, strlen(buf));
            for (int i=0; i<256; i++)
            {
                ss >> x;
                handgesture.push_back(x);

                if (i == 255)
                {
                    snprintf(buf, sizeof(buf), "%.8f", x);
                }
                else
                {
                    snprintf(buf, sizeof(buf), "%.8f, ", x);
                }
                gestureFile.write(buf, strlen(buf));
            }
            snprintf(buf, sizeof(buf), "},\n", NULL);
            gestureFile.write(buf, strlen(buf));

            m_pHandGestureBank.push_back(handgesture);
        }
        file.close();

        snprintf(buf, sizeof(buf), "};\n", NULL);
        gestureFile.write(buf, strlen(buf));
        gestureFile.close();
        #endif
        std::cout << "JH --- load exist hand feature" << std::endl;
    }

	// 计算当前 feature 与每个底库的距离
	for (int i = 0; i < m_pHandGestureBank.size(); ++i)
	{
		float dis = 0.0;
		for (int j = 0; j < m_pHandGestureBank[i].size(); ++j)
		{
			dis += pow(m_pHandGestureBank[i][j] - features[j], 2);
		}
		distance.push_back(sqrt(dis) + 1e-9);           // distance_sq:  face开方距离
	}

	// 找 face 距离最近的下标
	int idx = 0;
	float Min = distance[0];
	for (int i = 1; i < distance.size(); ++i)
	{
		if (distance[i] < Min)
		{
			Min = distance[i];
			idx = i;
		}
	}

	rec_info.dis = distance[idx];
	rec_info.id = idx;
	if (distance[idx]>0.9)
	{
		rec_info.id = 6;   // negtive
	}
}
