#include <iostream>
#include <vector>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/opencv.hpp>

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace cv;
using namespace std;

int loadModel(string model_name, Session ** session)
{
    GraphDef graph;
   
    Status status = NewSession(SessionOptions(), session);
    if (!status.ok()) {
        std::cerr << "tf error 1: " << status.ToString() << "\n";
        return -1;
    }

    // Читаем граф
    status = ReadBinaryProto(Env::Default(), model_name, &graph);
    if (!status.ok()) {
        std::cerr << "tf error 2: " << status.ToString() << "\n";
        return -1;
    }

    // Добавляем граф в сессию TensorFlow
    status = (*session)->Create(graph);
    if (!status.ok()) {
        std::cerr << "tf error 3: " << status.ToString() << "\n";
        return -1;
    }
    return 0;
}

int loadImage(string img_name, Mat * img)
{   
    *img = imread(img_name, IMREAD_GRAYSCALE);
    if(img->empty())
    {
        cout << "Could not open image" << endl;
        return -1;
    }
    return 0;
}

float computeMatMean(Mat * mat)
{
    const int ROWS = mat->rows;
    const int COLS = mat->cols;
    const float NUM_EL = ROWS*COLS;

    float sum = 0;

    Mat mat_f;
    mat->convertTo(mat_f, CV_32F);

    for(int row = 0; row < ROWS; row++)
    {
        for(int col = 0; col < COLS; col++)
        {   
            sum += mat_f.at<float>(row,col);
        }
    }

    return sum/NUM_EL;

}

float computeMatStd(Mat * mat)
{
    const int ROWS = mat->rows;
    const int COLS = mat->cols;
    float     mean = computeMatMean(mat);

    Mat mat_diff = Mat(ROWS, COLS, CV_32F);
    Mat mat_sq_diff = Mat(ROWS, COLS, CV_32F);
    
    mat->convertTo(mat_diff, CV_32F);
    mat_diff -= mean;
    pow(mat_diff, 2, mat_sq_diff);

    float mean_sq = computeMatMean(&mat_sq_diff);

    return pow(mean_sq, 0.5);
}


void copyMatToTensor(Mat *mat, Tensor *tensor)
{
    float * ptr = tensor->flat<float>().data();
    const int H = tensor->shape().dim_size(1);
    const int W = tensor->shape().dim_size(2);

    Mat tensor_mat(H, W, CV_32F, ptr);
    Mat mat_f(H, W, CV_32F);
    mat->convertTo(mat_f, CV_32F);
    mat_f.copyTo(tensor_mat); 
}

void copyTensorToMat(Tensor *tensor, Mat *mat)
{
    float *ptr = tensor->flat<float>().data();
    const int H = tensor->shape().dim_size(1);
    const int W = tensor->shape().dim_size(2);
    const int D = tensor->shape().dim_size(3);
    int dims[3] = {H, W, D};
    
    *mat = Mat(3, dims, CV_32F, ptr);
}

Mat computeCosine(Mat * mat_l, Mat * mat_r, int max_disp)
{

    const int H = mat_l->size[0];
    const int W = mat_l->size[1] - max_disp;
    const int NUM_FILT = mat_l->size[2];

    cout << "H        = " << H << endl;
    cout << "W        = " << W << endl;
    cout << "NUM_FILT = " << NUM_FILT << endl;

    int dims[3] = {H, W, max_disp};
    Mat predict = Mat(3, dims, CV_32F);

    for(int disp_idx = 0; disp_idx < max_disp; disp_idx++)
    {
        cout << "\rCompute cosine..." << (int)(disp_idx/(float)max_disp*100) << "%" << flush;
        for(int h = 0; h < H; h++)
        {
            for(int w = max_disp; w < W; w++)
            {
                float a = 0;
                float b = 0;
                float c = 0;
                for(int filt_idx = 0; filt_idx < NUM_FILT; filt_idx++)
                {
                    float fv_l = mat_l->at<float>(h, w, filt_idx);
                    float fv_r = mat_r->at<float>(h, w - max_disp + disp_idx, filt_idx);

                    a += fv_l * fv_r;
                    b += fv_l * fv_l;
                    c += fv_r * fv_r;

                }

                predict.at<float>(h, w - max_disp, disp_idx) = a / pow(b * c, 0.5);
            }
        }
    }

    cout << "\rCompute cosine... Done!" << endl;

    return predict;

}

float toMatchingCost(float prediction)
{

    if(isnan(prediction))
        return 1;

    return 1-prediction;
}

Mat computeSGBM(Mat * predict, int cross_size)
{
    const int max_disp = predict->size[2];
    const int n        = predict->size[0];
    const int m        = predict->size[1] + max_disp;
    const int p_column = predict->size[1];
	
    Mat disp_img = Mat::ones(n, m, CV_8U);

    std::vector<float> cost_row(max_disp, 0);
    cout << "M = " << m << endl;
    for (int i = 0; i < n; i++) 
    {
	for (int j = max_disp; j < m; j++)
	{
	    for (int d = 0; d < max_disp; d++)
	    {
	        float cost_h  = 0;
	        float cost_v  = 0;
	        float cost_rd = 0;
	        float cost_ld = 0;
	        int v_ptr     = 0;
	        int h_ptr     = 0;
	        int ld_ptr_y  = 0;

	        for (int r = 0; r < cross_size; r++)
	        {
		    h_ptr = j - max_disp - cross_size / 2 + r;
		    v_ptr = i - cross_size / 2 + r;

		    ld_ptr_y = j - max_disp + (cross_size/2 - r);

		    if (h_ptr < 0) h_ptr = 0;
		    if (h_ptr >= m - max_disp) h_ptr = m - max_disp - 1;
		    if (v_ptr < 0) v_ptr = 0;
		    if (v_ptr >= n) v_ptr = n - 1;
		    if (ld_ptr_y < 0) ld_ptr_y = 0;
		    if (ld_ptr_y >= m - max_disp) ld_ptr_y = m - max_disp - 1;

		    cost_h  += toMatchingCost(predict->at<float>(i, h_ptr, d));
		    cost_v  += toMatchingCost(predict->at<float>(v_ptr, j - max_disp, d));
		    cost_rd += toMatchingCost(predict->at<float>(v_ptr, h_ptr, d));
		    cost_ld += toMatchingCost(predict->at<float>(v_ptr, ld_ptr_y, d));
		}

		cost_row[d] = (cost_h + cost_v + cost_rd + cost_ld)/4;
                //cost_row[d] = predict->at<float>(i, j - max_disp, d);
	    }

	    int i_min = std::min_element(cost_row.begin(), cost_row.end()) - cost_row.begin();
	    //int i_min = std::max_element(cost_row.begin(), cost_row.end()) - cost_row.begin();

	    disp_img.at<uint8>(i, j) = i_min;
			
	}
		
        cout << "\r[i] = [" << i << "/" << n << "]" << flush;
    }

    cout << endl;

    return disp_img;
}

int main() {

    Mat img_l;
    Mat img_r;

    loadImage("../im0.png", &img_l);
    loadImage("../im1.png", &img_r);
  
    tensorflow::Session* model_l;
    tensorflow::Session* model_r;

    loadModel("../model_l.pb", &model_l);
    loadModel("../model_r.pb", &model_r);

    const int H = img_l.rows;
    const int W = img_l.cols;
    Tensor itensor_l (DT_FLOAT, TensorShape({1, H, W, 1}));
    Tensor itensor_r (DT_FLOAT, TensorShape({1, H, W, 1}));

    float img_l_mean = computeMatMean(&img_l);
    float img_l_std  = computeMatStd(&img_l);
    float img_r_mean = computeMatMean(&img_r);
    float img_r_std  = computeMatStd(&img_r);

    Mat img_l_norm = Mat(W, H, CV_32F);
    Mat img_r_norm = Mat(W, H, CV_32F);

    img_l.convertTo(img_l_norm, CV_32F);
    img_r.convertTo(img_r_norm, CV_32F);

    img_l_norm = (img_l_norm - img_l_mean)/img_l_std;
    img_r_norm = (img_r_norm - img_r_mean)/img_r_std;

    copyMatToTensor(&img_l_norm, &itensor_l);
    copyMatToTensor(&img_r_norm, &itensor_r);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs_l = {
        { "input_1", itensor_l }};
    std::vector<tensorflow::Tensor> otensor_l;
    
    std::vector<std::pair<string, tensorflow::Tensor>> inputs_r = {
        { "input_1", itensor_r }};
    std::vector<tensorflow::Tensor> otensor_r;

    // run models
    model_l->Run(inputs_l, {"lc4/Relu"}, {}, &otensor_l);
    model_r->Run(inputs_r, {"rc4/Relu"}, {}, &otensor_r);
    
    //доступ к тензорам-результатам
    Tensor out_l = otensor_l[0];
    Tensor out_r = otensor_r[0];
    Mat mout_l, mout_r;
 
    copyTensorToMat(&out_l, &mout_l);
    copyTensorToMat(&out_r, &mout_r);
    
    /*for(int i = 0; i < 112; i++)
    {
        cout << mout_l.at<float>(0, 0, i) << endl;
    }

    cout << "---" << endl;
    for (int i = 0; i < 112; i++) {
        cout << mout_r.at<float>(0,0,i) << endl;
    }*/

    int max_disp = 70;
    Mat predict = computeCosine(&mout_l, &mout_r, max_disp);

    //cout << "MAX = " << *(max_element(predict.begin<float>(),predict.end<float>())) << endl;
    //cout << "MIN = " << *(min_element(predict.begin<float>(),predict.end<float>())) << endl;

    /*cout << "Predict" << endl;
    cout << "---"     << endl;
    for(int i = 0; i < max_disp; i++)
    {
        float x = predict.at<float>(34, 268, i);
        cout << x << endl;
        if(isnan(x)) cout << "NAN!!!" << endl;
    }*/

    Mat disp_img = computeSGBM(&predict, 30);
    //norm result
    disp_img = 255*(max_disp - disp_img)/max_disp;
    imwrite("out.png", disp_img);

    return 0;
}

