#include <iostream>
#include <vector>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include <opencv2/opencv.hpp>
#include "boost/date_time/posix_time/posix_time.hpp"
#include "termcolor.hpp"
#include <thread>

#ifdef __AVX__
#include <immintrin.h>
#endif

using namespace tensorflow;
using namespace tensorflow::ops;
using namespace cv;
using namespace std;
using namespace boost::posix_time;

ptime getTimeUS()
{
    return microsec_clock::local_time();
}

float calcTimeDiffMS(ptime start_time)
{
    time_duration diff = getTimeUS() - start_time;
    return float(diff.total_microseconds())/1000.0;
}

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

float getElement(float * mas, int i, int j, int k, int j_max, int k_max)
{
    return mas[i*j_max*k_max + j*k_max + k];
}

float * getElementPtr(float * mas, int i, int j, int k, int j_max, int k_max)
{
    return &mas[i*j_max*k_max + j*k_max + k];
}

struct CosineData {
    float * predict_ptr;
    float * mat_l_ptr;
    float * mat_r_ptr;
    int W;
    int H;
    int max_disp;
    int NUM_FILT;

};

void computeCosineNDisp(CosineData * cosine_data, int start_disp, int end_disp)
{

    int W        = cosine_data->W;
    int H        = cosine_data->H;
    int max_disp = cosine_data->max_disp;
    int NUM_FILT = cosine_data->NUM_FILT;
    
    #ifdef __AVX__
    float * fv_l_mas;
    float * fv_r_mas;
    float avx_res_mas[8];
    
    __m256 mm_l_mas, mm_r_mas;
    __m256 mm_reg1, mm_reg2, mm_reg3, mm_reg4;
    
    const int NUM_FILT_AVX = 8*(NUM_FILT/8);
    
    #endif

    for(int disp_idx = start_disp; disp_idx < end_disp; disp_idx++)
    {
        for(int h = 0; h < cosine_data->H; h++)
        {
            for(int w = max_disp; w < W + max_disp; w++)
            {
                float a = 0;
                float b = 0;
                float c = 0;
                
                #ifdef __AVX__
                
                
                for(int filt_idx = 0; filt_idx < NUM_FILT_AVX; filt_idx += 8)
                {
 
                    fv_l_mas = getElementPtr(cosine_data->mat_l_ptr, h, w, filt_idx, W + max_disp, NUM_FILT);
                    fv_r_mas = getElementPtr(cosine_data->mat_r_ptr, h, w - max_disp + disp_idx, filt_idx, W + max_disp, NUM_FILT);
                    
                    mm_l_mas = _mm256_loadu_ps(fv_l_mas);
                    mm_r_mas = _mm256_loadu_ps(fv_r_mas);
                    
                    mm_reg1 = _mm256_mul_ps(mm_l_mas, mm_r_mas);
                    mm_reg2 = _mm256_mul_ps(mm_l_mas, mm_l_mas);
                    mm_reg3 = _mm256_mul_ps(mm_r_mas, mm_r_mas);
                    
                    mm_reg1 = _mm256_hadd_ps(mm_reg1, mm_reg2);
                    mm_reg1 = _mm256_hadd_ps(mm_reg1, mm_reg3);
                    
                    mm_reg2 = _mm256_permute2f128_ps(mm_reg1,  mm_reg1, 0x01);
                    mm_reg3 = _mm256_add_ps(mm_reg1, mm_reg2);
                    
                    _mm256_store_ps(avx_res_mas, mm_reg3);
                    
                    a += avx_res_mas[0];
                    b += avx_res_mas[1];                  
                    c += avx_res_mas[2] + avx_res_mas[3];
                    
                }
                
                for(int filt_idx = NUM_FILT_AVX; filt_idx < NUM_FILT; filt_idx++)
                {
                    float fv_l = getElement(cosine_data->mat_l_ptr, h, w, filt_idx, W + max_disp, NUM_FILT);
                    float fv_r = getElement(cosine_data->mat_r_ptr, h, w - max_disp + disp_idx, filt_idx, W + max_disp, NUM_FILT);

                    a += fv_l * fv_r;
                    b += fv_l * fv_l;
                    c += fv_r * fv_r;
                }
                
                #else
                for(int filt_idx = 0; filt_idx < NUM_FILT; filt_idx++)
                {
                    float fv_l = getElement(cosine_data->mat_l_ptr, h, w, filt_idx, W + max_disp, NUM_FILT);
                    float fv_r = getElement(cosine_data->mat_r_ptr, h, w - max_disp + disp_idx, filt_idx, W + max_disp, NUM_FILT);

                    a += fv_l * fv_r;
                    b += fv_l * fv_l;
                    c += fv_r * fv_r;
                }
                #endif

                cosine_data->predict_ptr[h*W*max_disp + (w - max_disp)*max_disp + disp_idx] = a / pow(b * c, 0.5);
            }
        }
    }
}

Mat computeCosine(Mat * mat_l, Mat * mat_r, int max_disp, int num_threads)
{

    const int H = mat_l->size[0];
    const int W = mat_l->size[1] - max_disp;
    const int NUM_FILT = mat_l->size[2];

    cout << "H        = " << H << endl;
    cout << "W        = " << W << endl;
    cout << "NUM_FILT = " << NUM_FILT << endl;

    int dims[3] = {H, W, max_disp};
    Mat predict = Mat(3, dims, CV_32F);

    CosineData cosine_data;
    cosine_data.predict_ptr = predict.ptr<float>();
    cosine_data.mat_l_ptr   = mat_l->ptr<float>();
    cosine_data.mat_r_ptr   = mat_r->ptr<float>();
    cosine_data.W           = W;
    cosine_data.H           = H;
    cosine_data.max_disp    = max_disp;
    cosine_data.NUM_FILT    = NUM_FILT;

    vector<thread> threads;
    int ndisp_per_thread = ceil(max_disp/num_threads);
    for(int i = 0; i < num_threads; i++)
    {
        int start_disp = i*ndisp_per_thread;
        int end_disp   = start_disp + ndisp_per_thread; 
        threads.push_back(thread(computeCosineNDisp, &cosine_data, start_disp, end_disp));
    }
    for_each(threads.begin(), threads.end(), mem_fn(&thread::join));

    cout << "\rCompute cosine... Done!" << endl;

    return predict;

}

float toMatchingCost(float prediction)
{

    if(isnan(prediction))
        return 1;

    return 1-prediction;
}

struct SGBMData {
    float * predict_ptr;
    int cross_size;
    Mat * disp_img_ptr;
    int max_disp;
    int m;
    int n;
};

void computeSGBMNDisp(SGBMData * sgbm_data, int start_row, int end_row)
{
    float * predict_ptr = sgbm_data->predict_ptr;
    int max_disp        = sgbm_data->max_disp;
    int m               = sgbm_data->m;
    int n               = sgbm_data->n;
    int cross_size      = sgbm_data->cross_size;
    
    vector<float> cost_row = vector<float>(max_disp, 0);

    for (int i = start_row; i < end_row; i++) 
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
	                
	                cost_h  += getElement(predict_ptr, i, h_ptr, d, m - max_disp, max_disp);
	                cost_v  += getElement(predict_ptr, v_ptr, j - max_disp, d, m - max_disp, max_disp);
	                cost_rd += getElement(predict_ptr, v_ptr, h_ptr, d, m - max_disp, max_disp);
	                cost_ld += getElement(predict_ptr, v_ptr, ld_ptr_y, d, m - max_disp, max_disp);

	            }

                cost_row[d] = (cost_h + cost_v + cost_rd + cost_ld)/4;
            }
            
            int i_min = std::min_element(cost_row.begin(), cost_row.end()) - cost_row.begin();

	        (sgbm_data->disp_img_ptr)->at<uint8>(i, j) = i_min;
        }
    }
        
}


Mat computeSGBM(Mat * predict, int cross_size, int num_threads)
{
    const int max_disp = predict->size[2];
    const int n        = predict->size[0];
    const int m        = predict->size[1] + max_disp;
    const int p_column = predict->size[1];
	
    Mat disp_img = Mat::ones(n, m, CV_8U);

    SGBMData sgbm_data;
    sgbm_data.predict_ptr  = predict->ptr<float>();
    sgbm_data.cross_size   = cross_size;
    sgbm_data.disp_img_ptr = &disp_img;
    sgbm_data.max_disp     = max_disp;
    sgbm_data.m            = m;
    sgbm_data.n            = n;

    for(int i = 0; i < n*p_column*max_disp; i++)
    {
        sgbm_data.predict_ptr[i] = toMatchingCost(sgbm_data.predict_ptr[i]);
    }
 
    //computeSGBMNDisp(&sgbm_data, 0, n);
    
    vector<thread> threads;
    int nrow_per_thread = ceil(n/num_threads);
    for(int tn = 0; tn < num_threads; tn++)
    {
        int start_row = tn*nrow_per_thread;
        int end_row   = start_row + nrow_per_thread; 
        threads.push_back(thread(computeSGBMNDisp, &sgbm_data, start_row, end_row));
    }
    for_each(threads.begin(), threads.end(), mem_fn(&thread::join));

    return disp_img;
}

int main() {

    ptime start_time;
    float exec_time;

    Mat img_l;
    Mat img_r;

    loadImage("../../../../samples/Middlebury_scenes_2014/trainingQ/Motorcycle/im0.png", &img_l);
    loadImage("../../../../samples/Middlebury_scenes_2014/trainingQ/Motorcycle/im1.png", &img_r);
  
    tensorflow::Session* model_l;
    tensorflow::Session* model_r;

    loadModel("../../models/model_3x3x10_l.pb", &model_l);
    loadModel("../../models/model_3x3x10_r.pb", &model_r);

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
    start_time = getTimeUS();
    
    // ATTENTION! Change to lcN, rcN, where N - is a number of layers
    model_l->Run(inputs_l, {"lc1/Relu"}, {}, &otensor_l);
    model_r->Run(inputs_r, {"rc1/Relu"}, {}, &otensor_r);
    exec_time = calcTimeDiffMS(start_time);

    float conv_time = exec_time/1000;
    cout << termcolor::green << 
        "Time (calc Conv layers): " << conv_time << " sec" <<
        termcolor::reset << endl;
    
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

    int max_disp    = 70;
    int num_threads = 9;

    start_time  = getTimeUS();
    Mat predict = computeCosine(&mout_l, &mout_r, max_disp, num_threads);
    exec_time   = calcTimeDiffMS(start_time);

    float computeCosine_time = exec_time/1000;
    cout << termcolor::green << 
        "Time (computeCosine): " << computeCosine_time << " sec" <<
        termcolor::reset << endl;

    /*cout << "Predict" << endl;
    cout << "---"     << endl;
    for(int i = 0; i < max_disp; i++)
    {
        float x = predict.at<float>(34, 268, i);
        cout << x << endl;
        if(isnan(x)) cout << "NAN!!!" << endl;
    }*/

    start_time   = getTimeUS();
    Mat disp_img = computeSGBM(&predict, 15, num_threads);
    exec_time    = calcTimeDiffMS(start_time); 

    float computeSGBM_time = exec_time/1000;
    cout << termcolor::green << 
        "Time (computeSGBM): " << computeSGBM_time << " sec" <<
        termcolor::reset << endl;
        
    float total_time = conv_time + computeCosine_time + computeSGBM_time;
    cout << termcolor::red << 
        "Total time: " << total_time << " sec" <<
        termcolor::reset << endl;

    //norm result
    disp_img = 255*(max_disp - disp_img)/max_disp;
    imwrite("out.png", disp_img);

    return 0;
}

