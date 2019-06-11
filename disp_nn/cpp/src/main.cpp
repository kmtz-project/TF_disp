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

void copyMatToTensor(Mat *mat, Tensor *tensor)
{
    float * ptr = tensor->flat<float>().data();
    const int W = tensor->shape().dim_size(1);
    const int H = tensor->shape().dim_size(2);

    Mat tensor_mat(W, H, CV_32F, ptr);
    Mat mat_f(W, H, CV_32F);
    mat->convertTo(mat_f, CV_32F);
    mat_f.copyTo(tensor_mat); 
}

void copyTensorToMat(Tensor *tensor, Mat *mat)
{
    float *ptr = tensor->flat<float>().data();
    const int W = tensor->shape().dim_size(1);
    const int H = tensor->shape().dim_size(2);
    const int D = tensor->shape().dim_size(3);
    int dims[3] = {W, H, D};
    
    *mat = Mat(3, dims, CV_32F, ptr);
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

    const int W = img_l.rows;
    const int H = img_l.cols;
    Tensor itensor_l (DT_FLOAT, TensorShape({1, W, H, 1}));
    Tensor itensor_r (DT_FLOAT, TensorShape({1, W, H, 1}));

    copyMatToTensor(&img_l, &itensor_l);
    copyMatToTensor(&img_r, &itensor_r);

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
    
    for(int i = 0; i < 112; i++)
    {
        cout << mout_l.at<float>(0, 0, i) << endl;
    }

    cout << "---" << endl;
    for (int i = 0; i < 112; i++) {
        cout << mout_r.at<float>(0,0,i) << endl;
    }

    return 0;
}

