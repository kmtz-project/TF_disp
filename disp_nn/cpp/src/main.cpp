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

int main() {

    /* Scope root = Scope::NewRootScope();

    auto A = Const(root, {{1.f, 2.f}, {3.f, 4.f}});
    auto b = Const(root, {{5.f, 6.f}});
    auto x = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
    std::vector<Tensor> outputs;

    std::unique_ptr<ClientSession> session = std::make_unique<ClientSession>(root);
    TF_CHECK_OK(session->Run({x}, &outputs));
    std::cout << outputs[0].matrix<float>();*/

    Mat image;

    image = imread("../im0.png", IMREAD_GRAYSCALE);

    if(image.empty())
    {
        std::cout << "Could not open image" << std::endl;
        return -1;
    }
    
    tensorflow::GraphDef graph_def;
    tensorflow::Session* session;
	
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << "tf error 1: " << status.ToString() << "\n";
    }

    // Читаем граф
    status = ReadBinaryProto(Env::Default(), "../model_l.pb", &graph_def);
    if (!status.ok()) {
        std::cerr << "tf error 2: " << status.ToString() << "\n";
    }

    // Добавляем граф в сессию TensorFlow
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << "tf error 3: " << status.ToString() << "\n";
    }

    const int W = image.rows;
    const int H = image.cols;
    Tensor inputTensor1 (DT_FLOAT, TensorShape({1, W, H, 1}));

    // copy OpenCV Mat to Tensor
    float * ptr = inputTensor1.flat<float>().data();
    Mat tensor_image(W, H, CV_32F, ptr);
    Mat image_f(W, H, CV_32FC1);
    image.convertTo(image_f, CV_32F);
    image_f.copyTo(tensor_image); 
   
    cout << tensor_image.type() << endl;
    std::cout << "H: " << image.rows << endl;
    std::cout << "W: " << image.cols << endl;
    for(int i = 0; i < 10; i++)
    {
        std::cout << (int)image_f.at<float>(0, i) << std::endl;
    }

    cout << "---" << endl;
    //заполнение тензоров-входных данных
    /*for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
	        //inputTensor1.matrix<float>()(0, i, j, 0) = image.at<uint8>(i, j);
                inputTensor1(0, i, j, 0) = image.at<uint8>(i, j);
	    }
    }*/
	
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        { "input_1", inputTensor1 }
    };
    //здесь мы увидим тензоры - результаты операций
    std::vector<tensorflow::Tensor> outputTensors;
    //операции возвращающие значения и не возвращающие передаются в разных параметрах
    status = session->Run(inputs, {"lc4/Relu"}, {}, &outputTensors);
    
    if (!status.ok()) {
        std::cerr << "tf error 4: " << status.ToString() << "\n";
	return 0;
    }
    
    //доступ к тензорам-результатам
    Tensor outputs = outputTensors[0];
    cout << outputs.shape() << endl;
    float *pout = outputs.flat<float>().data();
    int dims[3] = {W-8, H-8, 112};
    Mat tout(3, dims, CV_32F, pout);
    for (int i = 0; i < 112; i++) {
        //outputs [0].matrix<float>()(0, i++);
        cout << tout.at<float>(0,0,i) << endl;
    }

    return 0;
}

