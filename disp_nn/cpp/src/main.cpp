#include <iostream>
#include <vector>
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"

using namespace tensorflow;
using namespace tensorflow::ops;

int main() {

    /* Scope root = Scope::NewRootScope();

    auto A = Const(root, {{1.f, 2.f}, {3.f, 4.f}});
    auto b = Const(root, {{5.f, 6.f}});
    auto x = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
    std::vector<Tensor> outputs;

    std::unique_ptr<ClientSession> session = std::make_unique<ClientSession>(root);
    TF_CHECK_OK(session->Run({x}, &outputs));
    std::cout << outputs[0].matrix<float>();*/

    tensorflow::GraphDef graph_def;
    tensorflow::Session* session;
	
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        std::cerr << "tf error 1: " << status.ToString() << "\n";
    }

    // Читаем граф
    status = ReadBinaryProto(Env::Default(), "../model.pb", &graph_def);
    if (!status.ok()) {
        std::cerr << "tf error 2: " << status.ToString() << "\n";
    }

    // Добавляем граф в сессию TensorFlow
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cerr << "tf error 3: " << status.ToString() << "\n";
    }

    const int W = 497;
    const int H = 741;
    Tensor inputTensor1 (DT_FLOAT, TensorShape({1, W, H, 1}));

    //заполнение тензоров-входных данных
    /*for (int i = 0; i < W; i++) {
        for (int j = 0; j < H; j++) {
	        inputTensor1.matrix<float>()(i, j, 1) = 5;
	    }
    }*/
	
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        { "input_1", inputTensor1 }
    };
    //здесь мы увидим тензоры - результаты операций
    std::vector<tensorflow::Tensor> outputTensors;
    //операции возвращающие значения и не возвращающие передаются в разных параметрах
    status = session->Run(inputs, {"conv4/Relu"}, {}, &outputTensors);
    
    if (!status.ok()) {
        std::cerr << "tf error 4: " << status.ToString() << "\n";
	return 0;
    }
    
    //доступ к тензорам-результатам
    /* for (int i...) {
        outputs [0].matrix<float>()(0, i++);
    }*/

    return 0;
}

