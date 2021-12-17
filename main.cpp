#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/io.h"
#include "dynet/model.h"


#include <iostream>

using namespace std;
using namespace dynet;


int main() {
    ParameterCollection pc; // 创建ParameterCollection收集器, 专门收集相关weights
    SimpleSGDTrainer trainer(pc); // 把收集器交给trainer去更新参数, 这个trainer相当pytorch的optimizer?
    ComputationGraph cg; 
    Expression W = parameter(cg, pc.add_parameters({1,3})); //Expression类型在在torch中相当于是一个torch tensor的数据结构
    std::vector<dynet::real> x_value(3);
    Expression x = input(cg, {3}, &x_value);
    
    return 0;
}