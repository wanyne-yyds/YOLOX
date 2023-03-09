#!/bin/bash

# Usage:
#   MNNConvert [OPTION...]

#   -h, --help                    Convert Other Model Format To MNN Model

#   -v, --version                 显示当前转换器版本
  
#   -f, --framework arg           需要进行转换的模型类型, ex: [TF,CAFFE,ONNX,TFLITE,MNN,TORCH, JSON]
  
#       --modelFile arg           需要进行转换的模型文件名, ex: *.pb,*caffemodel
      
#       --prototxt arg            caffe模型结构描述文件, ex: *.prototxt
      
#       --MNNModel arg            转换之后保存的MNN模型文件名, ex: *.mnn
      
#       --fp16                    将conv/matmul/LSTM的float32参数保存为float16,
#       													模型将减小一半,精度基本无损
      
#       --benchmarkModel          不保存模型中conv/matmul/BN等层的参数,仅用于benchmark测试
      
#       --bizCode arg             MNN模型Flag, ex: MNN
      
#       --debug                   使用debug模型显示更多转换信息
      
#       --forTraining             保存训练相关算子,如BN/Dropout,default: false
      
#       --weightQuantBits arg     arg=2~8,此功能仅对conv/matmul/LSTM的float32权值进行量化,
#       													仅优化模型大小,加载模型后会解码为float32,量化位宽可选2~8,
#                                 运行速度和float32模型一致。8bit时精度基本无损,模型大小减小4倍
#                                 default: 0,即不进行权值量化
      
#       --compressionParamsFile arg
#                                 使用MNN模型压缩工具箱生成的模型压缩信息文件
                                
#       --saveStaticModel         固定输入形状,保存静态模型, default: false
      
#       --inputConfigFile arg     保存静态模型所需要的配置文件, ex: ~/config.txt。文件格式为:
#                                 input_names = input0,input1
#                                 input_dims = 1x3x224x224,1x3x64x64
#       --JsonFile arg            当-f MNN并指定JsonFile时,可以将MNN模型转换为Json文件
#       --info                    当-f MNN时,打印模型基本信息（输入名、输入形状、输出名、模型版本等）
#       --testdir arg             测试转换 MNN 之后,MNN推理结果是否与原始模型一致。
#                                 arg 为测试数据的文件夹,生成方式参考 "正确性校验" 一节
#       --thredhold arg           当启用 --testdir 后,设置正确性校验的误差允可范围
#                                 若不设置,默认是 0.01

onnxmodelpath=/home/ckn/Code/YOLOv3/model_wh_416_128_bw_0.25_fw_0.25_t_6_b_400_i_1_warp_epoch_100_2022-11-08-06-37-08_sim.onnx
mnnmodelpath=/home/ckn/Code/YOLOv3/model_wh_416_128_bw_0.25_fw_0.25_t_6_b_400_i_1_warp_epoch_100_2022-11-08-06-37-08_sim.mnn

/home/ckn/library/MNN/build/MNNConvert -f ONNX \
--modelFile ${onnxmodelpath} \
--MNNModel ${mnnmodelpath} \
--info \
--fp16 \
--bizCode biz