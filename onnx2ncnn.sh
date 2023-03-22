#!/bin/bash
onnxmodel_path="/home/ckn/Code/YOLOX/rtmdet_mobilenetv2_10xb128-100e_not-ratio.onnx"
ncnnmodel_save="/home/ckn/Code/YOLOX"
ncnnmodelname="rtmdet_mobilenetv2_10xb128-100e_not-ratio"
ncnnmodel_bin_save_path=${ncnnmodel_save}/${ncnnmodelname}.bin
ncnnmodel_param_save_path=${ncnnmodel_save}/${ncnnmodelname}.param
ncnnmodel_bin_sim_save_path=${ncnnmodel_save}/${ncnnmodelname}_sim.bin
ncnnmodel_param_sim_save_path=${ncnnmodel_save}/${ncnnmodelname}_sim.param
ncnnmodel_bin_H_save_path=${ncnnmodel_save}/${ncnnmodelname}.bin.h
ncnnmodel_param_H_save_path=${ncnnmodel_save}/${ncnnmodelname}.param.h
ncnn_software_path=/home/ckn/library/ncnn/build/linux2/install/bin
cd ${ncnn_software_path}

# 1. ONNX 转换 NCNN
./onnx2ncnn ${onnxmodel_path} ${ncnnmodel_param_save_path} ${ncnnmodel_bin_save_path}

#2. NCNN Optimize
./ncnnoptimize ${ncnnmodel_param_save_path} ${ncnnmodel_bin_save_path} ${ncnnmodel_param_sim_save_path} ${ncnnmodel_bin_sim_save_path} 65536

#3. NCNN 转换成 .H
./ncnn2mem ${ncnnmodel_param_save_path} ${ncnnmodel_bin_save_path} ${ncnnmodel_param_H_save_path} ${ncnnmodel_bin_H_save_path}