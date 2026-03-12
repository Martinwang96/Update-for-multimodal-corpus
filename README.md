# Repo 概述/Repository Overview
1. 本仓库记录了基于真实课堂教师视频数据开展的视觉模态初步生成实验。当前系统主要聚焦于多模态交互中的视觉通道，涵盖 头部（Head）、躯体（Body）、面部（Face）和手势（Gesture） 四个维度的识别与生成。  
2. 各模态的技术实现参考如下开源工具与项目:  
  - 头部与面部（Head & Face）：基于 OpenFace 2.2.0 进行特征提取与分析  
  - 躯体（Body）：基于 MMPose 2.0 进行人体姿态识别  
  - 手势（Gesture）：参考项目 https://github.com/EsamGhaleb/medal_workshop_on_multimodal_interaction

This repository documents preliminary experiments on visual-based multimodal generation using real classroom teacher recordings.  
The current system focuses on visual modalities in multimodal interaction and includes four main components: Head, Body, Face, and Gesture.  
The implementation of each modality is based on the following open-source frameworks:  
  - Head & Face: Implemented using OpenFace 2.2.0 for facial behavior analysis and feature extraction.  
  - Body: Implemented using MMPose 2.0 for human pose estimation.  
  - Gesture: Based on the project: https://github.com/EsamGhaleb/medal_workshop_on_multimodal_interaction  

# 更新日志/Update Log

## 2026-03-12
目前已将 躯体姿态识别模块 集成为一个独立的网页。  
运行该模块需要在已配置 MMPose 环境 的条件下执行脚本：综合处理-web.py  
该脚本将启动对应的网页界面用于躯体姿态处理。  

The body pose module has now been integrated into a standalone web interface.  
To run the module, ensure that the MMPose environment is properly installed and execute: 综合处理-web.py  
This script will launch the web interface for body pose processing.  
