# openpose\_tensorflow\_pred #
## 说明 ##
本项目是openpose框架（人体骨架检测）的预测过程的tensorflow实现。

[原项目地址](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)(今天才发现，上面已经有好几个版本的tensorflow实现了)

[原项目torch python版本实现](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)
## 文件结构 ##
-- pipline.py:将个功能模块串联起来，实现最终检测。  
-- preprocessimg.py:将输入的图片按不同比例缩放，并进行归一化处理，形成输入的batch。   
-- restoremodel.py: 加载训练好的权重。
-- getoutputs：计算原始图片和翻转后的图片经过网络后输出的heatMaps和pafs，并计算均值。  
-- NMS.py:非极大值抑制，获得heatMaps局部范围的极大值（关节点）和score。  
-- findconnectedjoints:将heatMaps检测到的关节连接成骨骼。  
-- grouplimbs.py:将检测到的骨骼分为不同人。  
-- plotlimbs.py:画出骨骼框架。  
-- model.py:特征提取网络和stage1-7网络。  
-- config.py:保存模型运行的配置。  
-- model  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- transform.py:将caffe模型转换为tensorflow模型。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- modelcaffe:保存了作者提供的caffe模型权重。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-- modeltensor:保存转换的tensorflow模型的权重。
## 运行方式 ##
python pipline.py
## 参考 ##
[悟乙己](https://blog.csdn.net/sinat_26917383/article/details/79704097)  
[Hibercraft](https://blog.csdn.net/hibercraft/article/details/79377997)