# YOLOP-opencv-dnn
使用OpenCV部署全景驾驶感知网络YOLOP，可同时处理交通目标检测、可驾驶区域分割、车道线检测，三项视觉感知任务，依然是包含C++和Python两种版本的程序实现

onnx文件从百度云盘下载，链接：https://pan.baidu.com/s/1A_9cldUHeY9GUle_HO4Crg 
提取码：mf1x

C++版本的主程序文件是main.cpp，Python版本的主程序文件是main.py。把onnx文件下载到主程序文件所在目录后，就可以运行程序了。文件夹images
里含有若干张测试图片，来自于bdd100k自动驾驶数据集。

本套程序是在华中科技大学视觉团队在最近发布的项目https://github.com/hustvl/YOLOP的基础上做的一个opencv推理部署程序，本套程序只依赖opencv库就可以运行，
从而彻底摆脱对任何深度学习框架的依赖。如果程序运行出错，那很有可能是您安装的opencv版本低了，这时升级opencv版本就能正常运行的。

此外，在本套程序里，还有一个export_onnx.py文件，它是生成onnx文件的程序。不过，export_onnx.py文件不能本套程序目录内运行的，
假如您想了解如何生成.onnx文件，需要把export_onnx.py文件拷贝到https://github.com/hustvl/YOLOP的主目录里之后，并且修改lib/models/common.py里的代码，
这时运行export_onnx.py就可以生成onnx文件了。在lib/models/common.py里修改哪些代码，可以参见我的csdn博客文章
https://blog.csdn.net/nihate/article/details/112731327
