# YOLOP-opencv-dnn
YOLOP, a panoramic driving perception network deployed using OpenCV, can handle traffic target detection, driveable area segmentation, and lane line detection, three visual perception tasks simultaneously, and still contains both C++ and Python versions of the program implementation

The onnx file is downloaded from Baidu Cloud Drive, link: https://pan.baidu.com/s/1A_9cldUHeY9GUle_HO4Crg Extraction code: mf1x

The main program file for the C++ version is main.cpp, and the main program file for the Python version is main.py. After downloading the onnx file to the directory where the main program file is located, you can run the program. The folderimages contains several test images from the bdd100k autopilot dataset.

This program is an opencv inference deployment program based on the recently released project https://github.com/hustvl/YOLOP by the vision team of Huazhong University of Science and Technology. This program can be run by relying only on the opencv library, thus completely getting rid of the dependency on any deep learning framework. If the program runs with errors, it is likely that the version of opencv you installed is low, so you can upgrade the version of opencv to run normally.

In addition, there is an export_onnx.py file in this set, which is the program that generates the onnx file. If you want to know how to generate .onnx files, you need to copy the export_onnx.py file to the home directory of https://github.com/hustvl/YOLOP, and modify the code in lib/models/ common.py, then run export_onnx.py to generate the onnx file. See my csdn blog post https://blog.csdn.net/nihate/article/details/112731327 for what code to change in lib/models/common.py.

Translated with www.DeepL.com/Translator (free version)
