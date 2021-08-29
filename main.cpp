#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

class YOLO
{
public:
	YOLO(string modelpath, float confThreshold, float nmsThreshold, float objThreshold);
	Mat detect(Mat& frame);
private:
	const float mean[3] = { 0.485, 0.456, 0.406 };
	const float std[3] = { 0.229, 0.224, 0.225 };
	const float anchors[3][6] = { {3,9,5,11,4,20}, {7,18,6,39,12,31},{19,50,38,81,68,157} };
	const float stride[3] = { 8.0, 16.0, 32.0 };
	const string classesFile = "bdd100k.names";
	const int inpWidth = 640;
	const int inpHeight = 640;
	float confThreshold;
	float nmsThreshold;
	float objThreshold;
	const bool keep_ratio = true;
	vector<string> classes;
	Net net;
	Mat resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left);
	void normalize(Mat& srcimg);
	void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
};

YOLO::YOLO(string modelpath, float confThreshold, float nmsThreshold, float objThreshold)
{
	this->confThreshold = confThreshold;
	this->nmsThreshold = nmsThreshold;
	this->objThreshold = objThreshold;

	ifstream ifs(this->classesFile.c_str());
	string line;
	while (getline(ifs, line)) this->classes.push_back(line);
	this->net = readNet(modelpath);
}

Mat YOLO::resize_image(Mat srcimg, int* newh, int* neww, int* top, int* left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
		}
		else
		{
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
		}
	}
	else
	{
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}

void YOLO::normalize(Mat& img)
{
	img.convertTo(img, CV_32F);
	int i = 0, j = 0;
	const float scale = 1.0 / 255.0;
	for (i = 0; i < img.rows; i++)
	{
		float* pdata = (float*)(img.data + i * img.step);
		for (j = 0; j < img.cols; j++)
		{
			pdata[0] = (pdata[0] * scale - this->mean[0]) / this->std[0];
			pdata[1] = (pdata[1] * scale - this->mean[1]) / this->std[1];
			pdata[2] = (pdata[2] * scale - this->mean[2]) / this->std[2];
			pdata += 3;
		}
	}
}

void YOLO::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)   // Draw the predicted bounding box
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 2);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	label = this->classes[classId] + ":" + label;

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 1);
}

Mat YOLO::detect(Mat& srcimg)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(srcimg, &newh, &neww, &padh, &padw);
	this->normalize(dstimg);
	Mat blob = blobFromImage(dstimg);
	this->net.setInput(blob);
	vector<Mat> outs;
	this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

	Mat outimg = srcimg.clone();
	float ratioh = (float)newh / srcimg.rows;
	float ratiow = (float)neww / srcimg.cols;
	int i = 0, j = 0, area = this->inpHeight*this->inpWidth;
	float* pdata_drive = (float*)outs[1].data;  ///drive area segment
	float* pdata_lane_line = (float*)outs[2].data;  ///lane line segment
	for (i = 0; i < outimg.rows; i++)
	{
		for (j = 0; j < outimg.cols; j++)
		{
			const int x = int(j*ratiow) + padw;
			const int y = int(i*ratioh) + padh;
			if (pdata_drive[y * this->inpWidth + x] < pdata_drive[area + y * this->inpWidth + x])
			{
				outimg.at<Vec3b>(i, j)[0] = 0;
				outimg.at<Vec3b>(i, j)[1] = 255;
				outimg.at<Vec3b>(i, j)[2] = 0;
			}
			if (pdata_lane_line[y * this->inpWidth + x] < pdata_lane_line[area + y * this->inpWidth + x])
			{
				outimg.at<Vec3b>(i, j)[0] = 255;
				outimg.at<Vec3b>(i, j)[1] = 0;
				outimg.at<Vec3b>(i, j)[2] = 0;
			}
		}
	}
	/////generate proposals
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	ratioh = (float)srcimg.rows / newh;
	ratiow = (float)srcimg.cols / neww;
	int n = 0, q = 0, nout = this->classes.size() + 5, row_ind = 0;
	float* pdata = (float*)outs[0].data;
	for (n = 0; n < 3; n++)   ///�߶�
	{
		int num_grid_x = (int)(this->inpWidth / this->stride[n]);
		int num_grid_y = (int)(this->inpHeight / this->stride[n]);
		for (q = 0; q < 3; q++)    ///anchor��
		{
			const float anchor_w = this->anchors[n][q * 2];
			const float anchor_h = this->anchors[n][q * 2 + 1];
			for (i = 0; i < num_grid_y; i++)
			{
				for (j = 0; j < num_grid_x; j++)
				{
					const float box_score = pdata[4];
					if (box_score > this->objThreshold)
					{
						Mat scores = outs[0].row(row_ind).colRange(5, outs[0].cols);
						Point classIdPoint;
						double max_class_socre;
						// Get the value and location of the maximum score
						minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
						if (max_class_socre > this->confThreshold)
						{
							float cx = (pdata[0] * 2.f - 0.5f + j) * this->stride[n];  ///cx
							float cy = (pdata[1] * 2.f - 0.5f + i) * this->stride[n];   ///cy
							float w = powf(pdata[2] * 2.f, 2.f) * anchor_w;   ///w
							float h = powf(pdata[3] * 2.f, 2.f) * anchor_h;  ///h

							int left = (cx - 0.5*w - padw)*ratiow;
							int top = (cy - 0.5*h - padh)*ratioh;   

							classIds.push_back(classIdPoint.x);
							confidences.push_back(max_class_socre * box_score);
							boxes.push_back(Rect(left, top, (int)(w*ratiow), (int)(h*ratioh)));
						}
					}
					row_ind++;
					pdata += nout;
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, outimg);
	}
	return outimg;
}

int main()
{
	YOLO yolo_model("yolop.onnx", 0.25, 0.45, 0.5);
	string imgpath = "images/0ace96c3-48481887.jpg";
	Mat srcimg = imread(imgpath);
	Mat outimg = yolo_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}