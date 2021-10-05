#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;
const int closeKernelSizeW = 17, closeKernelSizeH = 5;
Mat closeKernel = getStructuringElement(MORPH_RECT, Size(closeKernelSizeW, closeKernelSizeH)); //用于preprocess函数中的闭合操作
// 预处理车图像，返回二值图像
Mat preprocess(Mat &img) {
    resize(img, img, Size(1024, 768)); //统一大小
    Mat imgGray, imgBinary,imgClose;
    vector<Mat> img_channels;
    split(img,img_channels);
    imgGray=3*img_channels[0]-img_channels[1]-img_channels[2]; //提取颜色特征: 3*B-G-R
    imgGray.convertTo(imgGray,CV_8U);
    threshold(imgGray,imgBinary,200,255,THRESH_BINARY); //不用大津法，而是使用自定义阈值th，th可根据统计得出，但这里比较懒，主观设置了
    morphologyEx(imgBinary, imgClose, MORPH_CLOSE, closeKernel, Point(-1, -1), 3);
    return imgClose;
}
//根据二值图像找出车牌，并在原图像img中显示bounding box
void findBounder(Mat &imgBinary, Mat &img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(imgBinary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    float area, peri;
    for (int i = 0; i < contours.size(); i++) {
        area = contourArea(contours[i]); //面积
        peri = arcLength(contours[i], true); //周长
        approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
        boundRect[i] = boundingRect(conPoly[i]); //设置所有轮廓的bounding box
        if (area >= 4800 && area <= 55000) { //用轮廓的面积进行筛选
            if (boundRect[i].width >= 2 * boundRect[i].height && boundRect[i].width <= 4.25* boundRect[i].height) { //用bounding box的宽高比进行筛选
                rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
            }
        }
    }
}


int main() {
    string folder = "E:\\C++Projects\\openCV\\zhiNengShiBie\\ex2Copy\\sources"; //车图像根目录
    string fileName, resultPath;
    int pos;
    vector<String> imgFiles;
    glob(folder, imgFiles);
    Mat img, imgBinary;
    for (String &filepath:imgFiles) { //读取并处理所有车图像
        pos = filepath.find_last_of("\\");
        fileName = filepath.substr(pos + 1);
        img = imread(filepath);
        imgBinary = preprocess(img);
        findBounder(imgBinary, img);
        imshow(fileName + " img", img);
    }
    waitKey(0);
}

