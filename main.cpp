#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <bits/stdc++.h>
#include <direct.h>
#include <io.h>

using namespace std;
using namespace cv;

void getSpecificChannel(Mat &img, Mat &dst, int channel) {
    dst.create(img.size[0], img.size[1], CV_8U);
    int fromTo[2] = {channel, 0};
    mixChannels(img, dst, fromTo, 1);
}

const int closeKernelSizeW = 17, closeKernelSizeH = 5;
Mat closeKernel = getStructuringElement(MORPH_RECT, Size(closeKernelSizeW, closeKernelSizeH));
Mat rodeKernel = Mat::zeros(Size(5, 5), CV_8U);
Mat kernelX = getStructuringElement(MORPH_RECT, Size(55, 1));

Mat preorcess(Mat &img) {
    resize(img, img, Size(1024, 768));
    rodeKernel.col(2) = 1;
    Mat imgGray, imgBlur, edges, imgDil, imgRode, imgClose;
//    cvtColor(img,imgGray,COLOR_BGR2GRAY);
    getSpecificChannel(img, imgGray, 0);
    GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 3);
//    Sobel(imgBlur, edges, -1, 1, 0);
//    threshold(edges, edges, 0, 255, THRESH_OTSU);
//    Canny(imgBlur,edges,25,75);
    Laplacian(imgBlur, edges, -1);
    threshold(edges, edges, 0, 255, THRESH_OTSU);
    erode(edges, imgRode, rodeKernel);
    morphologyEx(imgRode, imgClose, MORPH_CLOSE, closeKernel, Point(-1, -1), 3);
    Mat bImgX;
    dilate(imgClose, bImgX, kernelX);
    return bImgX;
}

void findBounder(Mat &bImg, Mat &img) {
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(bImg, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    vector<vector<Point>> conPoly(contours.size());
    vector<Rect> boundRect(contours.size());
    float area, peri;
    vector<Point> retConPoly;
    double bestRatio=4.3,minInterval=3.5,curRatio;
    int bestRectInd=-1;
    for (int i = 0; i < contours.size(); i++) {
        area = contourArea(contours[i]);
        peri = arcLength(contours[i], true);
        approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
        boundRect[i] = boundingRect(conPoly[i]);
        if (area >= 12000 && area <= 55000) {
            curRatio=boundRect[i].width*1.0/boundRect[i].height;
            if (curRatio>=2&&curRatio<=5.5) {
                if(abs(curRatio-bestRatio)<minInterval){
                    minInterval=abs(curRatio-bestRatio);
                    bestRectInd=i;
                }
            }
        }
    }
    if(bestRectInd>=0){
        rectangle(img, boundRect[bestRectInd].tl(), boundRect[bestRectInd].br(), Scalar(0, 255, 0), 5);
    }

}


int main() {
//    string filepath = "E:\\C++Projects\\openCV\\zhiNengShiBie\\ex2Copy\\sources\\car3.bmp";
    string filepath = "sources\\1.jpg";
    Mat img, bImg;
    img = imread(filepath);
    bImg = preorcess(img);
    findBounder(bImg, img);
    imshow("img", img);
//    imshow("bImg", bImg);
    waitKey(0);
}

