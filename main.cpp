#include<opencv4/opencv2/opencv.hpp>
#include<iostream>
#include<bits/stdc++.h>
#include<math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace std;
using namespace cv;
using namespace cv::ml;

int main()
{
    Mat mask;
    Mat result,img;

	Mat image = imread("6.jpg");
    Mat img_show = image.clone();
    imshow("frame",img_show);
    
    Mat element = getStructuringElement(MORPH_RECT, Size(3,3));
    
    erode(image, mask, element);     
    //imshow("ero",mask); 
    cvtColor(mask, img , COLOR_BGR2GRAY);
    threshold(img, result, 130, 255, THRESH_BINARY);
    blur(result, image, Size(1,1));
   
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(result, contours, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
    
    if( !contours.empty() && !hierarchy.empty() ){    
        vector<vector<Point> >::iterator it;
        for( it = contours.begin(); it != contours.end(); ){  
            //按轮廓长度筛选
            if( arcLength(*it, true) < 450)
                contours.erase(it);
            else it ++;
        }
    }

Ptr<SVM> svm = StatModel::load<SVM>("mnist_svm.xml");
 Mat lasr = Mat::zeros(img.rows, img.cols, CV_8UC3);
 drawContours(lasr, contours, -1, Scalar(0,255,0), 1);
    vector<Rect> boundRect(contours.size()); 
    for(int i = 0; i< contours.size(); i++){ 
        boundRect[i] = boundingRect(contours[i]);
        if(contourArea(contours[i])>15000||contourArea(contours[i])<2000) 
        //框出轮廓 
        rectangle(img_show, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 1);

        Mat roi = img_show(boundRect[i]);
        Mat roi2,roi3,mask,drawing1;
        mask = roi;

        int len = max(roi.cols,roi.rows);
        Mat drawing = Mat::zeros(Size(len,len),CV_8UC3);
        threshold(roi,mask, 150, 255, THRESH_BINARY_INV);
        int a = (drawing.cols/2)-(roi.cols/2);
        int b = (drawing.rows/2)-(roi.rows/2);
        Mat imageROI = drawing(Rect(a,b,roi.cols, roi.rows));
       // cout <<imageROI.cols << endl;
        //cout <<mask.cols << endl;
        //cout <<imageROI.rows << endl;
        //cout <<mask.rows << endl;
        addWeighted(imageROI,0.5,mask,1.0,0, imageROI);
        //drawing.copyTo(imageROI.mask);
        //bitwise_or(drawing,roi,drawing1);
        imshow("11",drawing);
        threshold(drawing,drawing1, 150, 255, THRESH_BINARY_INV);
        imshow("22",drawing1);
        resize(drawing1,roi2,Size(28,28));
        imshow("33",roi2);
        vector<Mat> channels;
        Mat aChannels[3];
        split(roi2, aChannels);
        split(roi2, channels);
        threshold(channels[2],roi3, 150, 255, THRESH_BINARY_INV);
        roi3 = roi3.reshape(1,1);
        roi3.convertTo(roi3, CV_32F);
        cout <<roi3.cols << endl;
        cout <<svm->getVarCount() << endl;
        int ret = svm->predict(roi3);
        putText(img_show,to_string(ret),boundRect[i].br(),cv::FONT_HERSHEY_COMPLEX,1,Scalar(0, 255, 255),1, 5, 0);

       

        
    }

	imshow("img", img_show);
	waitKey(0);

    return 0;

}
