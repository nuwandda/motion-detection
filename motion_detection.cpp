#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void detector(Mat frame, VideoCapture capture);

Mat firstFrame;
const int newWidth = 400;
const int newHeight = 400;
int dilationElem = 0;
int dilationSize = 0;
int const maxElem = 2;
int const maxKernelSize = 21;
int const minArea = 400;

int main(int argc, const char** argv)
{
    VideoCapture capture;
    Mat frame;
    capture.open(0);
    if( capture.isOpened() )
    {
        cout << "Video capturing has been started ..." << endl;
        detector(frame, capture);
    }
    else
    {
        cerr << "ERROR: Could not initiate capture" << endl;
        return -1;
    }
    return 0;
}

void detector(Mat frame, VideoCapture capture)
{
    while (true)
    {
        capture >> frame;
        // if the frame is empty, then we reached the end
        if( frame.empty() )
            break;

        // resize the frame, convert it to grayscale, and blur it
        Mat resizedFrame;
        resize(frame, resizedFrame, Size(newWidth, newHeight));

        Mat grayFrame;
        cvtColor(resizedFrame, grayFrame, COLOR_BGR2GRAY);
        GaussianBlur(grayFrame, grayFrame, Size(21, 21), 0);

        // grab the first frame
        if (firstFrame.empty())
        {
            firstFrame = grayFrame;
            continue;
        }

        // compute absolute difference between the current frame and
        // first frame
        Mat deltaFrame;
        absdiff(firstFrame,grayFrame, deltaFrame);
        Mat threshFrame;
        threshold(deltaFrame, threshFrame, 25, 255, THRESH_BINARY);

        // dilate the thresholded frame
        int dilation_type = 0;
        if( dilationElem == 0 ){ dilation_type = MORPH_RECT; }
        else if( dilationElem == 1 ){ dilation_type = MORPH_CROSS; }
        else if( dilationElem == 2) { dilation_type = MORPH_ELLIPSE; }

        Mat element = getStructuringElement( dilation_type,
                             Size( 2*dilationSize + 1, 2*dilationSize+1 ),
                             Point( dilationSize, dilationSize ) );
        dilate(threshFrame, threshFrame, element);

        Mat threshFrameClone = threshFrame.clone();
        vector<vector<Point> > cnts;
        vector<Vec4i> hierarchy;

        // then find contours
        findContours(threshFrameClone, cnts, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<Rect> boundRect( cnts.size() );
        vector<vector<Point> > contoursPoly( cnts.size() );

        // loop over the contours
        for (size_t i = 0; i < cnts.size(); i++) {
           // use cnts[i] for the current contour
            if (contourArea(cnts[i]) < minArea)
            {
                continue;
            }
            approxPolyDP( Mat(cnts[i]), contoursPoly[i], 3, true );
            boundRect[i] = boundingRect(Mat(contoursPoly[i]));
        }

        for (size_t j = 0; j < cnts.size(); j++)
        {
            // loop over the contours again to draw rectangles
            rectangle( resizedFrame, boundRect[j].tl(), boundRect[j].br(), Scalar(0, 255, 0), 2, 8, 0 );
        }
        // Show in a window
        namedWindow("Motion", CV_WINDOW_AUTOSIZE);
        imshow("Motion", resizedFrame);

        char c = (char)waitKey(10);
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
}
