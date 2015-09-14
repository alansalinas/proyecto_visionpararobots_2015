#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "SDL/SDL.h"
/*
 * A simple 'getting started' interface to the ARDrone, v0.2
 * author: Tom Krajnik
 * The code is straightforward,
 * check out the CHeli class and main() to see
 */
#include <stdlib.h>
#include "CHeli.h"
#include <unistd.h>
#include <stdio.h>
#include <iostream>

#define KEYBOARD_DELAY 5

#define HSV_FILTER_H_MIN	60
#define HSV_FILTER_S_MIN	1
#define HSV_FILTER_V_MIN	10

#define HSV_FILTER_H_MAX	100
#define HSV_FILTER_S_MAX	19
#define HSV_FILTER_V_MAX	70

#define RGB_FILTER_R_MIN	100
#define RGB_FILTER_G_MIN	200
#define RGB_FILTER_B_MIN	90

#define RGB_FILTER_R_MAX	170
#define RGB_FILTER_G_MAX	220
#define RGB_FILTER_B_MAX	160

using namespace std;
using namespace cv;

bool stop = false;
CRawImage *droneImage, *cleanimage;
CHeli *heli;
float pitch, roll, yaw, height;
int hover=0;
// Joystick related
SDL_Joystick* m_joystick;
bool useJoystick;
int joypadRoll, joypadPitch, joypadVerticalSpeed, joypadYaw;
bool navigatedWithJoystick, joypadTakeOff, joypadLand, joypadHover;
string ultimo = "init";

int Px;
int Py;
int vR, vG, vB;
int vV, vS, vH;

// Global Mat images
Mat imagenClick, imagenHSV, imagenGrayscale;
Mat imagenThreshold, maskFiltroRGB, maskFiltroHSV, imagenFiltroRGB, imagenFiltroHSV;

// Here we will store points
vector<Point> points;

// Variables de histogramas
Mat b_hist, g_hist, r_hist;
Mat histImageR, histImageG, histImageB;
int hist_w = 516; int hist_h = 516;

/*
 * This method flips horizontally the sourceImage into destinationImage. Because it uses
 * "Mat::at" method, its performance is low (redundant memory access searching for pixels).
 */
void flipImageBasic(const Mat &sourceImage, Mat &destinationImage)
{
	if (destinationImage.empty())
		destinationImage = Mat(sourceImage.rows, sourceImage.cols, sourceImage.type());

	for (int y = 0; y < sourceImage.rows; ++y)
		for (int x = 0; x < sourceImage.cols / 2; ++x)
			for (int i = 0; i < sourceImage.channels(); ++i)
			{
				// i = 0 blue
				// i = 1 green
				// i = 2 red
				//int i = 1;
				destinationImage.at<Vec3b>(y, x)[i] = sourceImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i];
				destinationImage.at<Vec3b>(y, sourceImage.cols - 1 - x)[i] = sourceImage.at<Vec3b>(y, x)[i];
			}
}

// Convert CRawImage to Mat
void rawToMat( Mat &destImage, CRawImage* sourceImage)
{
	uchar *pointerImage = destImage.ptr(0);

	for (int i = 0; i < 240*320; i++)
	{
		pointerImage[3*i] = sourceImage->data[3*i+2];
		pointerImage[3*i+1] = sourceImage->data[3*i+1];
		pointerImage[3*i+2] = sourceImage->data[3*i];
	}
}

void getColorAverages(uchar* destination)
{
	points.push_back(Point(Px, Py));

	if (points.size() > 1)
	{
		int Rmax = 0;
		int Rmin = 256;
		int Gmax = 0;
		int Gmin = 256;
		int Bmax = 0;
		int Bmin = 256;

		double meanR = 0;
		double meanG = 0;
		double meanB = 0;
		double varR = 0;
		double varG = 0;
		double varB = 0;

		int auxCount = 0;

		double auxR = 0;
		double auxG = 0;
		double auxB = 0;

		int x0 = points[0].x;
		int x1 = points[1].x;
		int y0 = points[0].y;
		int y1 = points[1].y;

		cout << x0 << " " << x1 << " " << y0 << " " << y1 << "\n";

		for (int i = x0; i < x1; i++)
			for (int j = y0; j < y1; j++)
			{
				destination = (uchar*)imagenClick.ptr<uchar>(j);
				vB = destination[i * 3];
				vG = destination[i * 3 + 1];
				vR = destination[i * 3 + 2];

				meanR += vR;
				meanG += vG;
				meanB += vB;

				auxR += vR*vR;
				auxG += vG*vG;
				auxB += vB*vB;

				if (vB > Bmax)
					Bmax = vB;
				if (vB < Bmin)
					Bmin = vB;

				if (vR > Rmax)
					Rmax = vR;
				if (vR < Rmin)
					Rmin = vR;

				if (vG > Gmax)
					Gmax = vG;
				if (vG < Gmin)
					Gmin = vG;

				auxCount++;
			}

		//markMeanPoints(meanR, varR, meanG, varG, meanB, varB);

		points.clear();

		if (auxCount != 0)
		{


			meanR = meanR / auxCount;
			meanG = meanG / auxCount;
			meanB = meanB / auxCount;

			varR = auxR / auxCount - meanR*meanR;
			varG = auxG / auxCount - meanG*meanG;
			varB = auxB / auxCount - meanB*meanB;


			printf("Rmin: %d Rmax: %d Gmin: %d Gmax: %d Bmin: %d Bmax: %d\n\n", Rmin, Rmax, Gmin, Gmax, Bmin, Bmax);

			printf("Rmean: %f Rvar: %f Gmean: %f Gvar: %f Bmean: %f Bvar: %f\n\n", meanR, varR, meanG, varG, meanB, varB);

		}

	}
} // end getColorAverages

// handles mouse events in the click window
void mouseCoordinatesExampleCallback(int event, int x, int y, int flags, void* param)
{
    uchar* destination;
    switch (event)
    {
        case CV_EVENT_LBUTTONDOWN:
            Px=x;
            Py=y;
            destination = (uchar*) imagenClick.ptr<uchar>(Py);
            vB=destination[Px * 3];
            vG=destination[Px*3+1];
            vR=destination[Px*3+2];

						destination = (uchar*) imagenHSV.ptr<uchar>(Py);
						vH=destination[Px * 3];
            vS=destination[Px*3+1];
            vV=destination[Px*3+2];

		        /*  Draw a point */
		        //points.push_back(Point(Px, Py));

            break;
        case CV_EVENT_MOUSEMOVE:
            break;
        case CV_EVENT_LBUTTONUP:
            break;
        case CV_EVENT_RBUTTONDOWN:
        //flag=!flag;
            break;

    }
}

void getRGBHistogram(Mat src)
{
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	//int hist_w = 512; int hist_h = 400;
	//int bin_w = cvRound((double)hist_w / histSize);

	histImageR = Mat::zeros(hist_h, hist_w, CV_8UC3);
	histImageG = Mat::zeros(hist_h, hist_w, CV_8UC3);
	histImageB = Mat::zeros(hist_h, hist_w, CV_8UC3);

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImageB.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImageG.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImageR.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{

		line(histImageR, Point(i * 2, hist_h), Point(i * 2, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0,255), 2, 8, 0);
		line(histImageR, Point(i*2, 0), Point(i*2, 40), Scalar(0, 0, i), 2, 8, 0);

		line(histImageG, Point(i * 2, hist_h), Point(i * 2, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);
		line(histImageG, Point(i * 2, 0), Point(i * 2, 40), Scalar(0, i, 0), 2, 8, 0);

		line(histImageB, Point(i * 2, hist_h), Point(i * 2, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);
		line(histImageB, Point(i * 2, 0), Point(i * 2, 40), Scalar(i, 0, 0), 2, 8, 0);

	}

	// Draw marks
	line(histImageR, Point(vR *2, hist_h), Point(vR * 2, hist_h - cvRound(r_hist.at<float>(vR))), Scalar(255, 255, 255), 2, 8, 0);
	line(histImageG, Point(vG*2, hist_h), Point(vG*2, hist_h - cvRound(g_hist.at<float>(vG))), Scalar(255, 255, 255), 2, 8, 0);
	line(histImageB, Point(vB *2, hist_h), Point(vB*2, hist_h - cvRound(b_hist.at<float>(vB))), Scalar(255, 255, 255), 2, 8, 0);


	/// Display
	imshow("Hist R", histImageR);
	imshow("Hist G", histImageG);
	imshow("Hist B", histImageB);
}	// end histogramas

void displayConsoleData()
{
	// Clear the console
	printf("\033[2J\033[1;1H");
	// prints the drone telemetric data, helidata struct contains drone angles, speeds and battery status
	printf("===================== Parrot Basic Example =====================\n\n");
	fprintf(stdout, "Angles  : %.2lf %.2lf %.2lf \n", helidata.phi, helidata.psi, helidata.theta);
	fprintf(stdout, "Speeds  : %.2lf %.2lf %.2lf \n", helidata.vx, helidata.vy, helidata.vz);
	fprintf(stdout, "Battery : %.0lf \n", helidata.battery);
	fprintf(stdout, "Hover   : %d \n", hover);
	fprintf(stdout, "Joypad  : %d \n", useJoystick ? 1 : 0);
	fprintf(stdout, "  Roll    : %d \n", joypadRoll);
	fprintf(stdout, "  Pitch   : %d \n", joypadPitch);
	fprintf(stdout, "  Yaw     : %d \n", joypadYaw);
	fprintf(stdout, "  V.S.    : %d \n", joypadVerticalSpeed);
	fprintf(stdout, "  TakeOff : %d \n", joypadTakeOff);
	fprintf(stdout, "  Land    : %d \n", joypadLand);
	fprintf(stdout, "Navigating with Joystick: %d \n", navigatedWithJoystick ? 1 : 0);
	cout<<"Pos X: "<<Px<<" Pos Y: "<<Py<<" Valor RGB: ("<<vR<<","<<vG<<","<<vB<<")"<<endl;
	cout<<"Pos X: "<<Px<<" Pos Y: "<<Py<<" Valor HSV: ("<<vH<<","<<vS<<","<<vV<<")"<<endl;
}

// Polls the joystick controller for changes and commands them to drone
void pollJoystick()
{
	// Reading of events
	if (useJoystick)
	{
		SDL_Event event;
	  SDL_PollEvent(&event);

	  joypadRoll = SDL_JoystickGetAxis(m_joystick, 2);
	  joypadPitch = SDL_JoystickGetAxis(m_joystick, 3);
	  joypadVerticalSpeed = SDL_JoystickGetAxis(m_joystick, 1);
	  joypadYaw = SDL_JoystickGetAxis(m_joystick, 0);
	  joypadTakeOff = SDL_JoystickGetButton(m_joystick, 1);
	  joypadLand = SDL_JoystickGetButton(m_joystick, 2);
	  joypadHover = SDL_JoystickGetButton(m_joystick, 0);
	}

	// Writing takeoff/land commands to drone
	if (joypadTakeOff) {
	  heli->takeoff();
	  }
	if (joypadLand) {
	  heli->land();
	  }
	  //hover = joypadHover ? 1 : 0;

	//setting the drone angles
	if (joypadRoll != 0 || joypadPitch != 0 || joypadVerticalSpeed != 0 || joypadYaw != 0)
		{
				heli->setAngles(joypadPitch, joypadRoll, joypadYaw, joypadVerticalSpeed, hover);
				navigatedWithJoystick = true;
		}
		else
		{
				heli->setAngles(pitch, roll, yaw, height, hover);
				navigatedWithJoystick = false;
		}

} // end of pollJoystick

void pollKeyboard()	// Polls the keyboard for events, waits for KEYBOARD_DELAY milliseconds
{
	char key = waitKey(KEYBOARD_DELAY);
	switch (key) {
		case 'a': yaw = -20000.0; break;
		case 'd': yaw = 20000.0; break;
		case 'w': height = -20000.0; break;
		case 's': height = 20000.0; break;
		case 'q': heli->takeoff(); break;
		case 'e': heli->land(); break;
		case 'z': heli->switchCamera(0); break;
		case 'x': heli->switchCamera(1); break;
		case 'c': heli->switchCamera(2); break;
		case 'v': heli->switchCamera(3); break;
		case 'j': roll = -20000.0; break;
		case 'l': roll = 20000.0; break;
		case 'i': pitch = -20000.0; break;
		case 'k': pitch = 20000.0; break;
		case 'h': hover = (hover + 1) % 2; break;
		case 27: stop = true; break;
		default: pitch = roll = yaw = height = 0.0;
		}
}

void setup()
{
	//establishing connection with the quadcopter
	heli = new CHeli();

	// Initial values for control
	pitch = roll = yaw = height = 0.0;
	joypadPitch = joypadRoll = joypadYaw = joypadVerticalSpeed = 0.0;

	// Initialize joystick
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK);
	useJoystick = SDL_NumJoysticks() > 0;
	if (useJoystick)
	{
			SDL_JoystickClose(m_joystick);
			m_joystick = SDL_JoystickOpen(0);
	}

	/*
		OpenCV image objects and windows declarations
	*/

	namedWindow("Click");
	namedWindow("Hist R", CV_WINDOW_AUTOSIZE);
	namedWindow("Hist G", CV_WINDOW_AUTOSIZE);
	namedWindow("Hist B", CV_WINDOW_AUTOSIZE);

	droneImage = new CRawImage(320,240); //this class holds the image from the drone
	setMouseCallback("Click", mouseCoordinatesExampleCallback);
}

int main(int argc,char* argv[])
{
		Mat currentImage = Mat(240, 320, CV_8UC3); // Destionation OpenCV Mat

		setup();	// joystick, drone and image initializations

		/***********************************
								MAIN LOOP
		************************************/
    while (stop == false)
    {
			// Read controls
			pollJoystick();
			pollKeyboard();

    	displayConsoleData();	// print telemetric drone and image info through console

			heli->renewImage(droneImage); // image is captured
			rawToMat(currentImage, droneImage); // Copy to OpenCV Mat
			//rawToMat(imagenClick, droneImage);

			currentImage =  imread("/home/alan/Pictures/tv.jpg", 1);
			imagenClick =  imread("/home/alan/Pictures/tv.jpg", 1);


			getRGBHistogram(currentImage);

			// Conversiones de espacios de color
			cvtColor(currentImage, imagenGrayscale, CV_BGR2GRAY);
			cvtColor(currentImage, imagenHSV, CV_BGR2HSV);
			threshold(imagenGrayscale, imagenThreshold, 150, 255, THRESH_BINARY);

			// filtro de color RGB Scalar(B,G,R)
			inRange(imagenClick, Scalar(RGB_FILTER_B_MIN, RGB_FILTER_G_MIN, RGB_FILTER_R_MIN), Scalar(RGB_FILTER_B_MAX, RGB_FILTER_G_MAX, RGB_FILTER_R_MAX), maskFiltroRGB);
			bitwise_and(imagenClick, imagenClick, imagenFiltroRGB, maskFiltroRGB);

			// filtro de color HSV Scalar(V,S,H)
			inRange(imagenClick, Scalar(HSV_FILTER_V_MIN, HSV_FILTER_S_MIN, HSV_FILTER_H_MIN), Scalar(HSV_FILTER_V_MAX, HSV_FILTER_S_MAX, HSV_FILTER_H_MAX), maskFiltroHSV);
			bitwise_and(imagenClick, imagenClick, imagenFiltroRGB, maskFiltroRGB);

			// Show images
			//imshow("Original", currentImage);
			//imshow("ParrotCam", currentImage);
    	imshow("Click", imagenClick);
			imshow("Grayscale", imagenGrayscale);
			imshow("Threshold", imagenThreshold);
			imshow("Filtro RGB", imagenFiltroRGB);


        usleep(15000);
	}

	heli->land();
    SDL_JoystickClose(m_joystick);
    delete heli;
	delete droneImage;
	return 0;
}
