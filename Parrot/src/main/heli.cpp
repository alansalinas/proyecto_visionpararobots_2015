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
#include <math.h>
#include <queue>
#include <time.h>
#include <exception>
#include <vector>

#define KEYBOARD_DELAY 5
#define SLEEP_DELAY	15000
#define PI 3.14159265;


#define Trian1_min 0.182528
#define Trian1_max 0.191274
#define Trian2_min 0.0000007278
#define Trian2_max 0.0005193

#define R1_min 0.40
#define R1_max 1.2195
#define R2_min 0.13
#define R2_max 1.45908

#define Tach1_min 0.20
#define Tach1_max 0.246809
#define Tach2_min 0.014
#define Tach2_max 0.0331681

#define C1_min 0.159193
#define C1_max 0.159663
#define C2_min 0.000000160229
#define C2_max 0.000151656

using namespace std;
using namespace cv;

const string trackbarWindowName = "Trackbars";

typedef struct regiones_struct
{
	int id;
	int tipo;
	string accion;
	double area;
	double phi1;
	double phi2;
	double theta;
} regionStruct;

/*
	Global variable declarations
*/
stringstream texto;

int H_MIN = 34;
int H_MAX = 138;
int S_MIN = 79;
int S_MAX = 206;
int V_MIN = 0;
int V_MAX = 149;

vector<regionStruct> vectorRegiones;

RNG rng(12345);
int cont = 1;

bool stop = false;
bool freezeImage = false;
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
int vR, vG, vB;	// variables para modelo RGB
int vV, vS, vH;	// variables para modelo HSV
int vY, vCr, vCb;	// variables para modelo Y Cr Cb

double meanH = 0;
double meanS = 0;
double meanV = 0;
double varH = 0;
double varS = 0;
double varV = 0;

// Global Mat images
Mat imagenClick, imagenHSV, imagenGrayscale, imagenYCrCb;
Mat imagenThreshold, maskFiltroRGB, maskFiltroHSV, imagenFiltroRGB, imagenFiltroHSV;
Mat maskFiltroYCrCb, imagenFiltroYCrCb;

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
void filtroRuido();

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

						destination = (uchar*) imagenYCrCb.ptr<uchar>(Py);
						vY=destination[Px * 3];
            vCr=destination[Px*3+1];
            vCb=destination[Px*3+2];

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
	cout<<"Pos X: "<<Px<<" Pos Y: "<<Py<<" Valor Y Cr Cb: ("<<vY<<","<<vCr<<","<<vCb<<")"<<endl;
	cout << endl << "Numero de regiones:" << vectorRegiones.size() << endl;

	for (int k = 0; k < vectorRegiones.size(); k++)
	{
		cout << "tipo: " << vectorRegiones[k].tipo << endl;
		cout << "phi1: " << vectorRegiones[k].phi1 << endl;
		cout << "phi2: " << vectorRegiones[k].phi2 << endl;
		cout << "area: " << vectorRegiones[k].area << endl;
		cout << "theta: " << vectorRegiones[k].theta << endl;
		cout << "accion: " << vectorRegiones[k].accion << endl << endl;
	}

}

// Polls the joystick controller for changes and commands them to drone
void pollJoystick()
{
	// Reading of events
	if (useJoystick)
	{
		hover = 0;
		SDL_Event event;
	  SDL_PollEvent(&event);

	  joypadRoll = SDL_JoystickGetAxis(m_joystick, 3);
	  joypadPitch = SDL_JoystickGetAxis(m_joystick, 4);
	  joypadVerticalSpeed = SDL_JoystickGetAxis(m_joystick, 1);
	  joypadYaw = SDL_JoystickGetAxis(m_joystick, 0);
	  joypadTakeOff = SDL_JoystickGetButton(m_joystick, 0);
	  joypadLand = SDL_JoystickGetButton(m_joystick, 2);
	  joypadHover = SDL_JoystickGetButton(m_joystick, 3);
	}

	// Writing takeoff/land commands to drone
	if (joypadTakeOff) {
	  heli->takeoff();
	  }
	if (joypadLand) {
	  heli->land();
	  }
	  //hover = joypadHover ? 1 : 0;

} // end of pollJoystick

void setDroneAngles()
{
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
}

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
		case 'm': filtroRuido(); break;
		case '.': freezeImage = !freezeImage; break;
		case 27: stop = true; break;
		default: pitch = roll = yaw = height = 0.0;
		}
}

void adelante()
{
	//heli->setAngles(pitch, roll, yaw, height, hover);
	heli->setAngles(-10000.0, 0.0, 0.0, 0.0, 0.0);
	usleep(1000000);
	cout<<"pitch"<<endl;
}

void atras()
{
	//heli->setAngles(pitch, roll, yaw, height, hover);
	heli->setAngles(10000.0, 0.0, 0.0, 0.0, 0.0);
	usleep(1000000);
	cout<<"pitch"<<endl;
}

void izquierda()
{
	//heli->setAngles(pitch, roll, yaw, height, hover);
	heli->setAngles(0.0, -10000.0, 0.0, 0.0, 0.0);
	usleep(1000000);
	cout<<"pitch"<<endl;
}

void derecha()
{
	//heli->setAngles(pitch, roll, yaw, height, hover);
	heli->setAngles(0.0, 10000.0, 0.0, 0.0, 0.0);
	usleep(1000000);
}

void arriba()
{
	//heli->setAngles(pitch, roll, yaw, height, hover);
	heli->setAngles(0.0, 0.0, 0.0, -15000.0, 0.0);
	usleep(1000000);
}

void abajo()
{
	//heli->setAngles(pitch, roll, yaw, height, hover);
	heli->setAngles(0.0, 0.0, 0.0, 15000.0, 0.0);
	usleep(1000000);
}

void on_trackbar( int, void* )
{//This function gets called whenever a
    // trackbar position is changed

}

string intToString(int number){


    std::stringstream ss;
    ss << number;
    return ss.str();
}


void createTrackbars(){
    //create window for trackbars
    namedWindow(trackbarWindowName,0);


    //create memory to store trackbar name on window
    char TrackbarName[50];
    printf( TrackbarName, "H_MIN", H_MIN);
    printf( TrackbarName, "H_MAX", H_MAX);
    printf( TrackbarName, "S_MIN", S_MIN);
    printf( TrackbarName, "S_MAX", S_MAX);
    printf( TrackbarName, "V_MIN", V_MIN);
    printf( TrackbarName, "V_MAX", V_MAX);


    //create trackbars and insert them into window
    createTrackbar( "H_MIN", trackbarWindowName, &H_MIN, H_MAX, on_trackbar );
    createTrackbar( "H_MAX", trackbarWindowName, &H_MAX, H_MAX, on_trackbar );
    createTrackbar( "S_MIN", trackbarWindowName, &S_MIN, S_MAX, on_trackbar );
    createTrackbar( "S_MAX", trackbarWindowName, &S_MAX, S_MAX, on_trackbar );
    createTrackbar( "V_MIN", trackbarWindowName, &V_MIN, V_MAX, on_trackbar );
    createTrackbar( "V_MAX", trackbarWindowName, &V_MAX, V_MAX, on_trackbar );


}

CvPoint detecta(Mat imagen)
{
	int x = rand() % imagen.cols;
	int y = rand() % imagen.rows;
	CvPoint cvp;
	cvp.x = x;
	cvp.y = y;
	return cvp;
}


void explora(CvPoint p, Mat img, Mat regiones, int regionesActual)
{
	CvPoint actual = p;
	regiones.at<uchar>(actual) = regionesActual;

	texto.str("hola");

	int x = actual.x;
	int y = actual.y;
	queue<CvPoint> lista;
	lista.push(p);

	double area = 0;

	//momentos estadisticos
	double m10 = 0;
	double m01 = 0;

	double m11 = 0;
	double m20 = 0;
	double m02 = 0;

	//momentos centralizados
	double u20 = 0;
	double u02 = 0;
	double u11 = 0;

	double up20 = 0;
	double up02 = 0;
	double up11 = 0;

	double thetaInc = 0;
	double phi1 = 0;
	double phi2 = 0;

	while (!lista.empty())
	{
		area++;
		actual = lista.front();
		lista.pop();
		// Inicia expansion
		x = actual.x;
		y = actual.y;
		//Momentos estadisticos primer orden
		m10 += x;
		m01 += y;
		//momentos estadisticos segundo orden
		m11 += x*y;
		m20 += x*x;
		m02 += y*y;
		//arriba, izquierda, abajo, derecha
		Point v[4] = { Point(x, y + 1), Point(x - 1, y), Point(x, y - 1), Point(x + 1, y) };

		for (int i = 0; i < 4; i++)
		{
			if (v[i].x < 0)
				v[i].x = 0;
			if (v[i].x >= regiones.cols)
				v[i].x = regiones.cols - 1;

			if (v[i].y < 0)
				v[i].y = 0;
			if (v[i].y >= regiones.rows)
				v[i].y = regiones.rows - 1;

			int v1;

			v1 = img.at<uchar>(v[i]);
			if (v1 == 255 && regiones.at<uchar>(v[i]) == 0)
			{
				regiones.at<uchar>(v[i]) = regionesActual;
				lista.push(v[i]);
			}
		}
	}

	//centroides
	double xMedia = m10 / area;
	double yMedia = m01 / area;
	Point centro = Point(xMedia, yMedia);

	//Momentos centralizados
	u11 = m11 - yMedia*m10;
	u20 = m20 - xMedia*m10;
	u02 = m02 - yMedia*m01;

	//Momentos primo centralizados
	up11 = u11 / area;
	up20 = u20 / area;
	up02 = u02 / area;

	//Momentos invariantes
	double n20 = u20 / pow(area, 2);
	double n02 = u02 / pow(area, 2);
	double n11 = u11 / pow(area, 2);

	phi1 = n20 + n02;
	phi2 = pow((n20 - n02), 2) + (4 * pow(n11, 2));

	//theta de inclinacion
	//thetaInc = 0.5 * atan((2.0*up11) / (up20-up02));

	thetaInc = 0.5 * atan2((2.0*u11), (u20 - u02));
	double theta = thetaInc * (180 / 3.14159265);
	//desplegar resultados
	//cout << "m01: " << m10 << " m10: " << m01 << endl;
	//cout << "u11: " << u11 << endl;
	//cout << "n20: " << n20 << endl;
	//cout << "n02: " << n02 << endl;

	if (area > 250)
	{
		regionStruct reg;

		cout << "Area Region " << regionesActual << ": " << area << endl;
		cout << "phi1: " << phi1 << endl;
		cout << "phi2: " << phi2 << endl;
		cout << "theta: " << theta << endl;

		double minMarg = 0.90;
		double maxMarg = 1.10;
		if (phi1 > Trian1_min*minMarg && phi1 < Trian1_max*maxMarg && phi2 > Trian2_min*minMarg && phi2 < Trian2_max*maxMarg)
			reg.tipo = 0;//cout << "traingulo" << endl;
		if (phi1 > C1_min*minMarg && phi1 < C1_max*maxMarg && phi2 > C2_min*minMarg && phi2 < C2_max*maxMarg)
			reg.tipo = 1;//cout << "circulo" << endl;
		if (phi1 > R1_min*minMarg && phi1 < R1_max*maxMarg && phi2 > R2_min*minMarg && phi2 < R2_max*maxMarg)
		{
			cout << "rectangulo";
			reg.tipo = 2;
			if (theta > 10)
			{
				cout << " baja" << endl;
				reg.accion = "baja";
			}
			else if (theta < -10)
			{
				cout << " sube" << endl;
				reg.accion = "sube";
			}
			else
			{
				cout << " mismo nivel" << endl;
				reg.accion = "mismo nivel";
			}
		}
		if (phi1 > Tach1_min*minMarg && phi1 < Tach1_max*maxMarg && phi2 > Tach2_min*minMarg && phi2 < Tach2_max*maxMarg)
		{
			cout << "tacha";
			reg.tipo = 3;
			if (theta > 10)
			{
				cout << " baja" << endl;
				reg.accion = "baja";
			}
			else if (theta < -10)
			{
				cout << " sube" << endl;
				reg.accion = "sube";
			}
			else
			{
				cout << " mismo nivel" << endl;
				reg.accion = "sube";
			}
		}
		cout << "__________________________________________________" << endl;

		// generate structure and add to vector
		reg.id = regionesActual;
		reg.area = area;
		reg.phi1 = phi1;
		reg.phi2 = phi2;
		reg.theta = theta;

		vectorRegiones.push_back(reg);


		int val = sqrt(u20)/100;
		int val2 = sqrt(u02)/100;
		//circle(regiones, cvPoint(xMedia, yMedia),1, 0, 2, 8, 0);
		//line(regiones, (centro-Point(val*cos(thetaInc), val*sin(thetaInc))),
		//              (centro+Point(val*cos(thetaInc), val*sin(thetaInc))), 127, 1);

		line(regiones, (centro - Point(val*cos(thetaInc), val*sin(thetaInc))),
			(centro + Point(val*cos(thetaInc), val*sin(thetaInc))), 127, 1);

		line(regiones, (centro - Point(val2*cos(1.570796327 + thetaInc), val2*sin(1.570796327 + thetaInc))),
			(centro + Point(val2*cos(1.570796327 + thetaInc), val2*sin(1.570796327 + thetaInc))), 127, 1);

		//line(regiones, (Point(xMedia,yMedia)+Point(val*cos(theta2), val*sin(theta2))),
		//                (Point(xMedia, yMedia)-Point(val*cos(theta2), val*sin(theta2))), 160, 1);

	}

	//cout << "theta " << cont << ": " << thetaInc << endl;
	cont++;


}

Mat segmenta(Mat imgOriginal)
{


	Mat regiones = Mat::zeros(imgOriginal.rows, imgOriginal.cols, CV_8UC1);


	if (imgOriginal.channels() != 1)
	{
		cout << "Error: La imagen no est� en blanco y negro" << endl;
		return regiones;
	}

	int regionesActual = 1;
	int pixelesImagen = imgOriginal.rows*imgOriginal.cols;

	for (int i = 0; i < pixelesImagen*0.5; i++)
	{

		if (regionesActual > 255)
			break;

		CvPoint p = detecta(imgOriginal);

		int valor = 255;

		valor = imgOriginal.at<uchar>(p);

		if (valor == 255 && regiones.at<uchar>(p) == 0)
		{

			explora(p, imgOriginal, regiones, regionesActual);
			regionesActual += 1;
		}
	}

	return regiones;
}

void filtroRuido()
{
	blur(imagenFiltroHSV, imagenFiltroHSV, Size(5, 5), Point(-1, -1), 4);

	vectorRegiones.clear();
	// Erosion y luego dilatacion
	Mat element;
	int size = 3;
	int type = MORPH_RECT;

	element = getStructuringElement(type,Size(2 * size + 1, 2 * size + 1),	Point(size, size));
	morphologyEx(imagenFiltroHSV, imagenFiltroHSV, MORPH_OPEN ,element);
	morphologyEx(imagenFiltroHSV, imagenFiltroHSV, MORPH_CLOSE, element);

	cvtColor(imagenFiltroHSV, imagenGrayscale, CV_HSV2BGR);
	cvtColor(imagenGrayscale, imagenGrayscale, CV_BGR2GRAY);

	//	cvtColor(currentImage, imagenGrayscale, CV_BGR2GRAY);
	threshold(imagenGrayscale, imagenThreshold, 10, 255, THRESH_BINARY);
	imshow("entrada", imagenThreshold);
	bitwise_not(segmenta(imagenThreshold), imagenThreshold);
	imshow("salida", imagenThreshold);

	for (int k = 0; k < vectorRegiones.size(); k++)
	{
			if(vectorRegiones[k].tipo > 3)
				break;

			switch (vectorRegiones[k].tipo)
			{
			case 0: adelante(); break;
			case 1: atras(); break;
			case 2:
				if (vectorRegiones[k].theta > 20)
				abajo();
				else if(vectorRegiones[k].theta < -20)
				arriba();

				izquierda();
				break;
			case 3:
				if (vectorRegiones[k].theta > 20)
				abajo();
				else if(vectorRegiones[k].theta < -20)
				arriba();

				derecha();
			break;
			default: break;
			}

	}

}

void setup()
{
	srand(time(0));

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
	//create slider bars for HSV filtering
	createTrackbars();

	droneImage = new CRawImage(320,240); //this class holds the image from the drone
	//setMouseCallback("Click", mouseCoordinatesExampleCallback);
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
			setDroneAngles();

    	displayConsoleData();	// print telemetric drone and image info through console

			if (!freezeImage)
			{
			heli->renewImage(droneImage); // image is captured
			rawToMat(currentImage, droneImage); // Copy to OpenCV Mat
			//rawToMat(imagenClick, droneImage);
			}

			//currentImage =  imread("/home/alan/Pictures/naranja.jpeg", 1);
			//imagenClick =  imread("/home/alan/Pictures/tv.jpg", 1);

			imagenClick = currentImage;
			//getRGBHistogram(currentImage);

			imagenFiltroRGB = Mat::zeros(240, 320, CV_8UC3);
			imagenFiltroHSV = Mat::zeros(240, 320, CV_8UC3);

			// Conversiones de espacios de color
			cvtColor(currentImage, imagenGrayscale, CV_BGR2GRAY);
			cvtColor(currentImage, imagenHSV, CV_BGR2HSV);
			//threshold(imagenGrayscale, imagenThreshold, 150, 255, THRESH_BINARY);

			// filtro de color HSV Scalar(H,S,V)
			inRange(imagenHSV, Scalar(H_MIN, S_MIN, V_MIN), Scalar(H_MAX, S_MAX, V_MAX), maskFiltroHSV);
			bitwise_and(imagenHSV, imagenHSV, imagenFiltroHSV, maskFiltroHSV);

			// Show images
			//imshow("Original", currentImage);
			//imshow("ParrotCam", currentImage);
    	imshow("Click", imagenClick);
			imshow("Filtro HSV", imagenFiltroHSV);

      usleep(SLEEP_DELAY);
	}

	heli->land();
    SDL_JoystickClose(m_joystick);
    delete heli;
	delete droneImage;
	vectorRegiones.clear();
	return 0;
}
