#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>

#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <smmintrin.h>


using namespace std;

#define OMP_NUM_THREADS 256

inline double square(double x)
{
	return x*x;
}


class SLIC
{
public:
	SLIC();
	virtual ~SLIC();

	void calculate_super_pixel(int sz,int K,int N,double* seeds,int *belong);

	//============================================================================
	// Superpixel segmentation for a given number of superpixels	输入
	//============================================================================
	void PerformSLICO_ForGivenK(
		const unsigned int *ubuff, //Each 32 bit unsigned int contains ARGB pixel values.
		const int width,
		const int height,
		int *klabels,
		int &numlabels,
		const int &K,
		const double &m);

	//============================================================================
	// Save superpixel labels to pgm in raster scan order	保存聚类标记
	//============================================================================
	void SaveSuperpixelLabels2PPM(
		char *filename,
		int *labels,
		const int width,
		const int height);

private:
	//============================================================================
	// Magic SLIC. No need to set M (compactness factor) and S (step size). #两种模式
	// SLICO (SLIC Zero) varies only M dynamicaly, not S.
	//============================================================================
	void PerformSuperpixelSegmentation_VariableSandM(
		double* seeds,
		int *klabels,
		const int numk,
		const int &STEP,
		const int &NUMITR);

	// 计算单个点的梯度，对于一个初始seed计算周围八个点即可
	double DetectLABPixelEdge(
		const int &i);
	//============================================================================
	// Pick seeds for superpixels when number of superpixels is input.
	//============================================================================
	void GetLABXYSeeds_ForGivenK(
		double* seeds,
		int& numk,
		const int &K,
		const bool &perturbseeds);


	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int *&ubuff,
		double *&labvec);

	//============================================================================
	// Post-processing of SLIC segmentation, to avoid stray labels.
	//============================================================================
	void EnforceLabelConnectivity(
		const int *labels,
		const int &width,
		const int &height,
		int *nlabels,	//input labels that need to be corrected to remove stray labels
		int &numlabels, //the number of labels changes in the end if segments are removed
		const int &K);	//the number of superpixels desired by the user

private:
	int m_width;
	int m_height;
	int m_depth;

	// double *m_lvec;
	// double *m_avec;
	// double *m_bvec;
	double *m_labvec;
};


typedef chrono::high_resolution_clock Clock;

int numThreads = 256;

// For superpixels
const int dx4[4] = {-1, 0, 1, 0};
const int dy4[4] = {0, -1, 0, 1};
//const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = {-1, 0, 1, 0, -1, 1, 1, -1, 0, 0};
const int dy10[10] = {0, -1, 0, 1, -1, -1, 1, 1, 0, 0};
const int dz10[10] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 1};

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC()
{
	// m_lvec = NULL;
	// m_avec = NULL;
	// m_bvec = NULL;
	m_labvec = NULL;
}

SLIC::~SLIC()
{
	// if (m_lvec) _mm_free(m_lvec);
	// if (m_avec) _mm_free(m_avec);
	// if (m_bvec) _mm_free(m_bvec);
	// if (m_lvec) delete[] m_lvec;
	// if (m_avec) delete[] m_avec;
	// if (m_bvec) delete[] m_bvec;
	if (m_labvec) delete[] m_labvec;
}


void SLIC::DoRGBtoLABConversion(
	const unsigned int *&ubuff,
	double *&labvec)
{
	int sz = m_width * m_height;
	labvec = new double[3*sz];
	// m_lvec = new double[sz];
	// m_avec = new double[sz];
	// m_bvec = new double[sz];
	// m_lvec = (double*)_mm_malloc(sz * sizeof(double), 256);
	// m_avec = (double*)_mm_malloc(sz * sizeof(double), 256);
	// m_bvec = (double*)_mm_malloc(sz * sizeof(double), 256);
    //  建立查询表 利用向量化 加速计算
    double* tableRGB = new double[256]; //记得delete[]掉
    //#pragma omp simd 我觉得这里相比多发还是并行更好，因为浮点数的运算是64位，一个寄存器也就64位
    #pragma omp parallel for
    for (int i = 0; i < 11; ++i) {
        tableRGB[i] = i * (1.0 / 3294.6);
    }
    //#pragma omp simd
    #pragma omp parallel for
    for (int i = 11; i < 256; ++i) {
        tableRGB[i] = pow((i * (1.0 / 269.025) + 0.0521327014218009), 2.4);
    }
	#pragma omp parallel for
	for (int j = 0; j < sz; j++)
	{
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >> 8) & 0xFF;
		int b = (ubuff[j]) & 0xFF;
		double* labval = labvec + 3*j;
		
		//RGB2XYZ(sR, sG, sB, X, Y, Z);
		//直接查询
		double r0 = tableRGB[r];
		double g0 = tableRGB[g];
		double b0 = tableRGB[b];
		//转化为XYZ
		double X = r0 * 0.4124564 + g0 * 0.3575761 + b0 * 0.1804375;
		double Y = r0 * 0.2126729 + g0 * 0.7151522 + b0 * 0.0721750;
		double Z = r0 * 0.0193339 + g0 * 0.1191920 + b0 * 0.9503041;

		double fx, fy, fz;
		fx = X > 0.008417238336 ? cbrt(X * (1.0 / 0.950456)):(8.192982069151272 * X + 0.1379310344827586);
		fy = Y > 0.008856 ? fy = cbrt(Y): (7.787068965517241 * Y + 0.1379310344827586);
		fz = Z > 0.009642005424 ? cbrt(Z * (1.0 / 1.088754)): (7.152275872710678 * Z + 0.1379310344827586);
		//这里通过

		labvec[2] = 116.0 * fy - 16.0;
		labvec[1] = 500.0 * (fx - fy);
		labvec[0] = 200.0 * (fy - fz);
	}
	delete[] tableRGB;
}

// 计算单个点的梯度，对于一个初始seed计算周围八个点即可
double SLIC::DetectLABPixelEdge(
	const int &i)
{
	const double *labvec = m_labvec;
	const int i3 = 3*i;
	const int width3 = m_width*3;
	// double dx = (lvec[i - 1] - lvec[i + 1]) * (lvec[i - 1] - lvec[i + 1]) +
	// 			(avec[i - 1] - avec[i + 1]) * (avec[i - 1] - avec[i + 1]) +
	// 			(bvec[i - 1] - bvec[i + 1]) * (bvec[i - 1] - bvec[i + 1]);

	// double dy = (lvec[i - width] - lvec[i + width]) * (lvec[i - width] - lvec[i + width]) +
	// 			(avec[i - width] - avec[i + width]) * (avec[i - width] - avec[i + width]) +
	// 			(bvec[i - width] - bvec[i + width]) * (bvec[i - width] - bvec[i + width]);
	double dx = square(labvec[i3 - 1] - labvec[i3 + 5]) +
				square(labvec[i3 - 2] - labvec[i3 + 4]) +
				square(labvec[i3 - 3] - labvec[i3 + 3]);

	double dy = square(labvec[i3 - width3 + 2] - labvec[i3 + width3 + 2]) +
				square(labvec[i3 - width3 + 1] - labvec[i3 + width3 + 1])  +
				square(labvec[i3 - width3 + 0] - labvec[i3 + width3 + 0]) ;
	return dx + dy;
}


//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYSeeds_ForGivenK(
	double* seeds,
	int& numk,
	const int &K,
	const bool &perturbseeds)
{

	int sz = m_width * m_height;
	double step = sqrt(double(sz) / double(K));
	int T = step;
	int xoff = step / 2;
	int yoff = step / 2;
	double* labvec = m_labvec;
	//  int n(0);
	 int r(0);

	for (int y = 0; y < m_height; y++)
	{
		int Y = y * step + yoff;
		if (Y > m_height - 1)
			break;

		for (int x = 0; x < m_width; x++)
		{
			//int X = x*step + xoff;//square grid
			int X = x * step + (xoff << (r & 0x1)); //hex grid
			if (X > m_width - 1)
				break;

			int i = Y * m_width + X;
			double* seed = seeds + 5*numk;
			double* labval = labvec + 3*i;
			// kseedsl[numk]=m_lvec[i];
			// kseedsa[numk]=m_avec[i];
			// kseedsb[numk]=m_bvec[i];
			// kseedsx[numk]=X;
			// kseedsy[numk]=Y;
			#pragma omp parallel for
			for(int l=0;l<=2;l++)
			{
				seed[l] = labval[l];
			}
			seed[3] = X;
			seed[4] = Y;
			++numk;
			// n++;
		}
		 r++;
	}

	if (perturbseeds)
	{
		const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
		const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};
		#pragma omp parallel for
		for (int n = 0; n < numk; n++)
		{
			int ox = seeds[5*n+3]; //original x
			int oy = seeds[5*n+4]; //original y
			int oind = oy * m_width + ox;
			int storeind = oind;
			double minn = DetectLABPixelEdge(storeind);
			for (int i = 0; i < 8; i++)
			{
				int nx = ox + dx8[i]; //new x
				int ny = oy + dy8[i]; //new y

				if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
				{
					int nind = ny * m_width + nx;
					double test = DetectLABPixelEdge(nind);
					if ( test < minn)
					{
						storeind = nind;
						minn = test;
					}
				}
			}
			if (storeind != oind)
			{
				double* seed = seeds + 5*n;
				double* labval = labvec + 3*storeind;
				#pragma omp parallel for
				for(int l=0;l<=2;l++)
				{
					seed[l] = labval[l];
				}
				seed[3] = storeind % m_width;
				seed[4] = storeind / m_width;
				// kseedsx[n] = storeind % m_width;
				// kseedsy[n] = storeind / m_width;
				// kseedsl[n] = m_lvec[storeind];
				// kseedsa[n] = m_avec[storeind];
				// kseedsb[n] = m_bvec[storeind];
			}
		}
	}
}
void SLIC::calculate_super_pixel(int sz,int K,int N,double* seeds,int *belong) 
{
   const int TIME = 10;//迭代次数
   const int THREAD = 16;
   const int STEP = (int) (sqrt((double) (sz) / (double) (K)) + 2.0);
   int offset = STEP;
   if (STEP < 10) offset = (int) (STEP * 1.5);
	double * lab = m_labvec;
   double inv_xy = 1.0 / (STEP * STEP);
   int width = m_width;
   int height = m_height;

   double max_lab[N] __attribute__((aligned(32)));
   double max_lab_div[N] __attribute__((aligned(32)));

   double sigma[THREAD][N][5] __attribute__((aligned(32)));
   double max_lab_t[THREAD][N] __attribute__((aligned(32)));
   int cluster_size[THREAD][N] __attribute__((aligned(32)));


#pragma omp simd
   for (int i = 0; i < N; ++i) {
       max_lab[i] = 100;
       max_lab_div[i] = 0.01;
  }

   //迭代TIME次
   for (int t = 0; t < TIME; ++t) {
       //初始化清零数据
       memset(cluster_size, 0, N * THREAD * sizeof(int));
       memset(max_lab_t, 0, N * THREAD * sizeof(double));
       memset(sigma, 0, 5 * N * THREAD * sizeof(double));
       //遍历所有seed,计算最近的靠近位置
#pragma omp parallel for default(none) shared(height, width, N, offset, t, seeds, lab, belong, inv_xy, max_lab_t, max_lab_div, sigma, cluster_size) schedule(guided, 10)
       for (int y = 0; y < height; ++y) {
           double dis_vec_y[width] __attribute__((aligned(32)));
           double dis_lab_y[width] __attribute__((aligned(32)));
           memset(dis_vec_y, 127 , width * sizeof(double));
           const int y_index = y * width;
           for (int n = 0; n < N; n++) {
               double *seed = seeds + 5 * n;
               if ((int) (seed[4] - offset) <= y && y < (int) (seed[4] + offset)) {
                   const int x1 = max(0, (int) (seed[3] - offset));
                   const int x2 = min(width, (int) (seed[3] + offset));
                   const double div_lab = max_lab_div[n];
                   for (int x = x1; x < x2; ++x) {
                       int i = y_index + x;
                       dis_lab_y[x] = square(lab[i * 3] - seed[0]) +
                                      square(lab[i * 3 + 1] - seed[1]) +
                                      square(lab[i * 3 + 2] - seed[2]);
                  }
                   for (int x = x1; x < x2; ++x) {
                       int i = y_index + x;
                       double temp_dis_xy = square(x - seed[3]) + square(y - seed[4]);
                       double dist = dis_lab_y[x] * div_lab + temp_dis_xy * inv_xy;
                       if (dist < dis_vec_y[x]) {
                           dis_vec_y[x] = dist;
                           belong[i] = n;
                      }
                  }
              }
          }
           const int thread_num = omp_get_thread_num();
           for (int x = 0; x < width; ++x) {
               int i = width * y + x;
               int k = belong[i];
               if (max_lab_t[thread_num][k] < dis_lab_y[x]) {
                   max_lab_t[thread_num][k] = dis_lab_y[x];
              }
			#pragma omp simd
               for (int j = 0; j < 3; ++j) {
                   sigma[thread_num][k][j] += lab[i * 3 + j];
              }
               sigma[thread_num][k][3] += x;
               sigma[thread_num][k][4] += y;
               cluster_size[thread_num][k]++;
          }
      }

       // 重新计算种子点
       for (int k = 0; k < N; k++) {
           int seed_size = 0;
           double sigma_t[5] __attribute__((aligned(32))) = {0};
           for (int i = 0; i < THREAD; ++i) {
#pragma omp simd
               for (int j = 0; j < 5; ++j) {
                   sigma_t[j] += sigma[i][k][j];
              }
               if (max_lab[k] < max_lab_t[i][k]) {
                   max_lab[k] = max_lab_t[i][k];
              }
               seed_size += cluster_size[i][k];
          }
           if (seed_size == 0) seed_size = 1;
           double inv = 1.0 / seed_size;
           max_lab_div[k] = 1.0 / max_lab[k];
			#pragma omp simd
           for (int i = 0; i < 5; ++i) {
               seeds[5 * k + i] = sigma_t[i] * inv;
          }
      }
  }
}

// void SLIC::PerformSuperpixelSegmentation_VariableSandM(
// 	double* seeds,
// 	int *klabels,
// 	const int numk,
// 	const int &STEP,
// 	const int &NUMITR)
// {
// 	int sz = m_width * m_height;
// 	// const int numk = kseedsl.size();
// 	//double cumerr(99999.9);
// 	int numitr(0);

// 	//----------------
// 	int offset = STEP;
// 	if (STEP < 10)
// 		offset = STEP * 1.5;
// 	//----------------

// 	vector<double> sigmal(numk, 0);
// 	vector<double> sigmaa(numk, 0);
// 	vector<double> sigmab(numk, 0);
// 	vector<double> sigmax(numk, 0);
// 	vector<double> sigmay(numk, 0);
// 	vector<int> clustersize(numk, 0);
// 	vector<double> inv(numk, 0); //to store 1/clustersize[k] values
// 	vector<double> distxy(sz, DBL_MAX);
// 	vector<double> distlab(sz, DBL_MAX);
// 	vector<double> distvec(sz, DBL_MAX);
// 	vector<double> maxlab(numk, 10 * 10);	 //THIS IS THE VARIABLE VALUE OF M, just start with 10
// 	vector<double> maxxy(numk, STEP * STEP); //THIS IS THE VARIABLE VALUE OF M, just start with 10

// 	double invxywt = 1.0 / (STEP * STEP); //NOTE: this is different from how usual SLIC/LKM works

// 	while (numitr < NUMITR)
// 	{
// 		//------
// 		//cumerr = 0;
// 		numitr++;
// 		//------

// 		distvec.assign(sz, DBL_MAX);
// 		for (int n = 0; n < numk; n++)
// 		{
// 			int y1 = max(0, (int)(kseedsy[n] - offset));
// 			int y2 = min(m_height, (int)(kseedsy[n] + offset));
// 			int x1 = max(0, (int)(kseedsx[n] - offset));
// 			int x2 = min(m_width, (int)(kseedsx[n] + offset));

// 			for (int y = y1; y < y2; y++)
// 			{
// 				for (int x = x1; x < x2; x++)
// 				{
// 					int i = y * m_width + x;
// 					//_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0 );

// 					double l = m_lvec[i];
// 					double a = m_avec[i];
// 					double b = m_bvec[i];

// 					distlab[i] = (l - kseedsl[n]) * (l - kseedsl[n]) +
// 								 (a - kseedsa[n]) * (a - kseedsa[n]) +
// 								 (b - kseedsb[n]) * (b - kseedsb[n]);

// 					distxy[i] = (x - kseedsx[n]) * (x - kseedsx[n]) +
// 								(y - kseedsy[n]) * (y - kseedsy[n]);

// 					//------------------------------------------------------------------------
// 					double dist = distlab[i] / maxlab[n] + distxy[i] * invxywt; //only varying m, prettier superpixels
// 					//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
// 					//------------------------------------------------------------------------

// 					if (dist < distvec[i])
// 					{
// 						distvec[i] = dist;
// 						klabels[i] = n;
// 					}
// 				}
// 			}
// 		}
// 		//-----------------------------------------------------------------
// 		// Assign the max color distance for a cluster
// 		//-----------------------------------------------------------------
// 		if (0 == numitr)
// 		{
// 			maxlab.assign(numk, 1);
// 			maxxy.assign(numk, 1);
// 		}
// 		{
// 			for (int i = 0; i < sz; i++)
// 			{
// 				if (maxlab[klabels[i]] < distlab[i])
// 					maxlab[klabels[i]] = distlab[i];
// 				if (maxxy[klabels[i]] < distxy[i])
// 					maxxy[klabels[i]] = distxy[i];
// 			}
// 		}
// 		//-----------------------------------------------------------------
// 		// Recalculate the centroid and store in the seed values
// 		//-----------------------------------------------------------------
// 		sigmal.assign(numk, 0);
// 		sigmaa.assign(numk, 0);
// 		sigmab.assign(numk, 0);
// 		sigmax.assign(numk, 0);
// 		sigmay.assign(numk, 0);
// 		clustersize.assign(numk, 0);

// 		for (int j = 0; j < sz; j++)
// 		{
// 			int temp = klabels[j];
// 			//_ASSERT(klabels[j] >= 0);
// 			sigmal[klabels[j]] += m_lvec[j];
// 			sigmaa[klabels[j]] += m_avec[j];
// 			sigmab[klabels[j]] += m_bvec[j];
// 			sigmax[klabels[j]] += (j % m_width);
// 			sigmay[klabels[j]] += (j / m_width);

// 			clustersize[klabels[j]]++;
// 		}

// 		{
// 			for (int k = 0; k < numk; k++)
// 			{
// 				//_ASSERT(clustersize[k] > 0);
// 				if (clustersize[k] <= 0)
// 					clustersize[k] = 1;
// 				inv[k] = 1.0 / double(clustersize[k]); //computing inverse now to multiply, than divide later
// 			}
// 		}

// 		{
// 			for (int k = 0; k < numk; k++)
// 			{
// 				kseedsl[k] = sigmal[k] * inv[k];
// 				kseedsa[k] = sigmaa[k] * inv[k];
// 				kseedsb[k] = sigmab[k] * inv[k];
// 				kseedsx[k] = sigmax[k] * inv[k];
// 				kseedsy[k] = sigmay[k] * inv[k];
// 			}
// 		}
// 	}
// }


//===========================================================================
///	SaveSuperpixelLabels2PGM
///
///	Save labels to PGM in raster scan order.
//===========================================================================
void SLIC::SaveSuperpixelLabels2PPM(
	char *filename,
	int *labels,
	const int width,
	const int height)
{
	FILE *fp;
	char header[20];

	fp = fopen(filename, "wb");

	// write the PPM header info, such as type, width, height and maximum
	fprintf(fp, "P6\n%d %d\n255\n", width, height);

	// write the RGB data
	unsigned char *rgb = new unsigned char[(width) * (height)*3];
	int k = 0;
	unsigned char c = 0;
	for (int i = 0; i < (height); i++)
	{
		for (int j = 0; j < (width); j++)
		{
			c = (unsigned char)(labels[k]);
			rgb[i * (width)*3 + j * 3 + 2] = labels[k] >> 16 & 0xff; // r
			rgb[i * (width)*3 + j * 3 + 1] = labels[k] >> 8 & 0xff;	 // g
			rgb[i * (width)*3 + j * 3 + 0] = labels[k] & 0xff;		 // b

			// rgb[i*(width) + j + 0] = c;
			k++;
		}
	}
	fwrite(rgb, width * height * 3, 1, fp);

	delete[] rgb;

	fclose(fp);
}

//===========================================================================
///	EnforceLabelConnectivity
///
///		1. finding an adjacent label for each new component at the start
///		2. if a certain component is too small, assigning the previously found
///		    adjacent label to this component, and not incrementing the label.
//===========================================================================
void SLIC::EnforceLabelConnectivity(
	const int *labels, //input labels that need to be corrected to remove stray labels
	const int &width,
	const int &height,
	int *nlabels,	//new labels
	int &numlabels, //the number of labels changes in the end if segments are removed
	const int &K)	//the number of superpixels desired by the user
{
	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = {-1, 0, 1, 0};
	const int dy4[4] = {0, -1, 0, 1};

	const int sz = width * height;
	const int SUPSZ = sz / K;
	//nlabels.resize(sz, -1);
	for (int i = 0; i < sz; i++)
		nlabels[i] = -1;
	int label(0);
	int *xvec = new int[sz];
	int *yvec = new int[sz];
	int oindex(0);
	int adjlabel(0); //adjacent label
	for (int j = 0; j < height; j++)
	{
		for (int k = 0; k < width; k++)
		{
			if (0 > nlabels[oindex])
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{
					for (int n = 0; n < 4; n++)
					{
						int x = xvec[0] + dx4[n];
						int y = yvec[0] + dy4[n];
						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y * width + x;
							if (nlabels[nindex] >= 0)
								adjlabel = nlabels[nindex];
						}
					}
				}

				int count(1);
				for (int c = 0; c < count; c++)
				{
					for (int n = 0; n < 4; n++)
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y * width + x;

							if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}
					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if (count <= SUPSZ >> 2)
				{
					for (int c = 0; c < count; c++)
					{
						int ind = yvec[c] * width + xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	numlabels = label;

	if (xvec)
		delete[] xvec;
	if (yvec)
		delete[] yvec;
}

//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void SLIC::PerformSLICO_ForGivenK(
	const unsigned int *ubuff,
	const int width,
	const int height,
	int *klabels,
	int &numlabels,
	const int &K,	 //required number of superpixels
	const double &m) //weight given to spatial distance
{
	//--------------------------------------------------
	m_width = width;
	m_height = height;
	int sz = m_width * m_height;
	//--------------------------------------------------

	//-------------------------------------
	memset(klabels, -1, sizeof(int) * sz);
	double step = sqrt(double(sz) / double(K));
	int num = (m_width / step + 1) * (m_height / step + 1); 
	//-------------------------------------
    // double *kseedsl, *kseedsa, *kseedsb, *kseedsx, *kseedsy;
    // double* kseedsl = new double[num];
    // double* kseedsa = new double[num];
    // double* kseedsb = new double[num];
    // double* kseedsx = new double[num];
    // double* kseedsy = new double[num];
	double* seeds = new double[5*num];



	if (1) //LAB
	{
		DoRGBtoLABConversion(ubuff, m_labvec);
	}
	else //RGB
	{
		m_labvec = new double[3*sz];
        // m_lvec = (double*)_mm_malloc(sz * sizeof(double), 256);
        // m_avec = (double*)_mm_malloc(sz * sizeof(double), 256);
        // m_bvec = (double*)_mm_malloc(sz * sizeof(double), 256);
        for (int i = 0; i < sz; i++) {
            m_labvec[3*i+2] = ubuff[i] >> 16 & 0xff;
            m_labvec[3*i+1] = ubuff[i] >> 8 & 0xff;
            m_labvec[3*i] = ubuff[i] & 0xff;
        }
	}
	//--------------------------------------------------

	bool perturbseeds(true);
	// vector<double> edgemag(0);
	// if(perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(seeds, numlabels, K, perturbseeds);

	int STEP = sqrt(double(sz) / double(K)) + 2.0; //adding a small value in the even the STEP size is too small.
	calculate_super_pixel(sz,K,numlabels,seeds,klabels);
//	PerformSuperpixelSegmentation_VariableSandM(seeds, klabels, numlabels, STEP, 10);

	// delete[] kseedsl;
    // delete[] kseedsa;
    // delete[] kseedsb;
    // delete[] kseedsx;
    // delete[] kseedsy;
	delete[] seeds;

	int *nlabels =new int[sz];
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);
	{
		memcpy(klabels, nlabels, sizeof(int) * sz);
	}
	if (nlabels)
		delete[] (nlabels);
}

//===========================================================================
/// Load PPM file
///
///
//===========================================================================
void LoadPPM(char *filename, unsigned int **data, int *width, int *height)
{
	char header[1024];
	FILE *fp = NULL;
	int line = 0;

	fp = fopen(filename, "rb");

	// read the image type, such as: P6
	// skip the comment lines
	while (line < 2)
	{
		fgets(header, 1024, fp);
		if (header[0] != '#')
		{
			++line;
		}
	}
	// read width and height
	sscanf(header, "%d %d\n", width, height);

	// read the maximum of pixels
	fgets(header, 20, fp);

	// get rgb data
	unsigned char *rgb = new unsigned char[(*width) * (*height) * 3];
	fread(rgb, (*width) * (*height) * 3, 1, fp);

	*data = new unsigned int[(*width) * (*height) * 4];
	int k = 0;
	for (int i = 0; i < (*height); i++)
	{
		for (int j = 0; j < (*width); j++)
		{
			unsigned char *p = rgb + i * (*width) * 3 + j * 3;
			// a ( skipped )
			(*data)[k] = p[2] << 16; // r
			(*data)[k] |= p[1] << 8; // g
			(*data)[k] |= p[0];		 // b
			k++;
		}
	}

	// ofc, later, you'll have to cleanup
	delete[] rgb;

	fclose(fp);
}

//===========================================================================
/// Load PPM file
///
///
//===========================================================================
int CheckLabelswithPPM(char *filename, int *labels, int width, int height)
{
	char header[1024];
	FILE *fp = NULL;
	int line = 0, ground = 0;

	fp = fopen(filename, "rb");

	// read the image type, such as: P6
	// skip the comment lines
	while (line < 2)
	{
		fgets(header, 1024, fp);
		if (header[0] != '#')
		{
			++line;
		}
	}
	// read width and height
	int w(0);
	int h(0);
	sscanf(header, "%d %d\n", &w, &h);
	if (w != width || h != height)
		return -1;

	// read the maximum of pixels
	fgets(header, 20, fp);

	// get rgb data
	unsigned char *rgb = new unsigned char[(w) * (h)*3];
	fread(rgb, (w) * (h)*3, 1, fp);

	int num = 0, k = 0;
	for (int i = 0; i < (h); i++)
	{
		for (int j = 0; j < (w); j++)
		{
			unsigned char *p = rgb + i * (w)*3 + j * 3;
			// a ( skipped )
			ground = p[2] << 16; // r
			ground |= p[1] << 8; // g
			ground |= p[0];		 // b

			if (ground != labels[k])
				num++;

			k++;
		}
	}

	// ofc, later, you'll have to cleanup
	delete[] rgb;

	fclose(fp);

	return num;
}

//===========================================================================
///	The main function
///
//===========================================================================
int main(int argc, char **argv)
{
	unsigned int *img = NULL;
	int width(0);
	int height(0);

	LoadPPM((char *)"input_image.ppm", &img, &width, &height);
	if (width == 0 || height == 0)
		return -1;

	int sz = width * height;
	int* labels = new int[sz];
	int numlabels(0);
	SLIC slic;
	int m_spcount;
	double m_compactness;
	m_spcount = argc < 2 ? 400 : stoi(argv[1]);
	m_compactness = 10.0;
	auto startTime = Clock::now();
	
	//start
	slic.PerformSLICO_ForGivenK(img, width, height, labels, numlabels, m_spcount, m_compactness); //for a given number K of superpixels
	
	
	
	auto endTime = Clock::now();
	auto compTime = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
	cout << "Computing time=" << compTime.count() / 1000 << " ms" << endl;

	int num = CheckLabelswithPPM((char *)"check.ppm", labels, width, height);
	if (num < 0)
	{
		cout << "The result for labels is different from output_labels.ppm." << endl;
	}
	else
	{
		cout << "There are " << num << " points' labels are different from original file." << endl;
	}

	slic.SaveSuperpixelLabels2PPM((char *)"output_labels.ppm", labels, width, height);
	if (labels)
		delete[] labels;

	if (img)
		delete[] img;

	return 0;
}