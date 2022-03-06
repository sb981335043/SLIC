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
#include <deque>
#include <unordered_map>
#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <smmintrin.h>


class area_info
{
public:
	int index;
	int x, y;
	int count;
	int new_label;
	int seg_label;

	bool operator<(const area_info &other)
	{
		return index < other.index;
	}
};


using namespace std;

#define OMP_NUM_THREADS 16
inline double square(double x)
{
	return x*x;
}


class SLIC
{
public:
	SLIC();
	virtual ~SLIC();

	void super_pixel(int sz,int K,int N,double* seeds,int *klabels);

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
	// void PerformSuperpixelSegmentation_VariableSandM(
	// 	double* seeds,
	// 	int *klabels,
	// 	const int numk,
	// 	const int &STEP,
	// 	const int &NUMITR);

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
		int *labels,
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

int numThreads = 16;

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
	// for(int i= 0 ; i<256;++i) cout<<tableRGB[i];
	#pragma omp parallel for
	for (int j = 0; j < sz; j++)
	{
		int r = (ubuff[j] >> 16) & 0xFF;
		int g = (ubuff[j] >> 8) & 0xFF;
		int b = (ubuff[j]) & 0xFF;
		// cout<<r<<" "<<g<<" "<<b<<endl;
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

		labval[2] = 116.0 * fy - 16.0;
		labval[1] = 500.0 * (fx - fy);
		labval[0] = 200.0 * (fy - fz);
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
void SLIC::super_pixel(int sz,int K,int N,double* seeds,int *klabels) 
{
   const int TIME = 10;//迭代次数
   double * lab = m_labvec;
   const int STEP = (int) (sqrt((double) (sz) / (double) (K)) + 2.0);
   int offset = STEP;
   if (STEP < 10) offset = (int) (STEP * 1.5);
   double inv_xy = 1.0 / (STEP * STEP);
   int width = m_width;
   int height = m_height;

	//试了很多遍 32 最好
   double max_lab[N] __attribute__((aligned(32)));
   double max_lab_div[N] __attribute__((aligned(32)));

   double sigma[OMP_NUM_THREADS][N][5] __attribute__((aligned(32)));
   double max_lab_t[OMP_NUM_THREADS][N] __attribute__((aligned(32)));
   int cluster_size[OMP_NUM_THREADS][N] __attribute__((aligned(32)));


	#pragma omp simd
   for (int i = 0; i < N; ++i) {
       max_lab[i] = 100;
       max_lab_div[i] = 0.01;
  }

   //迭代TIME次
   for (int t = 0; t < TIME; ++t) {
       //初始化清零数据
       memset(cluster_size, 0, N * OMP_NUM_THREADS * sizeof(int));
       memset(max_lab_t, 0, N * OMP_NUM_THREADS * sizeof(double));
       memset(sigma, 0, 5 * N * OMP_NUM_THREADS * sizeof(double));
       //遍历所有seed,计算最近的靠近位置
	#pragma omp parallel for default(none) shared(height, width, N, offset, t, seeds, lab, klabels, inv_xy, max_lab_t, max_lab_div, sigma, cluster_size) schedule(static)
       for (int y = 0; y < height; ++y) {
           double dis_vec_y[width] __attribute__((aligned(32)));
           double dis_lab_y[width] __attribute__((aligned(32)));
           memset(dis_vec_y, 0x43 , width * sizeof(double));
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
                           klabels[i] = n;
                      }
                  }
              }
          }
           const int thread_num = omp_get_thread_num();
           for (int x = 0; x < width; ++x) {
               int i = width * y + x;
               int k = klabels[i];
               if (max_lab_t[thread_num][k] < dis_lab_y[x]) {
                   max_lab_t[thread_num][k] = dis_lab_y[x];
              }
			#pragma omp simd
               for (int j = 0; j < 3; ++j) {
                   sigma[thread_num][k][j] += lab[i * 3 + j];
              }
               sigma[thread_num][k][3] += x;
               sigma[thread_num][k][4] += y;
               ++cluster_size[thread_num][k];
          }
      }

       // 重新计算种子点
       for (int k = 0; k < N; k++) {
           int seed_size = 0;
		   double* seed = seeds + k*5;
           double sigma_t[5] __attribute__((aligned(32))) = {0};
           for (int i = 0; i < OMP_NUM_THREADS; ++i) {
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
               seed[i] = sigma_t[i] * inv;
          }
      }
  }
}


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

/* 并查集 */ //处理不好
// void SLIC::EnforceLabelConnectivity(
// 	const int*					belong,//input labels that need to be corrected to remove stray labels
// 	const int&					width,
// 	const int&					height,
// 	int*						tempLabel,//new labels
// 	int&						N,//the number of labels changes in the end if segments are removed
// 	const int&					K)
// {
// 		const int dx4[4] = {-1,  0,  1,  0};
// 		const int dy4[4] = { 0, -1,  0,  1};

// 		const int sz = width*height;
// 		const int SUPSZ = sz/K;
// 		int label(0);
// 		int oindex(0);
// 		int adjlabel(0);//adjacent label

// 		int* xvec = new int[sz];
// 		int* yvec = new int[sz];
// 		label = 0;
// 		memset(tempLabel, -1, sz*sizeof(int));
// 	// 直接并行BFS打标签并合并
// #pragma omp parallel for num_threads(OMP_NUM_THREADS) default(none) shared(thread_num, K0, N, Q, tempLabel, oindex, width, dx4, dy4, height, belong, P)
//     for (int id = 0; id < N; id++) {
//         int nowLabel = id;
 
//         int size = 0;
//         for (int tid = 0; tid < OMP_NUM_THREADS; tid++)
//             size += Q[tid][id].length;
//         int isOK = 0;
//         int *arr = (int *) malloc(size * sizeof(int));
//         while (isOK < size) {
//             int now = -1;
//             for (int tid = 0; tid < OMP_NUM_THREADS; tid++) {
//                 for (int i = 0; i < Q[tid][id].length; i++) {
//                     if (tempLabel[get(Q[tid][id], i)] == -1) {
//                         now = get(Q[tid][id], i);
//                         break;
//                     }
//                 }
//             }
//             int start = 0, finish = 0;
//             arr[finish++] = now;
//             tempLabel[now] = nowLabel;
//             oindex = now;
//             while (start < finish) {
//                 int k = arr[start++];
//                 int x = k % width;
//                 int y = k / width;
//                 for (int i = 0; i < 4; i++) {
//                     int xx = x + dx4[i];
//                     int yy = y + dy4[i];
//                     if ((xx >= 0 && xx < width) && (yy >= 0 && yy < height)) {
//                         int nindex = yy * width + xx;
 
//                         if (0 > tempLabel[nindex] && belong[oindex] == belong[nindex]) {
//                             arr[finish++] = nindex;
//                             tempLabel[nindex] = nowLabel;
//                         }
//                     }
//                 }
 
//             }
//             cut(&P[nowLabel], finish);
//             for (int i = 0; i < finish; i++)
//                 set(P[nowLabel], i, arr[i]);
//             isOK += finish;
//             nowLabel += K0;
//         }
//         free(arr);
//     }
// }

// ===========================================================================
// /	EnforceLabelConnectivity
// /
// /		1. finding an adjacent label for each new component at the start
// /		2. if a certain component is too small, assigning the previously found
// /		    adjacent label to this component, and not incrementing the label.
// ===========================================================================
void SLIC::EnforceLabelConnectivity(
	int*					labels,//input labels that need to be corrected to remove stray labels
	const int&					width,
	const int&					height,
	int*						nlabels,//new labels
	int&						numlabels,//the number of labels changes in the end if segments are removed
	const int&					K) //the number of superpixels desired by the user
{	

	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int sz = width*height;
	const int SUPSZ = sz/K;
	int label(0);
	int oindex(0);
	int adjlabel(0);//adjacent label

	// 1、确认连通性，多线程，合并
	// 2、踢掉小块的
	// 3、重新遍历，按照顺序重写 label

	// 下面的逻辑就是，每遇到一个没打标签的点，就把它广度优先搜索
	// 如果这片区域太小，就和别的合并
	// 并行难点在于，给每个区块打标签是按顺序的
	// printf("\nSecond\n\n");
	int* xvec = new int[sz];
	int* yvec = new int[sz];
	label = 0;
	memset(nlabels, -1, sz*sizeof(int));
	for( int j = 0; j < height; j++ )
	{
		for( int k = 0; k < width; k++ )
		{
			if( 0 > nlabels[oindex] )
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
				for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = y*width + x;
						if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}
				// 广度优先搜索，深度优先并不能更好利用缓存

				int count(1);
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];
						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = y*width + x;

							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
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
				if(count <= SUPSZ >> 2)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]*width+xvec[c];
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
	if(xvec) delete [] xvec;
	if(yvec) delete [] yvec;

}
// void SLIC::EnforceLabelConnectivity(
// 	int *labels, //input labels that need to be corrected to remove stray labels
// 	const int &width,
// 	const int &height,
// 	int *nlabels,	//new labels
// 	int &numlabels, //the number of labels changes in the end if segments are removed
// 	const int &K)	//the number of superpixels desired by the user
// {
// 	//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
// 	//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// 	const int dx4[4] = {-1, 0, 1, 0};
// 	const int dy4[4] = {0, -1, 0, 1};

// 	const int sz = width * height;
// 	const int SUPSZ = sz / K;
// //nlabels.resize(sz, -1);
// // #pragma omp parallel for simd //schedule(static, 2048)
// // 	for (int i = 0; i < sz; i++)
// // 		nlabels[i] = -1;
// memset(nlabels, -1, sz*sizeof(int));
// 	// int oindex(0);
// 	// int adjlabel(0); //adjacent label

// 	vector<area_info> seg_info;

// 	// BFS to tag new label, and gather info for mapping
// 	vector<omp_lock_t> lock_vec(numlabels);
// 	for (size_t i = 0; i < numlabels; i++)
// 	{
// 		omp_init_lock(&lock_vec[i]);
// 	}

// 	int local_label = 0;
// 	int label = 0;
// 	unordered_map<int, area_info *> seg_label_map;
// 	deque<pair<int, area_info *>> shrinked_area;

// #pragma omp parallel private(local_label)
// 	{
// 		int *xvec = new int[sz];
// 		int *yvec = new int[sz];
// 		const int thread_id = omp_get_thread_num();
// 		const int thread_num = omp_get_num_threads();

// #pragma omp for private(local_label)
// 		for (int j = 0; j < height; j++)
// 		{
// 			for (int k = 0; k < width; k++)
// 			{
// 				int seg_label = local_label * thread_num + thread_id;

// 				int oindex = j * width + k;
// 				if (nlabels[oindex] >= 0)
// 				{
// 					continue;
// 				}
// 				omp_set_lock(&lock_vec[labels[oindex]]);
// 				if (nlabels[oindex] >= 0)
// 				{
// 					omp_unset_lock(&lock_vec[labels[oindex]]);
// 					continue;
// 				}

// 				area_info info;
// 				info.index = oindex;
// 				info.x = k;
// 				info.y = j;
// 				info.seg_label = seg_label;
// 				info.new_label = 0;

// 				nlabels[oindex] = seg_label;
// 				//--------------------
// 				// Start a new segment
// 				//--------------------
// 				xvec[0] = k;
// 				yvec[0] = j;

// 				// BFS
// 				int count(1);
// 				for (int c = 0; c < count; c++)
// 				{
// 					for (int n = 0; n < 4; n++)
// 					{
// 						int x = xvec[c] + dx4[n];
// 						int y = yvec[c] + dy4[n];

// 						if ((x >= 0 && x < width) && (y >= 0 && y < height))
// 						{
// 							int nindex = y * width + x;

// 							if (nlabels[nindex] < 0 && labels[oindex] == labels[nindex])
// 							{
// 								xvec[count] = x;
// 								yvec[count] = y;
// 								if (info.index > nindex)
// 								{
// 									info.index = nindex;
// 									info.x = x;
// 									info.y = y;
// 								}

// 								nlabels[nindex] = seg_label;
// 								count++;
// 							}
// 						}
// 					}
// 				}
// 				info.count = count;
// #pragma omp critical
// 				{
// 					seg_info.push_back(info);
// 				}
// 				omp_unset_lock(&lock_vec[labels[oindex]]);
// 				local_label++;
// 			}
// 		}

// #pragma omp master
// 		{
// 			std::sort(seg_info.begin(), seg_info.end());

// 			for (auto &info : seg_info)
// 			{
// 				if (info.count <= SUPSZ >> 2)
// 				{
// 					// info.new_label = info.adjacent_index;
// 					shrinked_area.push_back(make_pair(info.seg_label, &info));
// 					continue;
// 				}
// 				info.new_label = label;
// 				seg_label_map[info.seg_label] = &info;
// 				label++;
// 			}

// 			while (!shrinked_area.empty())
// 			{

// 				auto pair = shrinked_area.front();
// 				if (pair.second->index == 0)
// 				{
// 					seg_label_map[pair.first] = pair.second;
// 					pair.second->new_label = 0;
// 					shrinked_area.pop_front();
// 					continue;
// 				}

// 				//-------------------------------------------------------
// 				// Quickly find an adjacent label for use later if needed
// 				//-------------------------------------------------------
// 				int adjacent_label = -1;
// 				for (int n = 0; n < 4; n++)
// 				{
// 					int x = pair.second->x + dx4[n];
// 					int y = pair.second->y + dy4[n];
// 					if ((x >= 0 && x < width) && (y >= 0 && y < height))
// 					{
// 						int nindex = y * width + x;
// 						if (nlabels[nindex] == pair.first)
// 						{
// 							continue;
// 						}

// 						if (seg_label_map.count(nlabels[nindex]) == 0)
// 						{
// 							continue;
// 							adjacent_label = -1;
// 							break;
// 						}
// 						else if (seg_label_map[nlabels[nindex]]->index < pair.second->index)
// 						{
// 							adjacent_label = nlabels[nindex];
// 						}
// 					}
// 				}
// 				if (adjacent_label == -1)
// 				{
// 					shrinked_area.push_back(pair);
// 				}
// 				else
// 				{
// 					seg_label_map[pair.first] = seg_label_map[adjacent_label];
// 				}
// 				shrinked_area.pop_front();
// 			}
// 		}

// // Map old label to new label
// #pragma omp barrier
// #pragma omp for simd
// 		for (size_t i = 0; i < sz; i++)
// 		{
// 			labels[i] = seg_label_map[nlabels[i]]->new_label;
// 		}
// 	}

// 	numlabels = label;
// }
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
//	int num = (m_width / step + 1) * (m_height / step + 1); 
	//-------------------------------------
    // double *kseedsl, *kseedsa, *kseedsb, *kseedsx, *kseedsy;
    // double* kseedsl = new double[num];
    // double* kseedsa = new double[num];
    // double* kseedsb = new double[num];
    // double* kseedsx = new double[num];
    // double* kseedsy = new double[num];
	double* seeds = new double[5*K];



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
	// for(int i=0;i<sz;i++)
	//  cout<<m_labvec[i]<<" ";
	//--------------------------------------------------

	bool perturbseeds(true);
	// vector<double> edgemag(0);
	// if(perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(seeds, numlabels, K, perturbseeds);
	// cout<<numlabels<<endl;
	// for(int i=0;i<5*K;i++) cout<<seeds[i]<<" ";

	//int STEP = sqrt(double(sz) / double(K)) + 2.0; //adding a small value in the even the STEP size is too small.
	super_pixel(sz,K,numlabels,seeds,klabels);
	// for(int i=0;i<sz;i++) cout<<klabels[i]<<" ";
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