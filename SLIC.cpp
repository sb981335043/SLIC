#include <stdio.h>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <fstream>
#include "SLIC.h"
#include <chrono>

#include <emmintrin.h>
#include <immintrin.h>
#include <omp.h>
#include <smmintrin.h>

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
	m_lvec = NULL;
	m_avec = NULL;
	m_bvec = NULL;
}

SLIC::~SLIC()
{
    if (m_lvec) delete[] m_lvec;
    if (m_avec) delete[] m_avec;
    if (m_bvec) delete[] m_bvec;
}


void SLIC::DoRGBtoLABConversion(
	const unsigned int *&ubuff,
	double *&lvec,
	double *&avec,
	double *&bvec)
{
	int sz = m_width * m_height;
    lvec = (double*)_mm_malloc(sz * sizeof(double), 256);
    avec = (double*)_mm_malloc(sz * sizeof(double), 256);
    bvec = (double*)_mm_malloc(sz * sizeof(double), 256);
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
		if (X > 0.008417238336) fx = cbrt(X * (1.0 / 0.950456));
		else fx = (8.192982069151272 * X + 0.1379310344827586);
		if (Y > 0.008856) fy = cbrt(Y);
		else fy = (7.787068965517241 * Y + 0.1379310344827586);
		if (Z > 0.009642005424) fz = cbrt(Z * (1.0 / 1.088754));
		else fz = (7.152275872710678 * Z + 0.1379310344827586);
		//这里通过

		lvec[j] = 116.0 * fy - 16.0;
		avec[j] = 500.0 * (fx - fy);
		bvec[j] = 200.0 * (fy - fz);
	}
	delete[] tableRGB;
}

// 计算单个点的梯度，对于一个初始seed计算周围八个点即可
double SLIC::DetectLABPixelEdge(
	const int &i)
{
	const double *lvec = m_lvec;
	const double *avec = m_avec;
	const double *bvec = m_bvec;
	const int width = m_width;

	double dx = (lvec[i - 1] - lvec[i + 1]) * (lvec[i - 1] - lvec[i + 1]) +
				(avec[i - 1] - avec[i + 1]) * (avec[i - 1] - avec[i + 1]) +
				(bvec[i - 1] - bvec[i + 1]) * (bvec[i - 1] - bvec[i + 1]);

	double dy = (lvec[i - width] - lvec[i + width]) * (lvec[i - width] - lvec[i + width]) +
				(avec[i - width] - avec[i + width]) * (avec[i - width] - avec[i + width]) +
				(bvec[i - width] - bvec[i + width]) * (bvec[i - width] - bvec[i + width]);
	return dx + dy;
}


//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYSeeds_ForGivenK(
	double* kseedsl,
	double* kseedsa,
	double* kseedsb,
	double* kseedsx,
	double* kseedsy,
	int& numk,
	const int &K,
	const bool &perturbseeds)
{

	int sz = m_width * m_height;
	double step = sqrt(double(sz) / double(K));
	int T = step;
	int xoff = step / 2;
	int yoff = step / 2;

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
			kseedsl[numk]=m_lvec[i];
			kseedsa[numk]=m_avec[i];
			kseedsb[numk]=m_bvec[i];
			kseedsx[numk]=X;
			kseedsy[numk]=Y;
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
			int ox = kseedsx[n]; //original x
			int oy = kseedsy[n]; //original y
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
				kseedsx[n] = storeind % m_width;
				kseedsy[n] = storeind / m_width;
				kseedsl[n] = m_lvec[storeind];
				kseedsa[n] = m_avec[storeind];
				kseedsb[n] = m_bvec[storeind];
			}
		}
	}
}

// void SLIC::PerformSuperpixelSegmentation_VariableSandM(
// 	double* kseedsl,
// 	double* kseedsa,
// 	double* kseedsb,
// 	double* kseedsx,
// 	double* kseedsy,
// 	int *klabels,
// 	const int numk, 
// 	const int &STEP,
// 	const int &NUMITR)
// {
// 	int sz = m_width * m_height;
// 	//const int numk = kseedsl.size();
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
// //	vector<double> distxy(sz, DBL_MAX);
// //	vector<double> distlab(sz, DBL_MAX);
// 	double* distlab = (double*)_mm_malloc(sz * sizeof(double), 256);
// 	#pragma omp parallel for
//     for (int i = 0; i < sz; ++i) {
//         // distxy[i] = DBL_MAX;
//         distlab[i] = DBL_MAX;
//     }

// 	double* distvec = (double*)_mm_malloc(sz * sizeof(double), 256);
	
// 	vector<double> maxlab(numk, 10 * 10);	 //THIS IS THE VARIABLE VALUE OF M, just start with 10
// 	// vector<double> maxxy(numk, STEP * STEP); //THIS IS THE VARIABLE VALUE OF M, just start with 10

// 	double invxywt = 1.0 / (STEP * STEP); //NOTE: this is different from how usual SLIC/LKM works
// 	//迭代10次

// 	//-----------------------------------------------
// 	 __m256d invxywt_vec = _mm256_set1_pd(invxywt);

//     // vector<int> distidx(sz, -1);
//     int* distidx = new int[sz];
//     double* _maxlab[OMP_NUM_THREADS];
//     // double* _maxxy[OMP_NUM_THREADS];
//     double* _sigmal[OMP_NUM_THREADS];
//     double* _sigmaa[OMP_NUM_THREADS];
//     double* _sigmab[OMP_NUM_THREADS];
//     double* _sigmax[OMP_NUM_THREADS];
//     double* _sigmay[OMP_NUM_THREADS];
//     int* _clustersize[OMP_NUM_THREADS];

// // memset(distidx, 0, sizeof(int) * sz);
// 	#pragma omp parallel for
//     for (int i = 0; i < OMP_NUM_THREADS; ++i) {
//         _maxlab[i] = new double[numk];
//         // _maxxy[i] = new double[numk];
//         _sigmal[i] = new double[numk];
//         _sigmaa[i] = new double[numk];
//         _sigmab[i] = new double[numk];
//         _sigmax[i] = new double[numk];
//         _sigmay[i] = new double[numk];
//         _clustersize[i] = new int[numk];
//     }
// 	//-----------------------------------------------
// 	while (numitr < NUMITR)
// 	{
// 		//------
// 		//cumerr = 0;
// 		numitr++;
// 		//------
		
// 		for (int i = 0; i < sz; ++i) {
//             distvec[i] = DBL_MAX;
//         }
// 		// not double max but large enough
//         // memset(distvec,0x7F,sz*sizeof(double));
// 		//遍历，算出每个点与seeds的最近距离进行分类
//         for (int n = 0; n < numk; n++) {
//             const double _kseedsl = kseedsl[n];
//             __m256d _kseedsl_vec = _mm256_set1_pd(kseedsl[n]);
//             const double _kseedsa = kseedsa[n];
//             __m256d _kseedsa_vec = _mm256_set1_pd(kseedsa[n]);
//             const double _kseedsb = kseedsb[n];
//             __m256d _kseedsb_vec = _mm256_set1_pd(kseedsb[n]);
//             const double _kseedsx = kseedsx[n];
//             __m256d _kseedsx_vec = _mm256_set1_pd(kseedsx[n]);
//             const double _kseedsy = kseedsy[n];
//             __m256d _kseedsy_vec = _mm256_set1_pd(kseedsy[n]);
//             __m256d maxlab_vec = _mm256_set1_pd(maxlab[n]);


//             const int y1 = max(0, (int)(_kseedsy - offset));
//             const int y2 = min(m_height, (int)(_kseedsy + offset));
//             const int x1 = max(0, (int)(_kseedsx - offset));
//             const int x2 = min(m_width, (int)(_kseedsx + offset));

// 			#pragma omp parallel for
//             for (int y = y1; y < y2; y++) {
//                 double* res_unpack =
//                     (double*)_mm_malloc(4 * sizeof(double), 256);
//                 for (int x = x1; x < x2;) {
//                     const int i = y * m_width + x;

//                     if ((i & 0x3) != 0 || x + 4 > x2) {
//                         // not aligned part
//                         const double l = m_lvec[i];
//                         const double a = m_avec[i];
//                         const double b = m_bvec[i];

//                         const double _distlab =
//                             (l - _kseedsl) * (l - _kseedsl) +
//                             (a - _kseedsa) * (a - _kseedsa) +
//                             (b - _kseedsb) * (b - _kseedsb);

//                         const double _distxy = (x - _kseedsx) * (x - _kseedsx) +
//                                                (y - _kseedsy) * (y - _kseedsy);

//                         //------------------------------------------------------------------------
//                         const double dist =
//                             _distlab / maxlab[n] +
//                             _distxy * invxywt;              

//                         distlab[i] = _distlab;

//                         if (dist < distvec[i]) {
//                             distvec[i] = dist;
//                             klabels[i] = n;
//                         }
//                         ++x;
//                     } else {
//                         // aligned part
//                         __m256d l_vec = _mm256_load_pd(&m_lvec[i]);
//                         __m256d a_vec = _mm256_load_pd(&m_avec[i]);
//                         __m256d b_vec = _mm256_load_pd(&m_bvec[i]);
//                         __m256d x_vec =
//                             _mm256_set_pd((double)(x + 3), (double)(x + 2),
//                                           (double)(x + 1), (double)(x));
//                         __m256d y_vec = _mm256_set1_pd((double)y);
//                         __m256d l_vec_t1 = _mm256_sub_pd(l_vec, _kseedsl_vec);
//                         l_vec_t1 = _mm256_mul_pd(l_vec_t1, l_vec_t1);
//                         __m256d a_vec_t1 = _mm256_sub_pd(a_vec, _kseedsa_vec);
//                         a_vec_t1 = _mm256_mul_pd(a_vec_t1, a_vec_t1);
//                         __m256d b_vec_t1 = _mm256_sub_pd(b_vec, _kseedsb_vec);
//                         b_vec_t1 = _mm256_mul_pd(b_vec_t1, b_vec_t1);
//                         __m256d _distlab_vec =
//                             _mm256_add_pd(l_vec_t1, a_vec_t1);
//                         _distlab_vec = _mm256_add_pd(_distlab_vec, b_vec_t1);
//                         __m256d x_vec_t1 = _mm256_sub_pd(x_vec, _kseedsx_vec);
//                         x_vec_t1 = _mm256_mul_pd(x_vec_t1, x_vec_t1);
//                         __m256d y_vec_t1 = _mm256_sub_pd(y_vec, _kseedsy_vec);
//                         y_vec_t1 = _mm256_mul_pd(y_vec_t1, y_vec_t1);
//                         __m256d _distxy_vec = _mm256_add_pd(x_vec_t1, y_vec_t1);
//                         __m256d dist_vec_t1 =
//                             _mm256_div_pd(_distlab_vec, maxlab_vec);
//                         __m256d dist_vec_t2 =
//                             _mm256_mul_pd(_distxy_vec, invxywt_vec);
//                         __m256d dist_vec =
//                             _mm256_add_pd(dist_vec_t1, dist_vec_t2);

//                         _mm256_store_pd(&distlab[i], _distlab_vec);

//                         __m256d distvec_vec = _mm256_load_pd(&distvec[i]);
//                         __m256d cmp_res_vec =
//                             _mm256_cmp_pd(dist_vec, distvec_vec, _CMP_LT_OQ);
//                         distvec_vec = _mm256_blendv_pd(distvec_vec, dist_vec,
//                                                        cmp_res_vec);
//                         _mm256_store_pd(&distvec[i], distvec_vec);

//                         __m256i permuted_vec = _mm256_permutevar8x32_epi32(
//                             _mm256_castpd_si256(cmp_res_vec), K_PERM_VEC);
//                         __m128i cmp_int_vec =
//                             _mm256_castsi256_si128(permuted_vec);

//                         __m128i n_vec = _mm_set1_epi32(n);
//                         _mm_maskstore_epi32(&klabels[i], cmp_int_vec, n_vec);;
//                         x += 4;
//                     }
//                 }
//                 _mm_free(res_unpack);
//             }
//         }
// 		#pragma omp parallel for
//         for (int i = 0; i < OMP_NUM_THREADS; ++i) {
//             memset(_maxlab[i], 0, sizeof(double) * numk);
//         }
// 		//-----------------------------------------------------------------
// 		// Assign the max color distance for a cluster
// 		//-----------------------------------------------------------------
// 		#pragma omp parallel for
//         for (int i = 0; i < sz; i++) {
//             int idx = omp_get_thread_num();
//             if (_maxlab[idx][klabels[i]] < distlab[i])
//                 _maxlab[idx][klabels[i]] = distlab[i];
//         }
// 		#pragma omp parallel for
//         for (int i = 0; i < numk; ++i)
//             for (int j = 0; j < OMP_NUM_THREADS; ++j) {
//                 if (maxlab[i] < _maxlab[j][i]) maxlab[i] = _maxlab[j][i];
//             }

// 		//-----------------------------------------------------------------
// 		// Recalculate the centroid and store in the seed values
// 		//-----------------------------------------------------------------
// 		sigmal.assign(numk, 0);
// 		sigmaa.assign(numk, 0);
// 		sigmab.assign(numk, 0);
// 		sigmax.assign(numk, 0);
// 		sigmay.assign(numk, 0);
// 		clustersize.assign(numk, 0);

// 		#pragma omp parallel for
//         for (int i = 0; i < OMP_NUM_THREADS; ++i) {
//             memset(_sigmal[i], 0, sizeof(double) * numk);
//             memset(_sigmaa[i], 0, sizeof(double) * numk);
//             memset(_sigmab[i], 0, sizeof(double) * numk);
//             memset(_sigmax[i], 0, sizeof(double) * numk);
//             memset(_sigmay[i], 0, sizeof(double) * numk);
//             memset(_clustersize[i], 0, sizeof(int) * numk);
//         }
		
// 		//计算分类后一个label里所有的点的向量
// 		#pragma omp parallel
// 		{
// 			int idx = omp_get_thread_num();
// 			#pragma omp for
// 			for (int j = 0; j < sz; j++) {

// 				_sigmal[idx][klabels[j]] += m_lvec[j];
// 				_sigmaa[idx][klabels[j]] += m_avec[j];
// 				_sigmab[idx][klabels[j]] += m_bvec[j];
// 				_sigmax[idx][klabels[j]] += (j % m_width);
// 				_sigmay[idx][klabels[j]] += (j / m_width);

// 				_clustersize[idx][klabels[j]]++;
// 			}
// 		}
// 		#pragma omp parallel for
// 		for (int i = 0; i < numk; ++i)
// 		for (int j = 0; j < OMP_NUM_THREADS; ++j) {
// 			sigmal[i] += _sigmal[j][i];
// 			sigmaa[i] += _sigmaa[j][i];
// 			sigmab[i] += _sigmab[j][i];
// 			sigmax[i] += _sigmax[j][i];
// 			sigmay[i] += _sigmay[j][i];
// 			clustersize[i] += _clustersize[j][i];
// 		}

// 		#pragma omp parallel
// 		{
// 			#pragma omp for
// 			for (int k = 0; k < numk; k++) {
// 				//_ASSERT(clustersize[k] > 0);
// 				if (clustersize[k] <= 0) clustersize[k] = 1;
// 				inv[k] = 1.0 /
// 						double(clustersize[k]);  // computing inverse now to
// 												// multiply, than divide later
// 			}
// 		}

// 		#pragma omp parallel
// 		{
// 			#pragma omp for
// 			for (int k = 0; k < numk; k++) {
// 				kseedsl[k] = sigmal[k] * inv[k];
// 				kseedsa[k] = sigmaa[k] * inv[k];
// 				kseedsb[k] = sigmab[k] * inv[k];
// 				kseedsx[k] = sigmax[k] * inv[k];
// 				kseedsy[k] = sigmay[k] * inv[k];
// 			}
// 		}
// 	}
// 	delete[] distidx;
// 	#pragma omp parallel for
// 	for (int i = 0; i < OMP_NUM_THREADS; ++i) {
// 		delete[] _maxlab[i];
// 		// delete[] _maxxy[i];
// 		delete[] _sigmal[i];
// 		delete[] _sigmaa[i];
// 		delete[] _sigmab[i];
// 		delete[] _sigmax[i];
// 		delete[] _sigmay[i];
// 		delete[] _clustersize[i];
// 	}
// 	_mm_free(distlab);
// 	// _mm_free(distxy);
// 	_mm_free(distvec);
// 	//_mm_free(res_unpack);
// }
void SLIC::PerformSuperpixelSegmentation_VariableSandM(
	double* kseedsl,
	double* kseedsa,
	double* kseedsb,
	double* kseedsx,
	double* kseedsy,
	int *klabels,
	const int numk,
	const int &STEP,
	const int &NUMITR)
{
	int sz = m_width * m_height;
	// const int numk = kseedsl.size();
	//double cumerr(99999.9);
	int numitr(0);

	//----------------
	int offset = STEP;
	if (STEP < 10)
		offset = STEP * 1.5;
	//----------------

	vector<double> sigmal(numk, 0);
	vector<double> sigmaa(numk, 0);
	vector<double> sigmab(numk, 0);
	vector<double> sigmax(numk, 0);
	vector<double> sigmay(numk, 0);
	vector<int> clustersize(numk, 0);
	vector<double> inv(numk, 0); //to store 1/clustersize[k] values
	vector<double> distxy(sz, DBL_MAX);
	vector<double> distlab(sz, DBL_MAX);
	vector<double> distvec(sz, DBL_MAX);
	vector<double> maxlab(numk, 10 * 10);	 //THIS IS THE VARIABLE VALUE OF M, just start with 10
	vector<double> maxxy(numk, STEP * STEP); //THIS IS THE VARIABLE VALUE OF M, just start with 10

	double invxywt = 1.0 / (STEP * STEP); //NOTE: this is different from how usual SLIC/LKM works

	while (numitr < NUMITR)
	{
		//------
		//cumerr = 0;
		numitr++;
		//------

		distvec.assign(sz, DBL_MAX);
		for (int n = 0; n < numk; n++)
		{
			int y1 = max(0, (int)(kseedsy[n] - offset));
			int y2 = min(m_height, (int)(kseedsy[n] + offset));
			int x1 = max(0, (int)(kseedsx[n] - offset));
			int x2 = min(m_width, (int)(kseedsx[n] + offset));

			for (int y = y1; y < y2; y++)
			{
				for (int x = x1; x < x2; x++)
				{
					int i = y * m_width + x;
					//_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0 );

					double l = m_lvec[i];
					double a = m_avec[i];
					double b = m_bvec[i];

					distlab[i] = (l - kseedsl[n]) * (l - kseedsl[n]) +
								 (a - kseedsa[n]) * (a - kseedsa[n]) +
								 (b - kseedsb[n]) * (b - kseedsb[n]);

					distxy[i] = (x - kseedsx[n]) * (x - kseedsx[n]) +
								(y - kseedsy[n]) * (y - kseedsy[n]);

					//------------------------------------------------------------------------
					double dist = distlab[i] / maxlab[n] + distxy[i] * invxywt; //only varying m, prettier superpixels
					//double dist = distlab[i]/maxlab[n] + distxy[i]/maxxy[n];//varying both m and S
					//------------------------------------------------------------------------

					if (dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i] = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Assign the max color distance for a cluster
		//-----------------------------------------------------------------
		if (0 == numitr)
		{
			maxlab.assign(numk, 1);
			maxxy.assign(numk, 1);
		}
		{
			for (int i = 0; i < sz; i++)
			{
				if (maxlab[klabels[i]] < distlab[i])
					maxlab[klabels[i]] = distlab[i];
				if (maxxy[klabels[i]] < distxy[i])
					maxxy[klabels[i]] = distxy[i];
			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		sigmal.assign(numk, 0);
		sigmaa.assign(numk, 0);
		sigmab.assign(numk, 0);
		sigmax.assign(numk, 0);
		sigmay.assign(numk, 0);
		clustersize.assign(numk, 0);

		for (int j = 0; j < sz; j++)
		{
			int temp = klabels[j];
			//_ASSERT(klabels[j] >= 0);
			sigmal[klabels[j]] += m_lvec[j];
			sigmaa[klabels[j]] += m_avec[j];
			sigmab[klabels[j]] += m_bvec[j];
			sigmax[klabels[j]] += (j % m_width);
			sigmay[klabels[j]] += (j / m_width);

			clustersize[klabels[j]]++;
		}

		{
			for (int k = 0; k < numk; k++)
			{
				//_ASSERT(clustersize[k] > 0);
				if (clustersize[k] <= 0)
					clustersize[k] = 1;
				inv[k] = 1.0 / double(clustersize[k]); //computing inverse now to multiply, than divide later
			}
		}

		{
			for (int k = 0; k < numk; k++)
			{
				kseedsl[k] = sigmal[k] * inv[k];
				kseedsa[k] = sigmaa[k] * inv[k];
				kseedsb[k] = sigmab[k] * inv[k];
				kseedsx[k] = sigmax[k] * inv[k];
				kseedsy[k] = sigmay[k] * inv[k];
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
	//-------------------------------------
    double *kseedsl, *kseedsa, *kseedsb, *kseedsx, *kseedsy;
    kseedsl = new double[K];
    kseedsa = new double[K];
    kseedsb = new double[K];
    kseedsx = new double[K];
    kseedsy = new double[K];



	if (1) //LAB
	{
		DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
	}
	else //RGB
	{
        m_lvec = new double[sz];
        m_avec = new double[sz];
        m_bvec = new double[sz];
        // m_lvec = (double*)_mm_malloc(sz * sizeof(double), 256);
        // m_avec = (double*)_mm_malloc(sz * sizeof(double), 256);
        // m_bvec = (double*)_mm_malloc(sz * sizeof(double), 256);
        for (int i = 0; i < sz; i++) {
            m_lvec[i] = ubuff[i] >> 16 & 0xff;
            m_avec[i] = ubuff[i] >> 8 & 0xff;
            m_bvec[i] = ubuff[i] & 0xff;
        }
	}
	//--------------------------------------------------

	bool perturbseeds(true);
	// vector<double> edgemag(0);
	// if(perturbseeds) DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);
	GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy,
							numlabels, K, perturbseeds);

	int STEP = sqrt(double(sz) / double(K)) + 2.0; //adding a small value in the even the STEP size is too small.
	PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb, kseedsx
												, kseedsy, klabels, numlabels, STEP, 10);

	delete[] kseedsl;
    delete[] kseedsa;
    delete[] kseedsb;
    delete[] kseedsx;
    delete[] kseedsy;

	int *nlabels =(int*)_mm_malloc(sz * sizeof(int), 256);
	EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);
	{
		memcpy(klabels, nlabels, sizeof(int) * sz);
	}
	if (nlabels)
		_mm_free(nlabels);
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
	int* labels = (int*)_mm_malloc(sz * sizeof(int), 256);
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
		_mm_free(labels);

	if (img)
		delete[] img;

	return 0;
}