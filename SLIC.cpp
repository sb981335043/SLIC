
#include "SLIC.h"

#include <emmintrin.h>
#include <immintrin.h>
//#include <mpi.h>
#include <omp.h>
#include <smmintrin.h>

#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

typedef chrono::high_resolution_clock Clock;

// For superpixels
const int dx4[4] = {-1, 0, 1, 0};
const int dy4[4] = {0, -1, 0, 1};
// const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
// const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

// For supervoxels
const int dx10[10] = {-1, 0, 1, 0, -1, 1, 1, -1, 0, 0};
const int dy10[10] = {0, -1, 0, 1, -1, -1, 1, 1, 0, 0};
const int dz10[10] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 1};

// For simd permute
const __m256i K_PERM_VEC = _mm256_setr_epi32(1, 3, 5, 7, 0, 2, 4, 6);

int world_size;
int world_rank;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

SLIC::SLIC() {
    m_lvec = NULL;
    m_avec = NULL;
    m_bvec = NULL;

    m_lvecvec = NULL;
    m_avecvec = NULL;
    m_bvecvec = NULL;
}

SLIC::~SLIC() {
    if (m_lvec) _mm_free(m_lvec);
    if (m_avec) _mm_free(m_avec);
    if (m_bvec) _mm_free(m_bvec);

    if (m_lvecvec) {
        for (int d = 0; d < m_depth; d++) delete[] m_lvecvec[d];
        delete[] m_lvecvec;
    }
    if (m_avecvec) {
        for (int d = 0; d < m_depth; d++) delete[] m_avecvec[d];
        delete[] m_avecvec;
    }
    if (m_bvecvec) {
        for (int d = 0; d < m_depth; d++) delete[] m_bvecvec[d];
        delete[] m_bvecvec;
    }
}

double look_up_table[256];

//==============================================================================
///	RGB2XYZ
///
/// sRGB (D65 illuninant assumption) to XYZ conversion
//==============================================================================
inline void SLIC::RGB2XYZ(const int& sR, const int& sG, const int& sB,
                          double& X, double& Y, double& Z) {
    double r = look_up_table[sR];
    double g = look_up_table[sG];
    double b = look_up_table[sB];

    X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

//===========================================================================
///	RGB2LAB
//===========================================================================
void SLIC::RGB2LAB(const int& sR, const int& sG, const int& sB, double& lval,
                   double& aval, double& bval) {
    //------------------------
    // sRGB to XYZ conversion
    //------------------------
    double X, Y, Z;
    RGB2XYZ(sR, sG, sB, X, Y, Z);

    //------------------------
    // XYZ to LAB conversion
    //------------------------
    const double epsilon = 0.008856;  // actual CIE standard
    const double kappa = 903.3;       // actual CIE standard

    double Xr = 0.950456;  // reference white
    // double Yr = 1.0;    // reference white
    double Zr = 1.088754;  // reference white

    const double xr = X / Xr;
    const double yr = Y;
    const double zr = Z / Zr;

    double fx, fy, fz;
    if (xr > epsilon)
        // fx = pow(xr, 1.0 / 3.0);
        fx = cbrt(xr);
    else
        // fx = (kappa * xr + 16.0) / 116.0;
        fx = (kappa / 116.0) * xr + 16.0 / 116.0;
    if (yr > epsilon)
        // fy = pow(yr, 1.0 / 3.0);
        fy = cbrt(yr);
    else
        fy = (kappa / 116.0) * yr + 16.0 / 116.0;
    if (zr > epsilon)
        // fz = pow(zr, 1.0 / 3.0);
        fz = cbrt(zr);
    else
        fz = (kappa / 116.0) * zr + 16.0 / 116.0;

    lval = 116.0 * fy - 16.0;
    aval = 500.0 * (fx - fy);
    bval = 200.0 * (fy - fz);
}

//===========================================================================
///	DoRGBtoLABConversion
///
///	For whole image: overlaoded floating point version
//===========================================================================
void SLIC::DoRGBtoLABConversion(const unsigned int*& ubuff, double*& lvec,
                                double*& avec, double*& bvec) {
    int sz = m_width * m_height;
    lvec = (double*)_mm_malloc(sz * sizeof(double), 256);
    avec = (double*)_mm_malloc(sz * sizeof(double), 256);
    bvec = (double*)_mm_malloc(sz * sizeof(double), 256);

#pragma omp parallel for
    for (int i = 0; i < 256; ++i) {
        if (i <= 10)
            look_up_table[i] = i / 255.0 / 12.92;
        else
            look_up_table[i] = pow((i / 255.0 + 0.055) / 1.055, 2.4);
    }

#pragma omp parallel for
    for (int j = 0; j < sz; j++) {
        int r = (ubuff[j] >> 16) & 0xFF;
        int g = (ubuff[j] >> 8) & 0xFF;
        int b = (ubuff[j]) & 0xFF;

        RGB2LAB(r, g, b, lvec[j], avec[j], bvec[j]);
    }
}

//==============================================================================
///	DetectLabEdges
//==============================================================================
void SLIC::DetectLabEdges(const double* lvec, const double* avec,
                          const double* bvec, const int& width,
                          const int& height, vector<double>& edges) {
    int sz = width * height;

    edges.resize(sz, 0);
#pragma omp parallel for
    for (int j = 1; j < height - 1; j++) {
        for (int k = 1; k < width - 1; k++) {
            int i = j * width + k;

            double dx =
                (lvec[i - 1] - lvec[i + 1]) * (lvec[i - 1] - lvec[i + 1]) +
                (avec[i - 1] - avec[i + 1]) * (avec[i - 1] - avec[i + 1]) +
                (bvec[i - 1] - bvec[i + 1]) * (bvec[i - 1] - bvec[i + 1]);

            double dy = (lvec[i - width] - lvec[i + width]) *
                            (lvec[i - width] - lvec[i + width]) +
                        (avec[i - width] - avec[i + width]) *
                            (avec[i - width] - avec[i + width]) +
                        (bvec[i - width] - bvec[i + width]) *
                            (bvec[i - width] - bvec[i + width]);

            // edges[i] = (sqrt(dx) + sqrt(dy));
            edges[i] = (dx + dy);
        }
    }
}

//===========================================================================
///	PerturbSeeds
//===========================================================================
void SLIC::PerturbSeeds(double* kseedsl, double* kseedsa, double* kseedsb,
                        double* kseedsx, double* kseedsy, const int numseeds,
                        const vector<double>& edges) {
    const int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    const int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    for (int n = 0; n < numseeds; n++) {
        int ox = kseedsx[n];  // original x
        int oy = kseedsy[n];  // original y
        int oind = oy * m_width + ox;

        int storeind = oind;
        for (int i = 0; i < 8; i++) {
            int nx = ox + dx8[i];  // new x
            int ny = oy + dy8[i];  // new y

            if (nx >= 0 && nx < m_width && ny >= 0 && ny < m_height) {
                int nind = ny * m_width + nx;
                if (edges[nind] < edges[storeind]) {
                    storeind = nind;
                }
            }
        }
        if (storeind != oind) {
            kseedsx[n] = storeind % m_width;
            kseedsy[n] = storeind / m_width;
            kseedsl[n] = m_lvec[storeind];
            kseedsa[n] = m_avec[storeind];
            kseedsb[n] = m_bvec[storeind];
        }
    }
}

//===========================================================================
///	GetLABXYSeeds_ForGivenK
///
/// The k seed values are taken as uniform spatial pixel samples.
//===========================================================================
void SLIC::GetLABXYSeeds_ForGivenK(double* kseedsl, double* kseedsa,
                                   double* kseedsb, double* kseedsx,
                                   double* kseedsy, int& numk, const int& K,
                                   const bool& perturbseeds,
                                   const vector<double>& edgemag) {
    numk = 0;
    int sz = m_width * m_height;
    double step = sqrt(double(sz) / double(K));
    int T = step;
    int xoff = step / 2;
    int yoff = step / 2;

    // int n(0);
    int r(0);
    for (int y = 0; y < m_height; y++) {
        int Y = y * step + yoff;
        if (Y > m_height - 1) break;

        for (int x = 0; x < m_width; x++) {
            // int X = x*step + xoff;//square grid
            int X = x * step + (xoff << (r & 0x1));  // hex grid
            if (X > m_width - 1) break;

            int i = Y * m_width + X;

            //_ASSERT(n < K);

            kseedsl[numk] = m_lvec[i];
            kseedsa[numk] = m_avec[i];
            kseedsb[numk] = m_bvec[i];
            kseedsx[numk] = X;
            kseedsy[numk] = Y;
            ++numk;
        }
        r++;
    }

    if (perturbseeds) {
        PerturbSeeds(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, numk,
                     edgemag);
    }
}

//===========================================================================
///	PerformSuperpixelSegmentation_VariableSandM
///
///	Magic SLIC - no parameters
///
///	Performs k mean segmentation. It is fast because it looks locally, not
/// over the entire image.
/// This function picks the maximum value of color distance as compact factor
/// M and maximum pixel distance as grid step size S from each cluster (13 April
/// 2011). So no need to input a constant value of M and S. There are two clear
/// advantages:
///
/// [1] The algorithm now better handles both textured and non-textured regions
/// [2] There is not need to set any parameters!!!
///
/// SLICO (or SLIC Zero) dynamically varies only the compactness factor S,
/// not the step size S.
//===========================================================================
void SLIC::PerformSuperpixelSegmentation_VariableSandM(
    double* kseedsl, double* kseedsa, double* kseedsb, double* kseedsx,
    double* kseedsy, int* klabels, const int numk, const int& STEP,
    const int& NUMITR) {
    int sz = m_width * m_height;
    // const int numk = kseedsl.size();
    // double cumerr(99999.9);
    int numitr(0);
    //----------------
    int offset = STEP;
    if (STEP < 10) offset = STEP * 1.5;
    //----------------

    vector<double> sigmal(numk, 0);
    vector<double> sigmaa(numk, 0);
    vector<double> sigmab(numk, 0);
    vector<double> sigmax(numk, 0);
    vector<double> sigmay(numk, 0);
    vector<int> clustersize(numk, 0);
    vector<double> inv(numk, 0);  // to store 1/clustersize[k] values
    // vector<double> distxy(sz, DBL_MAX);
    // double *distxy = (double*)_mm_malloc(sz*sizeof(double),256);
    // not double max but large enough
    // memset(distxy,0x7F,sz*sizeof(double));
    double* distlab = (double*)_mm_malloc(sz * sizeof(double), 256);
#pragma omp parallel for
    for (int i = 0; i < sz; ++i) {
        // distxy[i] = DBL_MAX;
        distlab[i] = DBL_MAX;
    }
    double* distvec = (double*)_mm_malloc(sz * sizeof(double), 256);
    // double *res_unpack = (double*)_mm_malloc(16*sizeof(double),256);
    // memset(distvec,0x7F,sz*sizeof(double));
    vector<double> maxlab(
        numk, 10 * 10);  // THIS IS THE VARIABLE VALUE OF M, just start with 10
    // vector<double> maxxy(
    //     numk,
    //     STEP * STEP);  // THIS IS THE VARIABLE VALUE OF M, just start with 10

    double invxywt =
        1.0 /
        (STEP * STEP);  // NOTE: this is different from how usual SLIC/LKM works

    __m256d invxywt_vec = _mm256_set1_pd(invxywt);

    // vector<int> distidx(sz, -1);
    int* distidx = new int[sz];
    double* _maxlab[OMP_NUM_THREADS];
    // double* _maxxy[OMP_NUM_THREADS];
    double* _sigmal[OMP_NUM_THREADS];
    double* _sigmaa[OMP_NUM_THREADS];
    double* _sigmab[OMP_NUM_THREADS];
    double* _sigmax[OMP_NUM_THREADS];
    double* _sigmay[OMP_NUM_THREADS];
    int* _clustersize[OMP_NUM_THREADS];

// memset(distidx, 0, sizeof(int) * sz);
#pragma omp parallel for
    for (int i = 0; i < OMP_NUM_THREADS; ++i) {
        _maxlab[i] = new double[numk];
        // _maxxy[i] = new double[numk];
        _sigmal[i] = new double[numk];
        _sigmaa[i] = new double[numk];
        _sigmab[i] = new double[numk];
        _sigmax[i] = new double[numk];
        _sigmay[i] = new double[numk];
        _clustersize[i] = new int[numk];
    }
    while (numitr < NUMITR) {
        //------
        // cumerr = 0;
        numitr++;
        //------

#pragma omp parallel for
        for (int i = 0; i < sz; ++i) {
            distvec[i] = DBL_MAX;
        }

        // not double max but large enough
        // memset(distvec,0x7F,sz*sizeof(double));

        for (int n = 0; n < numk; n++) {
            const double _kseedsl = kseedsl[n];
            __m256d _kseedsl_vec = _mm256_set1_pd(kseedsl[n]);
            const double _kseedsa = kseedsa[n];
            __m256d _kseedsa_vec = _mm256_set1_pd(kseedsa[n]);
            const double _kseedsb = kseedsb[n];
            __m256d _kseedsb_vec = _mm256_set1_pd(kseedsb[n]);
            const double _kseedsx = kseedsx[n];
            __m256d _kseedsx_vec = _mm256_set1_pd(kseedsx[n]);
            const double _kseedsy = kseedsy[n];
            __m256d _kseedsy_vec = _mm256_set1_pd(kseedsy[n]);
            __m256d maxlab_vec = _mm256_set1_pd(maxlab[n]);


            const int y1 = max(0, (int)(_kseedsy - offset));
            const int y2 = min(m_height, (int)(_kseedsy + offset));

            const int x1 = max(0, (int)(_kseedsx - offset));
            const int x2 = min(m_width, (int)(_kseedsx + offset));

#pragma omp parallel for
            for (int y = y1; y < y2; y++) {
                double* res_unpack =
                    (double*)_mm_malloc(4 * sizeof(double), 256);
                for (int x = x1; x < x2;) {
                    const int i = y * m_width + x;
                    //_ASSERT( y < m_height && x < m_width && y >= 0 && x >= 0
                    //);
                    if ((i & 0x3) != 0 || x + 4 > x2) {
                        // not aligned part
                        const double l = m_lvec[i];
                        const double a = m_avec[i];
                        const double b = m_bvec[i];

                        const double _distlab =
                            (l - _kseedsl) * (l - _kseedsl) +
                            (a - _kseedsa) * (a - _kseedsa) +
                            (b - _kseedsb) * (b - _kseedsb);

                        const double _distxy = (x - _kseedsx) * (x - _kseedsx) +
                                               (y - _kseedsy) * (y - _kseedsy);

                        //------------------------------------------------------------------------
                        const double dist =
                            _distlab / maxlab[n] +
                            _distxy * invxywt;  // only varying m, prettier
                                                // superpixels
                        // double dist = distlab[i]/maxlab[n] +
                        // distxy[i]/maxxy[n];//varying both m and S
                        //------------------------------------------------------------------------

                        distlab[i] = _distlab;
                        // distxy[i] = _distxy;

                        if (dist < distvec[i]) {
                            distvec[i] = dist;
                            klabels[i] = n;
                        }
                        ++x;
                    } else {
                        // aligned part
                        __m256d l_vec = _mm256_load_pd(&m_lvec[i]);
                        __m256d a_vec = _mm256_load_pd(&m_avec[i]);
                        __m256d b_vec = _mm256_load_pd(&m_bvec[i]);
                        __m256d x_vec =
                            _mm256_set_pd((double)(x + 3), (double)(x + 2),
                                          (double)(x + 1), (double)(x));
                        __m256d y_vec = _mm256_set1_pd((double)y);
                        __m256d l_vec_t1 = _mm256_sub_pd(l_vec, _kseedsl_vec);
                        l_vec_t1 = _mm256_mul_pd(l_vec_t1, l_vec_t1);
                        __m256d a_vec_t1 = _mm256_sub_pd(a_vec, _kseedsa_vec);
                        a_vec_t1 = _mm256_mul_pd(a_vec_t1, a_vec_t1);
                        __m256d b_vec_t1 = _mm256_sub_pd(b_vec, _kseedsb_vec);
                        b_vec_t1 = _mm256_mul_pd(b_vec_t1, b_vec_t1);
                        __m256d _distlab_vec =
                            _mm256_add_pd(l_vec_t1, a_vec_t1);
                        _distlab_vec = _mm256_add_pd(_distlab_vec, b_vec_t1);
                        __m256d x_vec_t1 = _mm256_sub_pd(x_vec, _kseedsx_vec);
                        x_vec_t1 = _mm256_mul_pd(x_vec_t1, x_vec_t1);
                        __m256d y_vec_t1 = _mm256_sub_pd(y_vec, _kseedsy_vec);
                        y_vec_t1 = _mm256_mul_pd(y_vec_t1, y_vec_t1);
                        __m256d _distxy_vec = _mm256_add_pd(x_vec_t1, y_vec_t1);
                        __m256d dist_vec_t1 =
                            _mm256_div_pd(_distlab_vec, maxlab_vec);
                        __m256d dist_vec_t2 =
                            _mm256_mul_pd(_distxy_vec, invxywt_vec);
                        __m256d dist_vec =
                            _mm256_add_pd(dist_vec_t1, dist_vec_t2);

                        _mm256_store_pd(&distlab[i], _distlab_vec);
                        // _mm256_store_pd(&distxy[i], _distxy_vec);

                        __m256d distvec_vec = _mm256_load_pd(&distvec[i]);
                        __m256d cmp_res_vec =
                            _mm256_cmp_pd(dist_vec, distvec_vec, _CMP_LT_OQ);
                        // int move_mask = _mm256_movemask_pd(cmp_res_vec);
                        // distvec_vec = _mm256_blend_pd(distvec_vec, dist_vec,
                        // move_mask);
                        distvec_vec = _mm256_blendv_pd(distvec_vec, dist_vec,
                                                       cmp_res_vec);
                        _mm256_store_pd(&distvec[i], distvec_vec);

                        __m256i permuted_vec = _mm256_permutevar8x32_epi32(
                            _mm256_castpd_si256(cmp_res_vec), K_PERM_VEC);
                        __m128i cmp_int_vec =
                            _mm256_castsi256_si128(permuted_vec);

                        // __m256 cmp_ps_vec = _mm256_castpd_ps(cmp_res_vec);
                        // __m128 cmp_lo_vec = _mm256_extractf128_ps(cmp_ps_vec,
                        // 0);
                        // __m128 cmp_hi_vec = _mm256_extractf128_ps(cmp_ps_vec,
                        // 1);
                        // __m128i cmp_int_vec =
                        // _mm_castps_si128(_mm_shuffle_ps(cmp_lo_vec,
                        // cmp_hi_vec, 1 + (3<<2) + (1<<4) + (3<<6)));

                        __m128i n_vec = _mm_set1_epi32(n);
                        _mm_maskstore_epi32(&klabels[i], cmp_int_vec, n_vec);
                        //__m128i klabels_vec =
                        //_mm_load_si128((__m128i*)&klabels[i]);
                        // klabels_vec = _mm_blend_epi32(klabels_vec, n_vec,
                        // move_mask);
                        //_mm_store_si128((__m128i*)&klabels[i], klabels_vec);
                        //__m128i cmp_int_vec = _mm256_cvtpd_epi32(cmp_res_vec);
                        // _mm256_store_pd(res_unpack, cmp_res_vec);
                        // klabels[i]=res_unpack[0]==0?klabels[i]:n;
                        // klabels[i+1]=res_unpack[1]==0?klabels[i+1]:n;
                        // klabels[i+2]=res_unpack[2]==0?klabels[i+2]:n;
                        // klabels[i+3]=res_unpack[3]==0?klabels[i+3]:n;
                        x += 4;
                    }
                }
                _mm_free(res_unpack);
            }
        }
        //-----------------------------------------------------------------
        // Assign the max color distance for a cluster
        //-----------------------------------------------------------------
        // if(0 == numitr)	// I think it won't be executed
        // {
        // 	maxlab.assign(numk,1);
        // 	maxxy.assign(numk,1);
        // }
#pragma omp parallel for
        for (int i = 0; i < OMP_NUM_THREADS; ++i) {
            memset(_maxlab[i], 0, sizeof(double) * numk);
            // memset(_maxxy[i], 0, sizeof(double) * numk);
        }
#pragma omp parallel for

        for (int i = 0; i < sz; i++) {
            int idx = omp_get_thread_num();
            if (_maxlab[idx][klabels[i]] < distlab[i])
                _maxlab[idx][klabels[i]] = distlab[i];
            // if (_maxxy[idx][klabels[i]] < distxy[i])
            //     _maxxy[idx][klabels[i]] = distxy[i];
        }
#pragma omp parallel for
        for (int i = 0; i < numk; ++i)
            for (int j = 0; j < OMP_NUM_THREADS; ++j) {
                if (maxlab[i] < _maxlab[j][i]) maxlab[i] = _maxlab[j][i];
                // if (maxxy[i] < _maxxy[j][i])
                //     maxxy[i] = _maxxy[j][i];
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

#pragma omp parallel for
        for (int i = 0; i < OMP_NUM_THREADS; ++i) {
            memset(_sigmal[i], 0, sizeof(double) * numk);
            memset(_sigmaa[i], 0, sizeof(double) * numk);
            memset(_sigmab[i], 0, sizeof(double) * numk);
            memset(_sigmax[i], 0, sizeof(double) * numk);
            memset(_sigmay[i], 0, sizeof(double) * numk);
            memset(_clustersize[i], 0, sizeof(int) * numk);
        }
#pragma omp parallel
        {
            int idx = omp_get_thread_num();
#pragma omp for

            for (int j = 0; j < sz; j++) {
                _sigmal[idx][klabels[j]] += m_lvec[j];
                _sigmaa[idx][klabels[j]] += m_avec[j];
                _sigmab[idx][klabels[j]] += m_bvec[j];
                _sigmax[idx][klabels[j]] += (j % m_width);
                _sigmay[idx][klabels[j]] += (j / m_width);

                _clustersize[idx][klabels[j]]++;
            }
        }
#pragma omp parallel for
        for (int i = 0; i < numk; ++i)
            for (int j = 0; j < OMP_NUM_THREADS; ++j) {
                sigmal[i] += _sigmal[j][i];
                sigmaa[i] += _sigmaa[j][i];
                sigmab[i] += _sigmab[j][i];
                sigmax[i] += _sigmax[j][i];
                sigmay[i] += _sigmay[j][i];
                clustersize[i] += _clustersize[j][i];
            }


#pragma omp parallel
        {
#pragma omp for
            for (int k = 0; k < numk; k++) {
                //_ASSERT(clustersize[k] > 0);
                if (clustersize[k] <= 0) clustersize[k] = 1;
                inv[k] = 1.0 /
                         double(clustersize[k]);  // computing inverse now to
                                                  // multiply, than divide later
            }
        }

#pragma omp parallel
        {
#pragma omp for
            for (int k = 0; k < numk; k++) {
                kseedsl[k] = sigmal[k] * inv[k];
                kseedsa[k] = sigmaa[k] * inv[k];
                kseedsb[k] = sigmab[k] * inv[k];
                kseedsx[k] = sigmax[k] * inv[k];
                kseedsy[k] = sigmay[k] * inv[k];
            }
        }
    }
    delete[] distidx;
#pragma omp parallel for
    for (int i = 0; i < OMP_NUM_THREADS; ++i) {
        delete[] _maxlab[i];
        // delete[] _maxxy[i];
        delete[] _sigmal[i];
        delete[] _sigmaa[i];
        delete[] _sigmab[i];
        delete[] _sigmax[i];
        delete[] _sigmay[i];
        delete[] _clustersize[i];
    }

    _mm_free(distlab);
    // _mm_free(distxy);
    _mm_free(distvec);
    //_mm_free(res_unpack);
}

//===========================================================================
///	SaveSuperpixelLabels2PGM
///
///	Save labels to PGM in raster scan order.
//===========================================================================
void SLIC::SaveSuperpixelLabels2PPM(char* filename, int* labels,
                                    const int width, const int height) {
    FILE* fp;
    char header[20];

    fp = fopen(filename, "wb");

    // write the PPM header info, such as type, width, height and maximum
    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    // write the RGB data
    unsigned char* rgb = new unsigned char[(width) * (height)*3];
    int k = 0;
    unsigned char c = 0;
    for (int i = 0; i < (height); i++) {
        for (int j = 0; j < (width); j++) {
            c = (unsigned char)(labels[k]);
            rgb[i * (width)*3 + j * 3 + 2] = labels[k] >> 16 & 0xff;  // r
            rgb[i * (width)*3 + j * 3 + 1] = labels[k] >> 8 & 0xff;   // g
            rgb[i * (width)*3 + j * 3 + 0] = labels[k] & 0xff;        // b

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
///		2. if a certain component is too small, assigning the previously
/// found 		    adjacent label to this component, and not
/// incrementing the label.
//===========================================================================
void SLIC::EnforceLabelConnectivity(
    int* labels,  // input labels that need to be corrected to remove
                        // stray labels
    const int& width, const int& height,
    int* nlabels,    // new labels
    int& numlabels,  // the number of labels changes in the end if segments are
                     // removed
    const int& K)    // the number of superpixels desired by the user
{
    //	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
    //	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

    const int dx4[4] = {-1, 0, 1, 0};
    const int dy4[4] = {0, -1, 0, 1};

    const int sz = width * height;

#pragma omp parallel for
    for(int i=0; i<height; ++i)
    {
        labels[i*width]|=(1<<31);
        labels[i*width+width-1]|=(1<<30);
    }

    const int SUPSZ = sz / K;
    memset(nlabels, -1, sizeof(int) * sz);
    int label(0);
    // int* xvec = new int[sz];
    // int* yvec = new int[sz];
    int *vec = new int[sz];
    int adjlabel(0);  // adjacent label
    for (int i = 0; i < sz; ++i) {
        if (0 > nlabels[i]) {
            nlabels[i] = label;
            //--------------------
            // Start a new segment
            //--------------------
            vec[0]=i;
            //-------------------------------------------------------
            // Quickly find an adjacent label for use later if needed
            //-------------------------------------------------------
            {
                for (int n = 0; n < 4; n++) {
                    int nindex = vec[0] + dx4[n] + dy4[n] * width;
                    if ((nindex>0&&nindex<sz)&&!(((labels[vec[0]]|labels[nindex])&0xC0000000)==0xC0000000)) {
                        if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
                    }
                }
            }

            int count(1);
            for (int c = 0; c < count; c++) {
                for (int n = 0; n < 4; n++) {
                    int nindex = vec[c] + dx4[n] + dy4[n] * width;

                    if ((nindex>0&&nindex<sz)&&!(((labels[vec[c]]|labels[nindex])&0xC0000000)==0xC0000000)) {
                        if (0 > nlabels[nindex] &&
                            (labels[i]&0x3FFFFFFF) == (labels[nindex]&0x3FFFFFFF)) {
                            vec[count] = nindex;
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
            if (count <= SUPSZ >> 2) {
                // #pragma omp parallel for
                for (int c = 0; c < count; c++) {
                    int ind = vec[c];
                    nlabels[ind] = adjlabel;
                }
                label--;
            }
            label++;
        }
    }
    numlabels = label;

    // if (xvec) delete[] xvec;
    // if (yvec) delete[] yvec;
    delete[] vec;
}

//===========================================================================
///	PerformSLICO_ForGivenK
///
/// Zero parameter SLIC algorithm for a given number K of superpixels.
//===========================================================================
void SLIC::PerformSLICO_ForGivenK(
    const unsigned int* ubuff, const int width, const int height, int* klabels,
    int& numlabels,
    const int& K,     // required number of superpixels
    const double& m)  // weight given to spatial distance
{
    double *kseedsl, *kseedsa, *kseedsb, *kseedsx, *kseedsy;

    //--------------------------------------------------
    m_width = width;
    m_height = height;
    int sz = m_width * m_height;
    //--------------------------------------------------
    // if(0 == klabels) klabels = new int[sz];
    memset(klabels, -1, sizeof(int) * sz);
    // for( int s = 0; s < sz; s++ ) klabels[s] = -1;
    //--------------------------------------------------

    double step = sqrt(double(sz) / double(K));
    kseedsl = (double*)_mm_malloc(
        (m_width / step + 1) * (m_height / step + 1) * sizeof(double), 256);
    kseedsa = (double*)_mm_malloc(
        (m_width / step + 1) * (m_height / step + 1) * sizeof(double), 256);
    kseedsb = (double*)_mm_malloc(
        (m_width / step + 1) * (m_height / step + 1) * sizeof(double), 256);
    kseedsx = (double*)_mm_malloc(
        (m_width / step + 1) * (m_height / step + 1) * sizeof(double), 256);
    kseedsy = (double*)_mm_malloc(
        (m_width / step + 1) * (m_height / step + 1) * sizeof(double), 256);

    if (1)  // LAB
    {
        DoRGBtoLABConversion(ubuff, m_lvec, m_avec, m_bvec);
    } else  // RGB
    {
        // m_lvec = new double[sz];
        // m_avec = new double[sz];
        // m_bvec = new double[sz];
        m_lvec = (double*)_mm_malloc(sz * sizeof(double), 256);
        m_avec = (double*)_mm_malloc(sz * sizeof(double), 256);
        m_bvec = (double*)_mm_malloc(sz * sizeof(double), 256);
        for (int i = 0; i < sz; i++) {
            m_lvec[i] = ubuff[i] >> 16 & 0xff;
            m_avec[i] = ubuff[i] >> 8 & 0xff;
            m_bvec[i] = ubuff[i] & 0xff;
        }
    }
    //--------------------------------------------------

    bool perturbseeds(true);
    vector<double> edgemag(0);

    if (perturbseeds)
        DetectLabEdges(m_lvec, m_avec, m_bvec, m_width, m_height, edgemag);

    GetLABXYSeeds_ForGivenK(kseedsl, kseedsa, kseedsb, kseedsx, kseedsy,
                            numlabels, K, perturbseeds, edgemag);

    int STEP =
        sqrt(double(sz) / double(K)) +
        2.0;  // adding a small value in the even the STEP size is too small.
    PerformSuperpixelSegmentation_VariableSandM(kseedsl, kseedsa, kseedsb,
                                                kseedsx, kseedsy, klabels,
                                                numlabels, STEP, 10);
    // numlabels = kseedsl.size();

    _mm_free(kseedsl);
    _mm_free(kseedsa);
    _mm_free(kseedsb);
    _mm_free(kseedsx);
    _mm_free(kseedsy);

    // int* nlabels = new int[sz];
    int* nlabels = (int*)_mm_malloc(sz * sizeof(int), 256);
    EnforceLabelConnectivity(klabels, m_width, m_height, nlabels, numlabels, K);
    {
        memcpy(klabels, nlabels, sizeof(int) * sz);
        // for(int i = 0; i < sz; i++) klabels[i] = nlabels[i];
    }
    if (nlabels) _mm_free(nlabels);
}

//===========================================================================
/// Load PPM file
///
///
//===========================================================================
void LoadPPM(char* filename, unsigned int** data, int* width, int* height) {
    char header[1024];
    FILE* fp = NULL;
    int line = 0;

    fp = fopen(filename, "rb");

    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
    sscanf(header, "%d %d\n", width, height);

    // read the maximum of pixels
    fgets(header, 20, fp);

    // get rgb data
    unsigned char* rgb = new unsigned char[(*width) * (*height) * 3];
    fread(rgb, (*width) * (*height) * 3, 1, fp);

    *data = new unsigned int[(*width) * (*height) * 4];
    int k = 0;
    for (int i = 0; i < (*height); i++) {
        for (int j = 0; j < (*width); j++) {
            unsigned char* p = rgb + i * (*width) * 3 + j * 3;
            // a ( skipped )
            (*data)[k] = p[2] << 16;  // r
            (*data)[k] |= p[1] << 8;  // g
            (*data)[k] |= p[0];       // b
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
int CheckLabelswithPPM(char* filename, int* labels, int width, int height) {
    char header[1024];
    FILE* fp = NULL;
    int line = 0, ground = 0;

    fp = fopen(filename, "rb");

    // read the image type, such as: P6
    // skip the comment lines
    while (line < 2) {
        fgets(header, 1024, fp);
        if (header[0] != '#') {
            ++line;
        }
    }
    // read width and height
    int w(0);
    int h(0);
    sscanf(header, "%d %d\n", &w, &h);
    if (w != width || h != height) return -1;

    // read the maximum of pixels
    fgets(header, 20, fp);

    // get rgb data
    unsigned char* rgb = new unsigned char[(w) * (h)*3];
    fread(rgb, (w) * (h)*3, 1, fp);

    int num = 0, k = 0;
    for (int i = 0; i < (h); i++) {
        for (int j = 0; j < (w); j++) {
            unsigned char* p = rgb + i * (w)*3 + j * 3;
            // a ( skipped )
            ground = p[2] << 16;  // r
            ground |= p[1] << 8;  // g
            ground |= p[0];       // b

            if (ground != labels[k]) num++;

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
//===========================================================================
int main(int argc, char** argv) {

    unsigned int* img = NULL;
    int width(0);
    int height(0);

    LoadPPM((char*)"input_image.ppm", &img, &width, &height);

    if (width == 0 || height == 0) return -1;

    int sz = width * height;
    // int* labels = new int[sz];
    int* labels = (int*)_mm_malloc(sz * sizeof(int), 256);
    int numlabels(0);
    SLIC slic;
    int m_spcount;
    double m_compactness;
    // m_spcount = 200;
	m_spcount = argc < 2 ? 400 : stoi(argv[1]);
    m_compactness = 10.0;

    auto startTime = Clock::now();

    slic.PerformSLICO_ForGivenK(
        img, width, height, labels, numlabels, m_spcount,
        m_compactness);  // for a given number K of superpixels

    auto endTime = Clock::now();
        auto compTime =
            chrono::duration_cast<chrono::microseconds>(endTime - startTime);
        cout << "Computing time=" << compTime.count() / 1000 << " ms" << endl;


    int num = CheckLabelswithPPM((char*)"check.ppm", labels, width, height);

        if (num < 0) {
            cout << "The result for labels is different from output_labels.ppm."
                 << endl;
        } else {
            cout << "There are " << num
                 << " points' labels are different from original file." << endl;
        }

    slic.SaveSuperpixelLabels2PPM((char*)"output_labels.ppm", labels, width,
                                  height);

    if (labels) _mm_free(labels);
    if (img) delete[] img;
    return 0;
}
