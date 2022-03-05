#if !defined(_SLIC_H_INCLUDED_)
#define _SLIC_H_INCLUDED_

#include <vector>
#include <string>
#include <algorithm>
using namespace std;

class SLIC
{
public:
	SLIC();
	virtual ~SLIC();

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
		vector<double> &kseedsl,
		vector<double> &kseedsa,
		vector<double> &kseedsb,
		vector<double> &kseedsx,
		vector<double> &kseedsy,
		int *klabels,
		const int &STEP,
		const int &NUMITR);

	// 计算单个点的梯度，对于一个初始seed计算周围八个点即可
	double DetectLABPixelEdge(
		const int &i);
	//============================================================================
	// Pick seeds for superpixels when number of superpixels is input.
	//============================================================================
	void GetLABXYSeeds_ForGivenK(
		vector<double> &kseedsl,
		vector<double> &kseedsa,
		vector<double> &kseedsb,
		vector<double> &kseedsx,
		vector<double> &kseedsy,
		const int &STEP,
		const bool &perturbseeds);


	//============================================================================
	// sRGB to CIELAB conversion for 2-D images
	//============================================================================
	void DoRGBtoLABConversion(
		const unsigned int *&ubuff,
		double *&lvec,
		double *&avec,
		double *&bvec);

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

	double *m_lvec;
	double *m_avec;
	double *m_bvec;

	double **m_lvecvec;
	double **m_avecvec;
	double **m_bvecvec;
};

#endif // !defined(_SLIC_H_INCLUDED_)