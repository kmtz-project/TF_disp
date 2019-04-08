#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>

// OpenCV 4.0.1 is required
#include <opencv2/core.hpp>
// -------------------------

using namespace boost::python;
namespace np = boost::python::numpy;

np::ndarray compute(np::ndarray data)
{
	const int max_disp = data.shape(2);
	const int n        = data.shape(0);
	const int m        = data.shape(1) + max_disp;
	

	tuple out_shape = make_tuple(n, m);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray disp_img = np::zeros(out_shape, dtype);
	
	int shape[3] = { n, m - max_disp, max_disp };
	cv::Mat predict  = cv::Mat(3, shape, CV_32F, data.get_data());
	//cv::Mat disp_img = cv::Mat(n, m, CV_8U);

	for (int i = 0; i < n; i++) 
	{
		for (int j = max_disp; j < m; j++)
		{
			//float element = extract<float>(data[0][0][i]);

			int min_loc, max_loc;
			double min, max;

			float * row = predict.ptr<float>(i, j - max_disp);
			std::vector<float> disp_row(row, row + max_disp);
			int i_max = std::max_element(disp_row.begin(), disp_row.end()) - disp_row.begin();

			disp_img[i][j] = i_max;
		}
	}

	return disp_img;
}

BOOST_PYTHON_MODULE(sgbm)
{
    np::initialize();
    def("compute", compute);
}
