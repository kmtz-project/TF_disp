#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>

// OpenCV 4.0.1 is required
#include <opencv2/core.hpp>
// -------------------------

#include <iostream>

using namespace boost::python;
namespace np = boost::python::numpy;

np::ndarray compute(np::ndarray data, int cross_size, float P)
{
	const int max_disp = data.shape(2);
	const int n        = data.shape(0);
	const int m        = data.shape(1) + max_disp;
	
	tuple out_shape = make_tuple(n, m);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray disp_img = np::zeros(out_shape, dtype);
	
	int shape[3] = { n, m - max_disp, max_disp };
	cv::Mat predict   = cv::Mat(3, shape, CV_32F, data.get_data());
	cv::Mat sgbm_cost = cv::Mat(3, shape, CV_32F);
	//cv::Mat disp_img = cv::Mat(n, m, CV_8U);

	float * row = nullptr;
	float * cost_row = nullptr;

	//std::vector<float> cost_left(max_disp, 0);
	//std::vector<float> cost_up(max_disp, 0);
	//std::vector<float> cost_down(max_disp, 0);

	for (int i = 0; i < n; i++) 
	{
		for (int j = max_disp; j < m; j++)
		{
			//float element = extract<float>(data[0][0][i]);

			row      = predict.ptr<float>(i, j - max_disp);
			cost_row = sgbm_cost.ptr<float>(i, j - max_disp);

			for (int d = 0; d < max_disp; d++)
			{
				float cost_h  = 0;
				float cost_v  = 0;
				float cost_rd = 0;
				float cost_ld = 0;
				int v_ptr     = 0;
				int h_ptr     = 0;
				int ld_ptr_y  = 0;

				for (int r = 0; r < cross_size; r++)
				{
					h_ptr = j - max_disp - cross_size / 2 + r;
					v_ptr = i - cross_size / 2 + r;

					ld_ptr_y = j - max_disp + (cross_size - h_ptr);

					if (h_ptr < 0) h_ptr = 0;
					if (h_ptr >= m - max_disp) h_ptr = m - max_disp - 1;
					if (v_ptr < 0) v_ptr = 0;
					if (v_ptr >= n) v_ptr = n - 1;
					if (ld_ptr_y < 0) ld_ptr_y = 0;
					if (ld_ptr_y >= m - max_disp) ld_ptr_y = m - max_disp - 1;
									   					 				  					
					float * row_h = predict.ptr<float>(i, h_ptr);
					float * row_v = predict.ptr<float>(v_ptr, j - max_disp);

					float * row_rd = predict.ptr<float>(v_ptr, h_ptr);
					float * row_ld = predict.ptr<float>(v_ptr, ld_ptr_y);

					/*float row_h_prev = row_h[d];
					float row_h_next = row_h[d];
					
					if (d - 1 >= 0) row_h_prev = row_h[d - 1] + P;
					if (d + 1 < max_disp) row_h_next = row_h[d + 1] + P;

					float h_array[3] = {
						row_h[d],
						row_h_prev,
						row_h_next
					}; 

					float row_v_prev = row_v[d];
					float row_v_next = row_v[d];

					if (d - 1 >= 0) row_v_prev = row_v[d - 1] + P;
					if (d + 1 < max_disp) row_v_next = row_v[d + 1] + P;

					float v_array[3] = {
						row_v[d],
						row_v_prev,
						row_v_next
					};*/

					cost_h  += row_h[d];//*std::min_element(h_array, h_array + 3);
					cost_v  += row_v[d];//*std::min_element(v_array, v_array + 3);
					cost_rd += row_rd[d];
					cost_ld += row_ld[d];
				}

				//printf("cost[%d][%d][%d] = %f\n", i, h_ptr, d, cost_h);
				cost_row[d] = (cost_h + cost_v + (cost_rd + cost_ld)/3)/4;
			}

			//row = sgbm_cost.ptr<float>(i, j - max_disp);

			std::vector<float> disp_row(cost_row, cost_row + max_disp);
			int i_min = std::min_element(disp_row.begin(), disp_row.end()) - disp_row.begin();

			disp_img[i][j] = i_min;
			
		}
		
		printf("\r[i] = [%d/%d]", i, n);
	}

	printf("\n");

	return disp_img;
}

BOOST_PYTHON_MODULE(sgbm)
{
    np::initialize();
    def("compute", compute);
}
