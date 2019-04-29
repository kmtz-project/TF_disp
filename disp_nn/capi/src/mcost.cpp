#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>

#define SAD_COST    1
#define SSD_COST    2
#define NCC_COST    3
#define CENSUS_COST 4


int sad(int * left_pix, int * right_pix, int win_size)
{

}

np::ndarray compute(np::ndarray left, np::ndarray rigth, int win_size, int max_disp, int type)
{
	const int X        = left.shape(1);
    const int Y        = left.shape(0);
	
	tuple shape = make_tuple(X, Y, max_disp);
	np::dtype dtype = np::dtype::get_builtin<int>();
	np::ndarray cost = np::zeros(shape, dtype);

    int * left_pix  = (int *) left.get_data();
    int * right_pix = (int *) right.get_data();

    for(int i = 0; i < X; i++)
    {
        for(int j = 0; j < Y; j++)
        {
            // epipolar line walk
            for(int k = j - max_disp - 1; k <= j; k++)
            {
                switch(type)
                {
                    case SAD_COST:
                        sad(left_pix, right_pix, win_size);
                        break;
                }
            }
        }
    }

    return cost;

}


BOOST_PYTHON_MODULE(mcost)
{
    np::initialize();
    def("sad", sad);
}