#define BOOST_PYTHON_STATIC_LIB
#define BOOST_NUMPY_STATIC_LIB

#include <boost/python.hpp>
#include <boost/python/def.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numpy.hpp>

#include <iostream>
#include <time.h>
#include "elas.h"
#include "image.h"
#include <math.h>

using namespace std;
//using namespace boost::python;
namespace np = boost::python::numpy;

// check whether machine is little endian
int littleendian() {
    int intval = 1;
    uchar *uval = (uchar *)&intval;
    return uval[0] == 1;
}

// write pfm image (added by DS 10/24/2013)
// 1-band PFM image, see http://netpbm.sourceforge.net/doc/pfm.html
void WriteFilePFM(float *data, int width, int height, const char* filename, float scalefactor=1/255.0) {
    // Open the file
    FILE *stream = fopen(filename, "wb");
    if (stream == 0) {
        fprintf(stderr, "WriteFilePFM: could not open %s\n", filename);
	exit(1);
    }

    // sign of scalefact indicates endianness, see pfms specs
    if (littleendian())
	scalefactor = -scalefactor;

    // write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
    fprintf(stream, "Pf\n%d %d\n%f\n", width, height, scalefactor);

    int n = width;
    // write rows -- pfm stores rows in inverse order!
    for (int y = height-1; y >= 0; y--) {
	float* ptr = data + y * width;
	// change invalid pixels (which seem to be represented as -10) to INF
	for (int x = 0; x < width; x++) {
	    if (ptr[x] < 0)
		ptr[x] = INFINITY;
	}
	if ((int)fwrite(ptr, sizeof(float), n, stream) != n) {
	    fprintf(stderr, "WriteFilePFM: problem writing data\n");
	    exit(1);
	}
    }
    
    // close file
    fclose(stream);
}

// compute disparities of pgm image input pair file_1, file_2
void compute (const char* file_1, const char* file_2, const char* outfile, int maxdisp, int no_interp, int cosine_weight, 
              np::ndarray conv_left_arr, np::ndarray conv_right_arr) {
    float* conv_left = (float *) conv_left_arr.get_data();
    float* conv_right = (float *) conv_right_arr.get_data();
    clock_t c0 = clock();

    // load images
    image<uchar> *I1,*I2;
    I1 = loadPGM(file_1);
    I2 = loadPGM(file_2);

    // check for correct size
    if (I1->width()<=0 || I1->height() <=0 || I2->width()<=0 || I2->height() <=0 ||
	I1->width()!=I2->width() || I1->height()!=I2->height()) {
	cout << "ERROR: Images must be of same size, but" << endl;
	cout << "       I1: " << I1->width() <<  " x " << I1->height() << 
	    ", I2: " << I2->width() <<  " x " << I2->height() << endl;
	delete I1;
	delete I2;
	return;    
    }

    // get image width and height
    int32_t width  = I1->width();
    int32_t height = I1->height();

    // allocate memory for disparity images
    const int32_t dims[3] = {width,height,width}; // bytes per line = width
    float* D1_data = (float*)malloc(width*height*sizeof(float));
    float* D2_data = (float*)malloc(width*height*sizeof(float));
  
    // process
    Elas::parameters param(Elas::MIDDLEBURY);
    if (no_interp) {
	//param = Elas::parameters(Elas::ROBOTICS);
	// don't use full 'robotics' setting, just the parameter to fill gaps
        param.ipol_gap_width = 3;
    }
    param.postprocess_only_left = false;
    param.disp_max = maxdisp;
    param.cosine_weight = cosine_weight;
    Elas elas(param);
    elas.process(I1->data,I2->data,D1_data,D2_data,dims,conv_left, conv_right);

    // added runtime output - DS 4/4/2013
    clock_t c1 = clock();
    double secs = (double)(c1 - c0) / CLOCKS_PER_SEC;
    printf("runtime: %.2fs  (%.2fs/MP)\n", secs, secs/(width*height/1000000.0));

    // save disparity image

    WriteFilePFM(D1_data, width, height, outfile, 1.0/maxdisp);

    // free memory
    delete I1;
    delete I2;
    free(D1_data);
    free(D2_data);
}

BOOST_PYTHON_MODULE(elasCNN)
{
    np::initialize();
    def("compute", compute);
}
