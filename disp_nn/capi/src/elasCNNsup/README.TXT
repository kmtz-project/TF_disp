Modified version of libelas. This version allows using libelas in python environment. 

This code is based on modified version of libelas for Middlebury stereo evaluation written by Daniel Scharstein, June 5 2014.

This modified version of libelas can be distributed under GNU GPL License. 
See original README.TXT for license and please check the site of original version of libelas: http://www.cvlibs.net/software/libelas/.

Extract from original libeals README.TXT
----------------------------------------

####################################################################################
# Copyright 2011. All rights reserved.                                             #
# Institute of Measurement and Control Systems                                     #
# Karlsruhe Institute of Technology, Germany                                       #
#                                                                                  #
# This file is part of libelas.                                                    #
# Authors:  Andreas Geiger                                                         #
#           Please send any bugreports to geiger@kit.edu                           #
#                                                                                  #
# libelas is free software; you can redistribute it and/or modify it under the     #
# terms of the GNU General Public License as published by the Free Software        #
# Foundation; either version 3 of the License, or any later version.               #
#                                                                                  #
# libelas is distributed in the hope that it will be useful, but WITHOUT ANY       #
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A  #
# PARTICULAR PURPOSE. See the GNU General Public License for more details.         #
#                                                                                  #
# You should have received a copy of the GNU General Public License along with     #
# libelas; if not, write to the Free Software Foundation, Inc., 51 Franklin        #
# Street, Fifth Floor, Boston, MA 02110-1301, USA                                  #
####################################################################################

+++++++++++++++++++++++++++++++++++
+          INTRODUCTION           +
+++++++++++++++++++++++++++++++++++

Libelas (LIBrary for Efficient LArge-scale Stereo matching) is a cross-platfrom C++
library with MATLAB wrappers for computing disparity maps of large images. Input
is a rectified greyscale stereo image pair of same size. Output are the corresponding
disparity maps.

If you distribute a software that uses libelas, you have to distribute it under GPL
with the source code. Another option is to contact us to purchase a commercial license.

If you find this software useful or if you use this software for your research,
we would be happy if you cite the following related publication:

@INPROCEEDINGS{Geiger10,
 author = {Andreas Geiger and Martin Roser and Raquel Urtasun},
 title = {Efficient Large-Scale Stereo Matching},
 booktitle = {Asian Conference on Computer Vision},
 year = {2010},
 month = {November},
 address = {Queenstown, New Zealand}
}

