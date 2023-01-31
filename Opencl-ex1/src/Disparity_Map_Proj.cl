//////////////////////////////////////////////////////////////////////////////
// GPU Project Group 1: Disparity Map
// File Name : Disparity_Map_Proj.cl
// Author : Rucha Golwalkar st179996@stud.uni-stuttgart.de , University of Stuttgart
//			Gopal Panigrahi st179549@stud.uni-stuttgart.de , University of Stuttgart
//			Mandar Kharde   st179460@stud.uni-stuttgart.de , University of Stuttgart
//////////////////////////////////////////////////////////////////////////////

#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp> // Hack to make syntax highlighting in Eclipse work
#endif

int getIndexGlobal(size_t countX, int i, int j) 
{
	return j * countX + i;
}

// Read value from global array a, return 0 if outside image

const sampler_t sampling = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

float getValueImage(__read_only image2d_t img, int i, int j)
{
	return read_imagef(img, sampling, (int2) { i, j }).x;
}


// Kernel to get a Disparity Map using SAD algorithm

__kernel void DisparityMap1(__read_only image2d_t d_input_Left, __read_only image2d_t d_input_Right, __global float* d_output)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	int window_size = 13;
	int window_range = window_size/2 ;
	float SAD = 0.0;
	float min_SAD = 10000.0;
	float disparity = 0.0;
	int Max_Disparity = 100;


	for (int d = 0; d <= Max_Disparity; d++)
	{
		float SAD = 0;
		for (int imageWidth = 1- window_range; imageWidth < window_range; imageWidth++)
		{
			for (int imageHeight = 1 - window_range; imageHeight < window_range; imageHeight++)
			{
				float diff;
				diff = (getValueImage(d_input_Left, imageWidth + i, imageHeight + j)) - (getValueImage(d_input_Right, imageWidth + i - d, imageHeight + j));;
				if (diff < 0)
					SAD = SAD + (-1 * diff);	// To get absolute value ofthe diffrence 
				else
					SAD = SAD + diff; 
			}
		}

		if (SAD < min_SAD)
		{
			min_SAD = SAD;
			disparity = (float)(d) / Max_Disparity;

		}

	}
	d_output[getIndexGlobal(countX, i, j)] = disparity;		// Update output 
}

// Kernel to get a Disparity Map using SSD algorithm

__kernel void DisparityMap2(__read_only image2d_t d_input_Left, __read_only image2d_t d_input_Right, __global float* d_output)
{
	int i = get_global_id(0);
	int j = get_global_id(1);

	size_t countX = get_global_size(0);
	size_t countY = get_global_size(1);

	int window_size = 13;
	int window_range = window_size / 2;
	float SSD = 0.0;
	float diff = 0.0;
	float min_SSD = 10000.0;
	float disparity = 0.0;
	int Max_Disparity = 100;
	float temp;

	for (int d = 0; d <= Max_Disparity; d++)
	{
		SSD = 0;
		for (int imageWidth = 1 - window_range; imageWidth < window_range; imageWidth++)
		{
			for (int imageHeight = 1 - window_range; imageHeight < window_range; imageHeight++)
			{
				diff = (getValueImage(d_input_Left, imageWidth + i, imageHeight + j)) - (getValueImage(d_input_Right, imageWidth + i - d, imageHeight + j));;
				SSD = SSD + diff * diff;
			}
		}

		if (SSD < min_SSD)
		{
			min_SSD = SSD;
			disparity = (float)(d) / Max_Disparity;

		}

	}
	d_output[getIndexGlobal(countX, i, j)] = disparity;		// Update output 
}
