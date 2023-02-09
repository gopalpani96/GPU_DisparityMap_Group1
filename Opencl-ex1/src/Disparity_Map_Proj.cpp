//////////////////////////////////////////////////////////////////////////////
// GPU Project Group 1: Disparity Map
// File Name : Disparity_Map_Proj.cpp
// Author : Rucha Golwalkar st179996@stud.uni-stuttgart.de , University of Stuttgart
//			Gopal Panigrahi st179549@stud.uni-stuttgart.de , University of Stuttgart
//			Mandar Kharde   st179460@stud.uni-stuttgart.de , University of Stuttgart
//////////////////////////////////////////////////////////////////////////////

// Includes Libraries
#include <stdio.h>
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

#include <boost/lexical_cast.hpp>

//////////////////////////////////////////////////////////////////////////////
// CPU implementation
//////////////////////////////////////////////////////////////////////////////

int getIndexGlobal(std::size_t countX, int i, int j)
{
	return j * countX + i;
}
// Read value from global array a, return 0 if outside image
float getValueGlobal(const std::vector<float>& a, std::size_t countX, std::size_t countY, int i, int j)
{
	if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
		return 0;
	else
		return a[getIndexGlobal(countX, i, j)];
}

// CPU implementation - SAD 

void DisparityMap_SAD(const std::vector<float>& img1, const std::vector<float>& img2, std::vector<float>& h_outputSAD, size_t countX, size_t countY)
{
	int window_size = 13;
	float SAD = 0.0;
	float min_SAD = 10000.0;
	float disparity = 0.0;
	int Max_Disparity = 100;


	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			min_SAD = 10000.0;
			for (int d = 0; d <= Max_Disparity; d++)
			{
				SAD = 0;
				for (int width = 1- (window_size / 2); width < (window_size / 2); width++)
				{
					for (int height = 1- (window_size / 2); height < (window_size / 2); height++)
					{
						SAD = SAD + abs(getValueGlobal(img1, countX, countY, width + i, height + j) - getValueGlobal(img2, countX, countY, width + i - d, height + j));
					}
				}
				if (min_SAD > SAD)
				{
					min_SAD = SAD;
					disparity = (float)(d) / Max_Disparity;
				}
			}
			h_outputSAD[getIndexGlobal(countX, i, j)] = disparity;
		}

	}
}

// CPU implementation - SSD 

void DisparityMap_SSD(const std::vector<float>& img1, const std::vector<float>& img2, std::vector<float>& h_outputSSD, size_t countX, size_t countY)
{
	int window_size = 13;
	float SSD = 0.0;
	float SSDmin = 10000.0;
	float disparity = 0.0;
	int Max_Disparity = 100;
	float t = 0;

	for (int i = 0; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			SSDmin = 10000.0;
			for (int d = 0; d <= Max_Disparity; d++)
			{
				SSD = 0;
				for (int width = 1 - (window_size / 2); width < (window_size / 2); width++)
				{
					for (int height = 1 - (window_size / 2); height < (window_size / 2); height++)
					{
						t = abs(getValueGlobal(img1, countX, countY, width + i, height + j) - getValueGlobal(img2, countX, countY, width + i - d, height + j));
						SSD = SSD + (t * t);
					}
				}
				if (SSDmin > SSD)
				{
					SSDmin = SSD;
					disparity = (float)(d) / Max_Disparity;
				}
			}
			h_outputSSD[getIndexGlobal(countX, i, j)] = disparity;
		}

	}
}

//////////////////////////////////////////////////////////////////////////////
// Main Function
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
	// Create a context	
	//cl::Context context(CL_DEVICE_TYPE_GPU);
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.size() == 0) {
		std::cerr << "No platforms found" << std::endl;
		return 1;
	}
	int platformId = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		if (platforms[i].getInfo<CL_PLATFORM_NAME>() == "AMD Accelerated Parallel Processing") {
			platformId = i;
			break;
		}
	}
	
	cl_context_properties prop[4] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[platformId](), 0, 0 };
	std::cout << "Using platform '" << platforms[platformId].getInfo<CL_PLATFORM_NAME>() << "' from '" << platforms[platformId].getInfo<CL_PLATFORM_VENDOR>() << "'" << std::endl;
	cl::Context context(CL_DEVICE_TYPE_GPU, prop);

	// Get a device of the context
	int deviceNr = argc < 2 ? 1 : atoi(argv[1]);
	std::cout << "Using device " << deviceNr << " / " << context.getInfo<CL_CONTEXT_DEVICES>().size() << std::endl;
	ASSERT(deviceNr > 0);
	ASSERT((size_t)deviceNr <= context.getInfo<CL_CONTEXT_DEVICES>().size());
	cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceNr - 1];
	std::vector<cl::Device> devices;
	devices.push_back(device);
	OpenCL::printDeviceInfo(std::cout, device);

	// Declare some values
	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	std::size_t countX = wgSizeX * 24; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = wgSizeY * 18;

	std::size_t count = countX * countY; // Overall number of elements
	std::size_t size = count * sizeof(float); // Size of data in bytes

	//Allocate space for input and output data from CPU and GPU.
	std::vector<float> h_input_Left(count);
	std::vector<float> h_input_Right(count);
	std::vector<float> h_output_GPU(count);
	std::vector<float> h_output_CPU_SAD(count);
	std::vector<float> h_output_CPU_SSD(count);

	// Allocate space for input and output data on the device
	cl::Buffer d_output(context, CL_MEM_READ_WRITE, (countX * countY) * sizeof(int));

	// Create a command queue
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load the source code
	cl::Program program = OpenCL::loadProgramSource(context, "src/Disparity_Map_Proj.cl");
	// Compile the source code. This is similar to program.build(devices) but will print more detailed error messages
	OpenCL::buildProgram(program, devices);

	// Creation of Image - inputs
	cl::Image2D Left_image(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);
	cl::Image2D Right_image(context, CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), countX, countY);



	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;
	cl::size_t<3> region;
	region[0] = countX;
	region[1] = countY;
	region[2] = 1;

	memset(h_input_Left.data(), 255, size);
	memset(h_input_Right.data(), 255, size);
	memset(h_output_GPU.data(), 255, size);
	memset(h_output_CPU_SAD.data(), 255, size);
	memset(h_output_CPU_SSD.data(), 255, size);

	//////// Load input data ////////////////////////////////

	//	Use an image as input data and store in buffer

	std::vector<float> Left_input;
	std::vector<float>Right_input;
	std::size_t Left_input_width, Left_input_height, Right_input_width, Right_input_height;
	Core::readImagePGM("data_input/Teddy_Left.pgm", Left_input, Left_input_width, Left_input_height);
	Core::readImagePGM("data_input/Teddy_Right.pgm", Right_input, Right_input_width, Right_input_height);

	for (size_t j = 0; j < countY; j++)
	{
		for (size_t i = 0; i < countX; i++)
		{
			h_input_Left[i + countX * j] = Left_input[(i % Left_input_width) + Left_input_width * (j % Left_input_height)];
			h_input_Right[i + countX * j] = Right_input[(i % Right_input_width) + Right_input_width * (j % Right_input_height)];
		}
	}

	// Do calculation on the host side

	// Iterate over all implementations (impl 1 - SAD, impl 2 - SSD)
	for (int impl = 1; impl <= 2; impl++)
	{
		Core::TimeSpan start_time = Core::getCurrentTime();
		Core::TimeSpan CPU_time = start_time;
		if (impl == 1)
		{
			std::cout << "\nImplementation of SAD :" << std::endl;
			std::cout << "********************************************************************" << std::endl;
			std::cout << "                       CPU & GPU Execution for SAD                  " << std::endl;
			std::cout << "********************************************************************" << std::endl;

			Core::TimeSpan start_time = Core::getCurrentTime();

			// Function call SAD -- CPU Implementation
			DisparityMap_SAD(h_input_Left, h_input_Right, h_output_CPU_SAD, countX, countY);
			Core::TimeSpan end_time = Core::getCurrentTime();
			CPU_time = end_time - start_time;

			//////// Store Output Image - CPU with SAD ///////////////////////////////////
			Core::writeImagePGM("data_output/output_disparity_CPU_SAD.pgm", h_output_CPU_SAD, countX, countY);
			std::cout << "\ndata_output/output_disparity_CPU_SAD.pgm Has been Generated" << std::endl;
			std::cout << "\n__________________ CPU Execution for SAD Completed _________________" << std::endl;
		}
		if (impl == 2)
		{
			std::cout << "\nImplementation of SSD :" << std::endl;
			std::cout << "********************************************************************" << std::endl;
			std::cout << "                      CPU & GPU Execution for SSD                   " << std::endl;
			std::cout << "********************************************************************" << std::endl;

			Core::TimeSpan start_time = Core::getCurrentTime();

			// Function call SSD -- CPU Implementation
			DisparityMap_SSD(h_input_Left, h_input_Right, h_output_CPU_SSD, countX, countY);
			Core::TimeSpan end_time = Core::getCurrentTime();
			CPU_time = end_time - start_time;

			//////// Store Output Image - CPU with SSD ///////////////////////////////////
			Core::writeImagePGM("data_output/output_disparity_CPU_SSD.pgm", h_output_CPU_SSD, countX, countY);
			std::cout << "\ndata_output/output_disparity_CPU_SSD.pgm Has been Generated" << std::endl;
			std::cout << "\n__________________ CPU Execution for SSD Completed _________________" << std::endl;
		}

		// Reinitialize output memory to 0xff
		memset(h_output_GPU.data(), 255, size);

		// Copy input data to device
		queue.enqueueWriteBuffer(d_output, true, 0, size, h_output_GPU.data());


		//Enqueue the images to the kernel
		cl::Event copy1;
		queue.enqueueWriteImage(Left_image, true, origin, region, countX * (sizeof(float)), 0, h_input_Left.data(), NULL, &copy1);
		queue.enqueueWriteImage(Right_image, true, origin, region, countX * (sizeof(float)), 0, h_input_Right.data(), NULL, &copy1);

		// Create kernel object
		std::string kernelName = "DisparityMap" + boost::lexical_cast<std::string> (impl);
		cl::Kernel DisparityMap(program, kernelName.c_str());

		// Set Kernel Arguments
		cl::Event execution;
		DisparityMap.setArg<cl::Image2D>(0, Left_image);
		DisparityMap.setArg<cl::Image2D>(1, Right_image);
		DisparityMap.setArg<cl::Buffer>(2, d_output);

		//Launch Kernel on the device
		queue.enqueueNDRangeKernel(DisparityMap, 0, cl::NDRange(countX, countY), cl::NDRange(wgSizeX, wgSizeY), NULL, &execution);


		std::cout << "__________________ Successful execution of Kernel __________________" << std::endl;

		// Copy output data from GPU back to host
		cl::Event copy2;
		queue.enqueueReadBuffer(d_output, true, 0, count * sizeof(int), h_output_GPU.data(), NULL, &copy2);

		// Print performance data
		Core::TimeSpan GPU_time = OpenCL::getElapsedTime(execution);
		Core::TimeSpan copyTime = OpenCL::getElapsedTime(copy1) + OpenCL::getElapsedTime(copy2);
		Core::TimeSpan overallGpuTime = GPU_time + copyTime;
		std::cout << "\nCPU Time: " << CPU_time.toString() << std::endl;;
		std::cout << "Memory copy Time: " << copyTime.toString() << std::endl;
		std::cout << "GPU Time w/o memory copy : " << GPU_time.toString() << " \nSpeedup = " << (CPU_time.getSeconds() / GPU_time.getSeconds()) << std::endl;
		std::cout << "GPU Time with memory copy: " << overallGpuTime.toString() << "\nSpeedup = " << (CPU_time.getSeconds() / overallGpuTime.getSeconds())  << std::endl;
		if (impl == 1)
		{
		std::cout << "__________________ GPU Execution for SAD Completed _________________" << std::endl;
		}
		if (impl == 2)
		{
		std::cout << "__________________ GPU Execution for SSD Completed _________________" << std::endl;
		}

		//////// Store Output Image GPU ///////////////////////////////////

		if(impl == 1)
		{
		Core::writeImagePGM("data_output/output_disparity_GPU_SAD.pgm", h_output_GPU, countX, countY);

		std::cout << "\ndata_output/output_disparity_GPU_SAD.pgm has been Generated" << std::endl;
		}
		if(impl == 2)
		{
		Core::writeImagePGM("data_output/output_disparity_GPU_SSD.pgm", h_output_GPU, countX, countY);

		std::cout << "\ndata_output/output_disparity_GPU_SSD.pgm has been Generated" << std::endl;
		}
		std::cout << "\nSUCCESS" << std::endl;
	}

}
