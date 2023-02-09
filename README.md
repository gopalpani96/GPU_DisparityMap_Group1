# GPU_DisparityMap_Group1
# Disparity Map Estimation using SSD and SAD Algorithm
Instructions to run this code on Linux Environment

# Environment Configuration:
1. Connect to the CISCO Secure client VPN for the GPU access on Linux at CIS
2. Download the GPU_DisparityMap_Group1 project from the github and transfer the file using WinSCP to deimos.cis.iti.uni-stuttgart.de

# Building and Executing the project on the Server:
1. Open Powershell or Command prompt in your local machine to build and execute the following commands
2. Connect to ssh username@phobos.cis.iti.uni-stuttgart.de
3. Connect to ssh phoebe
4. Enter the password provided by CIS team in order to access the server
5. Unzip the above downloaded project from GitHub
6. Change your directy to Opencl-ex1 using- `cd Opencl-ex1`
7. Make a new directory build using- `mkdir build`
8. Go to directory build- `cd build`
9. Execute the Cmake command- `cmake ..`
10. Build the project using- `make`
11. Exit the build folder using- `cd ..`
12. Execute the project using this command- `build/Opencl-ex1`
13. Output screen will be displayed in your powershell window with all the details provided in the report

# Executing different input samples:
1. Different .pgm images are stored in the data_input folder
2. To run different files set we need both right and left images. Change the image name in the Disparity_Map_Proj.cpp source file
3. Rebuild and execute the project
4. Output .pgm files will be stored in data_output folder for CPU and GPU separately for different algorithms

# Changes in the code:
After every modification in the "CMakeLists.txt" and "Disparity_Map_Proj.cl" it is recomended to do cmake, make, build and execute again.



