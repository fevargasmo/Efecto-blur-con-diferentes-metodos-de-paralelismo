#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <opencv2/core/cuda.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
    const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,    
                               blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;    
    if(p.x >= numCols || p.y >= numRows)
         return;
    float color = 0.0f;
    for(int f_y = 0; f_y < filterWidth; f_y++) {
        for(int f_x = 0; f_x < filterWidth; f_x++) {
   
            int c_x = p.x + f_x - filterWidth/2;
            int c_y = p.y + f_y - filterWidth/2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = filter[f_y*filterWidth + f_x];
            color += filter_value*static_cast<float>(inputChannel[c_y*numCols + c_x]);            
        }
    }    
    outputChannel[m] = color;
}

void checkCudaErrors( cudaError_t event )
{
    if( event != cudaSuccess )
    {
        printf("Error code %s \n", cudaGetErrorString(event));
        exit(EXIT_FAILURE);
    }
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  const int2 p = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                            blockIdx.y * blockDim.y + threadIdx.y);
  const int m = p.y * numCols + p.x;    
  if(p.x >= numCols || p.y >= numRows)
      return;
  redChannel[m]   = inputImageRGBA[m].x;
  greenChannel[m] = inputImageRGBA[m].y;
  blueChannel[m]  = inputImageRGBA[m].z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;
  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);
  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_filter, sizeof( float) * filterWidth * filterWidth));
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));
}

void cudaBlur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred, 
                        unsigned char *d_greenBlurred, 
                        unsigned char *d_blueBlurred,
                        const int filterWidth,int numthreads)
{
  separateChannels<<<1, numthreads>>>(d_inputImageRGBA,numRows,numCols,d_red,d_green,d_blue);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<1, numthreads>>>(d_red,d_redBlurred,numRows,numCols,d_filter,filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<1, numthreads>>>(d_blue,d_blueBlurred,numRows,numCols,d_filter,filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  gaussian_blur<<<1, numthreads>>>(d_green,d_greenBlurred,numRows,numCols,d_filter,filterWidth);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  recombineChannels<<<1, numthreads>>>(d_redBlurred,d_greenBlurred,d_blueBlurred,d_outputImageRGBA,numRows,numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
}

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }

void preBlur(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                float **h_filter, int *filterWidth,
                const std::string &filename) {
  checkCudaErrors(cudaFree(0));
  cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }
  cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);
  imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);
  if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }
  *h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
  *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);
  const size_t numPixels = numRows() * numCols();
  checkCudaErrors(cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4))); 
  checkCudaErrors(cudaMemcpy(*d_inputImageRGBA, *h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));
  d_inputImageRGBA__  = *d_inputImageRGBA;
  d_outputImageRGBA__ = *d_outputImageRGBA;
  const int blurKernelWidth = *filterWidth;
  const float blurKernelSigma = 2.;
  *filterWidth = blurKernelWidth;
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  h_filter__ = *h_filter;
  float filterSum = 0.f; 
  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }
  float normalizationFactor = 1.f / filterSum;
  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
    }
  }
  checkCudaErrors(cudaMalloc(d_redBlurred,    sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels));
}

void postProcess(const std::string& output_file, uchar4* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC4, (void*)data_ptr);
  cv::Mat imageOutputBGR;
  cv::cvtColor(output, imageOutputBGR, CV_RGBA2BGR);
  cv::imwrite(output_file.c_str(), imageOutputBGR);
}

void cleanUp(void)
{
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
  delete[] h_filter__;
}

void channelConvolution(const unsigned char* const channel,
                        unsigned char* const channelBlurred,
                        const size_t numRows, const size_t numCols,
                        const float *filter, const int filterWidth)
{
  assert(filterWidth % 2 == 1);
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result = 0.f;
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
		  int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));
          float image_value = static_cast<float>(channel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
          result += image_value * filter_value;
        }
      }
      channelBlurred[r * numCols + c] = result;
    }
  }
}

void referenceCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                          size_t numRows, size_t numCols,
                          const float* const filter, const int filterWidth)
{
  unsigned char *red   = new unsigned char[numRows * numCols];
  unsigned char *blue  = new unsigned char[numRows * numCols];
  unsigned char *green = new unsigned char[numRows * numCols];
  unsigned char *redBlurred   = new unsigned char[numRows * numCols];
  unsigned char *blueBlurred  = new unsigned char[numRows * numCols];
  unsigned char *greenBlurred = new unsigned char[numRows * numCols];
  for (size_t i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = rgbaImage[i];
    red[i]   = rgba.x;
    green[i] = rgba.y;
    blue[i]  = rgba.z;
  }
  channelConvolution(red, redBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(green, greenBlurred, numRows, numCols, filter, filterWidth);
  channelConvolution(blue, blueBlurred, numRows, numCols, filter, filterWidth);
  for (size_t i = 0; i < numRows * numCols; ++i) {
    uchar4 rgba = make_uchar4(redBlurred[i], greenBlurred[i], blueBlurred[i], 255);
    outputImage[i] = rgba;
  }
  delete[] red;
  delete[] green;
  delete[] blue;
  delete[] redBlurred;
  delete[] greenBlurred;
  delete[] blueBlurred;
}

int main(int argc, char **argv) {
  uchar4 *h_inputImageRGBA,  *d_inputImageRGBA;
  uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
  float *h_filter;

  //Test Blur 
  std::string input_file = std::string(argv[1]);//input image 
  std::string output_file = std::string(argv[2]);
  int filterWidth = atoi(argv[3]);//size of kernel
  int numthreads = atoi(argv[4]);//number of threads

  //Test Time
   /* 
  std::string input_file = std::string(argv[1]);//input image 
  int filterWidth = atoi(argv[2]);//size of kernel
  int numthreads = atoi(argv[3]);//number of threads
  */

  preBlur(&h_inputImageRGBA, &h_outputImageRGBA, &d_inputImageRGBA, &d_outputImageRGBA,
             &d_redBlurred, &d_greenBlurred, &d_blueBlurred,&h_filter, &filterWidth, input_file);

  allocateMemoryAndCopyToGPU(numRows(), numCols(), h_filter, filterWidth);

  cudaBlur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA, numRows(), numCols(),
                     d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth,numthreads);

  cudaDeviceSynchronize(); 
  checkCudaErrors(cudaGetLastError());

  size_t numPixels = numRows()*numCols();
  
  checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA__, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost));
  
  referenceCalculation(h_inputImageRGBA, h_outputImageRGBA,numRows(), numCols(),h_filter, filterWidth);
  postProcess(output_file, h_outputImageRGBA);
  
  checkCudaErrors(cudaFree(d_redBlurred));
  checkCudaErrors(cudaFree(d_greenBlurred));
  checkCudaErrors(cudaFree(d_blueBlurred));
  
  cleanup();
  cleanUp();
  return 0;
}
