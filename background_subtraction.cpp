#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>

using namespace std;

extern "C" {
using namespace std;
void update_histogram(const uint8_t* __restrict__ image, int h, int w,
                      uint8_t* __restrict__ histogram) {
#pragma omp parallel for schedule(static)
    for (int v = 0; v < h; v++) {
        for (int u = 0; u < w; u++) {
            int image_index = v * w + u;
            uint8_t image_value = image[image_index];
            int histogram_bin = image_index * 256 + image_value;
            if (histogram[histogram_bin] < 255) histogram[histogram_bin] += 1;
        }
    }
}

void median_of_histogram_image(const uint8_t * __restrict__ histogram, uint8_t * __restrict__ median, int pixels, uint8_t samples)
{
  // Take an image that has a 256 bin histogram at each pixel, and return an
  // image containing the median value at each pixel, rounded down.
  //
  // STATUS: THIS HAS BUGS!!! REFER TO THE PYTHON IMPLEMENTATION FOR NOW!
#pragma omp parallel for schedule(static)
    for(int pixel_index = 0; pixel_index<pixels; pixel_index++)
  {
    const uint8_t * hist = histogram + pixel_index*256;

    int sum = samples;
    int sum_left = 0;
    int sum_right = sum;
    
    // Explicitly handle edge cases
    if(hist[0]*2>sum) median[pixel_index] = 0;
    if(hist[255]*2>sum) median[pixel_index] = 255;

    // Handle the middle cases
    sum_right -= hist[0];
    int current_plateau_width = 0;
    int last_difference = sum;
    for(int i=1; i<256; i++)
    {
      sum_left += hist[i-1];
      sum_right -= hist[i];
      if (hist[i]==0)
      {
        current_plateau_width += 1;
        continue;
      }
      int difference = abs(sum_right - sum_left);
      if (difference > last_difference)
      {
        median[pixel_index] = i-current_plateau_width/2; // Implements averaging
      }
      current_plateau_width = 0;
    }
  }
}

// Take a pointer to a 256 bin histogram of uint8 type, and return the median.
uint8_t histogram_to_median(const uint8_t* hist)
{
  // This requires an extra pass through the data, but the next pass will be
  // cached anyway so it probably doesn't make a difference.
  int samples=0; // The number of samples.
  for (int i = 0; i < 256; i++) samples += hist[i];
    
   // Explicitly handle edge cases
  if (hist[0] * 2 > samples) return 0;
  if (hist[255] * 2 > samples) return 255;

  // Handle the middle cases
  int sum_left = 0;
  int sum_right = samples - hist[0];
  int lowest_difference = sum_right;
  unsigned int plateau_start = 0;
  unsigned int plateau_end = 0;
  for (int i = 0; i < 256; i++) {
      int difference = abs(sum_right - sum_left);
      if (difference < lowest_difference) {
          lowest_difference = difference;
          plateau_start = i;
          plateau_end = i;
      }
      if (difference == lowest_difference) {
          plateau_end = i;
      }
      if (difference > lowest_difference) {
          plateau_end = i - 1;
          break;
      }
      if (i < 255) {
          sum_left += hist[i];
          sum_right -= hist[i + 1];
      }
  }

  // Note: the following implements averaging. Note: should get compiled to a
  // shift. Rounds down implicitly.
  uint8_t median = (plateau_end + plateau_start)/2;  
  return median;

}

void absdiff_uint8(const uint8_t* __restrict__ left,
                   const uint8_t* __restrict__ right,
                   uint8_t* __restrict__ absdiff, int num_values) {
  // Compute the absolute difference between two images
#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_values; i++) {
        auto l = left[i];
        auto r = right[i];
        absdiff[i] = r > l ? r - l : l - r;
    }
}

}
