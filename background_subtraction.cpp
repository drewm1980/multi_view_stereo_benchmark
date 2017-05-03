#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>

using namespace std;

extern "C" {
using namespace std;
void update_histogram_image(const uint8_t* __restrict__ image, int pixels,
                      uint8_t* __restrict__ histogram_image) {
#pragma omp parallel for schedule(static)
    for(int pixel_index = 0; pixel_index<pixels; pixel_index++){
      uint8_t image_value = image[pixel_index];
      int histogram_bin = pixel_index * 256 + image_value; // linear index into the histogram image
      int old_histogram_value = histogram_image[histogram_bin];
      if (old_histogram_value == 255) continue; // Saturate instead of wrapping.
      histogram_image[histogram_bin] = old_histogram_value + 1;
    }
}

// Take a pointer to a 256 bin histogram of uint8 type, and return the median.
uint8_t histogram_to_median(const uint8_t* hist)
{
  // This requires an extra pass through the data, but the next pass will be
  // cached anyway so it probably doesn't make a difference.
  // Provides more gracefull degredation for histograms that may have saturated in
  // some bins.
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

void median_of_histogram_image(const uint8_t * __restrict__ histogram_image, uint8_t * __restrict__ median_image, int pixels)
{
  // Take an image that has a 256 bin histogram at each pixel, and return an
  // image containing the median value at each pixel, rounded down.
#pragma omp parallel for schedule(static)
  for (int pixel_index = 0; pixel_index < pixels; pixel_index++) {
      const uint8_t* histogram = histogram_image + pixel_index * 256;
      median_image[pixel_index] = histogram_to_median(histogram);
  }
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
