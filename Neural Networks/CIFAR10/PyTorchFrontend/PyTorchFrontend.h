#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#ifndef NET_CONSTANTS

// Precalculated means and stds for each channel (RGB) used for input normalizations
static std::vector<float> const CHANNEL_MEANS = { 0.4914, 0.4822, 0.4465 };
static std::vector<float> const CHANNEL_STDS = { 0.2023, 0.1994, 0.2010 };

// Index references to avoid confusion
enum PIL_Channel { red, green, blue };
enum CV_Channel { blue, green, red };

#endif // !NET_CONSTANTS


/*
	This header file provides the layout/structure of the PyTorch CNN developed for
	motion detection (identifying and tracking vessels on the water).

	Import this header to implement the network.


	Installation
	------------
	Be sure to download the LibTorch distro at
	https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip

	On your project settings...
	1. Include
		- 'libtorch/include'
		- 'libtorch/include/torch/scrc/api/include'
	2. Add 'libtorch/lib' to libraries
*/

class Net {
private:
	cv::Mat image;								// Image storage
	std::vector<torch::jit::IValue> input;		// Input loader for the network
	torch::jit::script::Module net;				// The network loaded from TorchScript serialized file
	at::Tensor output;							// Network output
	at::Tensor percentage;						// Outputs after softmax activation
	at::Tensor tensorImage;						// Image converted to tensor
	int threshold = 60;							// Threshold value to determine likelihood of a ship

	void transformData() {
		// Resize image
		cv::resize(image, image, cv::Size(32, 32));

		// Convert all pixel values to 0-1
		image *= 1.0 / 255;

		// Normalize all pixels and reformat from BGR to RGB
		image.forEach<cv::Vec3f>(
			[](cv::Vec3f &pixel, const int * position) -> void {

			// Normalizations -> (pixel - mean) / std
			pixel[CV_Channel::red] = (
				(pixel[CV_Channel::red] - CHANNEL_MEANS[PIL_Channel::red]) / CHANNEL_STDS[PIL_Channel::red]
				);
			pixel[CV_Channel::green] = (
				(pixel[CV_Channel::green] - CHANNEL_MEANS[PIL_Channel::green]) / CHANNEL_STDS[PIL_Channel::green]
				);
			pixel[CV_Channel::blue] = (
				(pixel[CV_Channel::blue] - CHANNEL_MEANS[PIL_Channel::blue]) / CHANNEL_STDS[PIL_Channel::blue]
				);

			// OpenCV uses BGR, PIL uses RGB. Network was trained on RGB, so we swap red and blue channels.
			// This appears easier to do while we are looking at each pixel rather than after converting to a tensor.
			float inter = pixel[CV_Channel::red];
			pixel[CV_Channel::red] = pixel[CV_Channel::blue];
			pixel[CV_Channel::blue] = inter;
			}
		);
	}

public:
	// Initialize the network from a serialized file
	Net(const char* path) {
		try {
			// Deserialize the TorchScript network from file
			net = torch::jit::load(path);
		}
		catch (const c10::Error& e) {
			std::cerr << "Error loading the network";
		}
	}

	bool isShip(cv::Mat image){
		//Load image to object memory
		this->image = image;

		// Normalize and transform image
		transformData();

		// Convert image to tensor
		tensorImage = torch::from_blob(this->image.data, { 1, 3, this->image.rows, this->image.cols }, at::kByte);
		tensorImage = tensorImage.to(at::kFloat);

		// Write tensor into input loader
		input.emplace_back(tensorImage);

		// Feed forward and read outputs
		output = net.forward(input).toTensor();
		percentage = torch::nn::functional::softmax(output, torch::nn::functional::SoftmaxFuncOptions(1));

		// Check output node 9 (ship) against the threshold value
		return (int)(percentage[8].item<float> * 100) >= threshold;
	}

	// Adjust threshold value
	void setThreshold(int threshold) {
		this->threshold = threshold;
	}
};