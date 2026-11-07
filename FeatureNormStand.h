#pragma once
#include <vector>
#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "globals.h"

const int numSingleFeatures = 6;
const int numHistograms = 5;
const int numHistogramBins = 20;
const int numFeatures = numSingleFeatures + numHistograms * numHistogramBins;
constexpr size_t NUM_SINGLE_GROUPS = 1;
constexpr size_t NUM_GROUPS = NUM_SINGLE_GROUPS + numHistograms;   // 1 + 5 = 6
constexpr double SINGLE_MASS_SHIFT = 3.0;
using FeatureWeights = std::array<double, NUM_GROUPS>;

void normalizeFeatures(std::vector<std::vector<double>> &features)
{
	std::ofstream file(normalizationDataFilePath);
	file << "Min value;Max value\n";

	int fSize = features.size();

	for (int col = 0; col < features[0].size(); col++)
	{
		double maxVal = -INFINITY;
		double minVal = INFINITY;

		for (int row = 0; row < features.size(); row++)
		{
			if (features[row][col] > maxVal)
				maxVal = features[row][col];
			if (features[row][col] < minVal)
				minVal = features[row][col];

		}
		//std::cout << "MinValue: " << minVal << " MaxValue: " << maxVal << std::endl;

		double diff = maxVal - minVal;
		for (int row = 0; row < features.size(); row++)
		{
			if (diff > 1e-9)
				features[row][col] = (features[row][col] - minVal) / (maxVal - minVal);
			else
				features[row][col] = 0.0;
		}

		file << minVal << ";" << maxVal << "\n";
	}

	file.close();
}

void standardizeSingleFeatures(std::vector<std::vector<double>>& features)
{
	std::ofstream file(standardizationDataFilePath);
	file << "Mean;Standard Deviation\n";

	int fSize = features.size();

	for (int col = 0; col < numSingleFeatures; col++)
	{
		double sum = 0.0;
		for (int row = 0; row < fSize; row++)
		{
			sum += features[row][col];
		}
		double mean = sum / fSize;

		double sqrtSum = 0.0;
		for (int row = 0; row < fSize; row++)
		{
			sqrtSum += pow(features[row][col] - mean, 2);
		}
		double stddev = sqrt(sqrtSum / fSize);

		for (int row = 0; row < fSize; row++)
		{
			if (stddev > 1e-9) // Avoid division by zero
				features[row][col] = (features[row][col] - mean) / stddev;
			else
				features[row][col] = 0.0; // If stddev is zero, set to zero
		}
		file << mean << ";" << stddev << "\n";
	}

	file.close();
}

void normalizeHistogram(std::vector<double>& row)
{
	if (row.size() < numFeatures) return;
	for (int i = 0; i < numHistograms; i++)
	{
		int startCol = numSingleFeatures + i * numHistogramBins;
		int endCol = startCol + numHistogramBins;

		double sum = 0.0;
		for (int col = startCol; col < endCol; col++)
		{
			sum += row[col];
		}
		if (sum > 1e-9) // Avoid division by zero
		{
			for (int col = startCol; col < endCol; col++)
			{
				row[col] /= sum;
			}
		}
		else
		{
			for (int col = startCol; col < endCol; col++)
			{
				row[col] = 0.0; // If sum is zero, set all to zero
			}
		}
	}
}

// written by copilot, need to check if it works correctly
void normalizeHistograms(std::vector<std::vector<double>>& features)
{
	for (auto& row : features)
	{
		normalizeHistogram(row);
	}
}

void standardizeFeatures(std::vector<std::vector<double>>& features)
{
	standardizeSingleFeatures(features);
	normalizeHistograms(features); // Normalizing histograms is equivalent to standardizing them
}

void promptNormOrStandard(std::vector<std::vector<double>>& features)
{
	std::cout << "Type 1 to normalize features, 2 to standardize features: ";

	int nors;
	std::cin >> nors;

	if (nors == 1)
	{
		normalizeFeatures(features);
	}
	else if (nors == 2)
	{
		standardizeFeatures(features);
	}
	else
	{
		std::cout << "Invalid input. No normalization or standardization applied.\n";
	}
}

struct NormalizationData
{
	std::vector<double> minValues;
	std::vector<double> maxValues;
};

struct StandardizationData
{
	std::vector<double> means;
	std::vector<double> stddevs;
};

StandardizationData readStandardizationData(const std::string& filename)
{
	StandardizationData data;
	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		return data;
	}
	std::string line;
	std::getline(file, line); // Skip header line
	while (std::getline(file, line))
	{
		std::stringstream ss(line);
		std::string word;
		std::getline(ss, word, ';'); // Read mean
		data.means.push_back(std::stod(word));
		std::getline(ss, word, ';'); // Read stddev
		data.stddevs.push_back(std::stod(word));
	}
	file.close();
	return data;
}

NormalizationData readNormalizationData(const std::string& filename)
{
	NormalizationData data;
	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		return data;
	}
	std::string line;
	std::getline(file, line); // Skip header line
	while (std::getline(file, line))
	{
		std::stringstream ss(line);
		std::string word;
		std::getline(ss, word, ';'); // Read min value
		data.minValues.push_back(std::stod(word));
		std::getline(ss, word, ';'); // Read max value
		data.maxValues.push_back(std::stod(word));
	}
	file.close();
	return data;
}

void normalizeWithData(std::vector<double>& features, const NormalizationData& data)
{
	for (size_t i = 0; i < features.size(); ++i)
	{
		if (data.maxValues[i] - data.minValues[i] > 1e-9) // Avoid division by zero
		{
			//std::cout << "MinValue: " << data.minValues[i] << " MaxValue: " << data.maxValues[i] << std::endl;
			features[i] = (features[i] - data.minValues[i]) / (data.maxValues[i] - data.minValues[i]);
		}
		else
		{
			//std::cout << "MinValue: " << data.minValues[i] << " MaxValue: " << data.maxValues[i] << std::endl;
			features[i] = 0.0; // If max and min are the same, set to zero
		}
			
	}
}

void standardizeWithData(std::vector<double>& features, const StandardizationData& data)
{
	for (size_t i = 0; i < numSingleFeatures; ++i)
	{
		if (data.stddevs[i] > 1e-9) // Avoid division by zero
			features[i] = (features[i] - data.means[i]) / data.stddevs[i];
		else
			features[i] = 0.0; // If stddev is zero, set to zero
	}
}




