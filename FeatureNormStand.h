#pragma once
#include <vector>

void normalizeFeatures(std::vector<std::vector<double>> &features)
{
	std::ofstream file("normalization_data.csv");
	file << "Min value;Max value\n";

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

void standardizeFeatures(std::vector<std::vector<double>>& features)
{
	std::ofstream file("standardization_data.csv");
	file << "Mean;Standard Deviation\n";

	for (int col = 0; col < features[0].size(); col++)
	{
		double sum = 0.0;
		int n = features.size();
		for (int row = 0; row < n; row++)
		{
			sum += features[row][col];
		}
		double mean = sum / n;
		double variance = (pow(sum, 2) / n) - (mean * mean);
		double stddev = sqrt(variance);
		for (int row = 0; row < n; row++)
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
			features[i] = (features[i] - data.minValues[i]) / (data.maxValues[i] - data.minValues[i]);
		else
			features[i] = 0.0; // If max and min are the same, set to zero
	}
}

void standardizeWithData(std::vector<double>& features, const StandardizationData& data)
{
	for (size_t i = 0; i < features.size(); ++i)
	{
		if (data.stddevs[i] > 1e-9) // Avoid division by zero
			features[i] = (features[i] - data.means[i]) / data.stddevs[i];
		else
			features[i] = 0.0; // If stddev is zero, set to zero
	}
}

void promptNormOrStandard(std::vector<double>& queryFeatures)
{
	std::cout << "Type 1 to match with normalized features, 2 to match with standardized features: ";
	int choice;
	std::cin >> choice;

	if (choice == 1)
	{
		NormalizationData normData = readNormalizationData("normalization_data.csv");
		normalizeWithData(queryFeatures, normData);
	}
	else if (choice == 2)
	{
		StandardizationData stdData = readStandardizationData("standardization_data.csv");
		standardizeWithData(queryFeatures, stdData);
	}
	else
	{
		std::cout << "Invalid input.\n";
		promptNormOrStandard(queryFeatures);
	}
}

std::vector<std::vector<double>> readFeatures(fs::path filePath, std::vector<std::string> &labels)
{
	labels.clear();
	std::vector<std::vector<double>> features;

	std::ifstream file(filePath);
	
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filePath << std::endl;
		return features;
	}

	std::string line;
	std::getline(file, line); // Skip header line

	while (std::getline(file, line))
	{
		std::stringstream ss(line);
		std::string word;

		std::getline(ss, word, ';'); // Read ID
		std::getline(ss, word, ';'); // Read label
		labels.push_back(word);

		std::vector<double> row;
		while (std::getline(ss, word, ';'))
		{
			if (!word.empty())
				row.push_back(std::stod(word));
		}
		features.push_back(row);
	}

	return features;
}

