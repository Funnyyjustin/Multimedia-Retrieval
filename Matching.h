#pragma once
#include <ANN/ANN.h>

fs::path promptPathOrSelect()
{
	std::cout << "Type 1 to select a file, 2 to enter a path: ";
	int choice;
	std::cin >> choice;
	if (choice == 1)
	{
		return selectModelFile("ObjectFiles");
	}
	else if (choice == 2)
	{
		std::cout << "Enter the path: ";
		std::string path;
		std::cin >> path;
		return fs::path(path);
	}
	else
	{
		std::cout << "Invalid choice. Please try again.\n";
		return promptPathOrSelect();
	}
}


struct FeatureData
{
	std::vector<std::string> labels;
	std::vector<std::vector<double>> features;
};

FeatureData readFeatures(fs::path filePath)
{
	std::vector<std::vector<double>> features;
	std::vector<std::string> labels;

	std::ifstream file(filePath);

	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filePath << std::endl;
		return {};
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


	return FeatureData{ labels, features };
}

inline double histogramEMD(const std::vector<double>& a, const std::vector<double>& b,
	int start, int bins)
{
	double cumA = 0.0, cumB = 0.0, work = 0.0;
	for (int i = start; i < start + bins; i++)
	{
		cumA += a[i];
		cumB += b[i];
		work += std::fabs(cumA - cumB);   // mass crossing the boundary after bin i
	}
	return work;
}

float distanceLp(std::vector<double> feature, std::vector<double> query)
{
	double p = 2;
	double sum = 0;
	for (int i = 0; i < feature.size(); i++)
	{
		sum += pow(fabs(feature[i] - query[i]), p);
	}

	return pow(sum, 1.0 / p);
}

std::array<double, NUM_GROUPS> groupDistances(
	const std::vector<double>& a, const std::vector<double>& b)
{
	std::array<double, NUM_GROUPS> d{};
	d[0] = distanceLp(a, b);                              // group 0: all single features
	int minLen = static_cast<int>(std::min(a.size(), b.size()));
	int histBins = (minLen > numSingleFeatures) ? (minLen - numSingleFeatures) / numHistograms : numHistogramBins;
	for (int h = 0; h < numHistograms; h++)
	{
		int start = numSingleFeatures + h * histBins;
		d[NUM_SINGLE_GROUPS + h] = histogramEMD(a, b, start, histBins);
	}
	return d;
}

inline void normalizeWeights(FeatureWeights& w)
{
	double sum = 0.0;
	for (double x : w) sum += x;
	if (sum > 1e-9) for (double& x : w) x /= sum;
}

inline double weightedDistance(const std::vector<double>& a,
	const std::vector<double>& b,
	const FeatureWeights& w)
{
	double p = 2.0;
	auto d = groupDistances(a, b);
	double total = 0.0;
	for (size_t g = 0; g < NUM_SINGLE_GROUPS; g++)
		total += w[g] * pow(d[g], p);
	total = pow(total, 1.0 / p);

	for (size_t g = NUM_SINGLE_GROUPS; g < NUM_GROUPS; g++)
		total += w[g] * d[g];
	return total;
}

void saveWeights(const FeatureWeights& weights)
{
	std::ofstream file(weightsFilePath);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << weightsFilePath.string() << " for writing.\n";
		return;
	}
	for (size_t g = 0; g < NUM_GROUPS; g++)
	{
		file << weights[g];
		if (g < NUM_GROUPS - 1)
			file << ";";
	}
	file << "\n";
	file.close();
}

std::array<double, NUM_GROUPS> readWeights()
{
	std::array<double, NUM_GROUPS> weights;
	std::ifstream file(weightsFilePath);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << weightsFilePath.string() << " for reading.\n";
	}
	for (size_t g = 0; g < NUM_GROUPS; g++)
	{
		file >> weights[g];
		if (g < NUM_GROUPS - 1)
			file.ignore();
	}
	file.close();
	return weights;
}

// Weight each group inversely to its mean distance over the database
inline FeatureWeights computeWeightsFromDatabase(
	const std::vector<std::vector<double>>& features)
{
	std::array<double, NUM_GROUPS> meanDist{};
	size_t total = 0;
	for (size_t i = 0; i < features.size(); i++)
		for (size_t j = i + 1; j < features.size(); j++)
		{
			auto d = groupDistances(features[i], features[j]);
			for (size_t g = 0; g < NUM_GROUPS; g++) meanDist[g] += d[g];
			total++;
		}

	FeatureWeights w{};
	for (size_t g = 0; g < NUM_GROUPS; g++)
	{
		double m = total ? meanDist[g] / total : 0.0;
		w[g] = (m > 1e-9) ? 1.0 / m : 0.0;
	}
	normalizeWeights(w);
	return w;
}


// Distance functions

std::vector<std::vector<float>> makeDistanceVector(
	const std::vector<std::vector<double>>& features,
	const std::vector<double>& query,
	const FeatureWeights& w)
{
	std::vector<std::vector<float>> distances;
	distances.reserve(features.size());

	for (size_t i = 0; i < features.size(); i++)
		distances.push_back({ (float)i, (float)weightedDistance(features[i], query, w) });

	return distances;
}

// Transform a feature-vector to weighted Hellinger space for ANN
inline std::vector<double> embedFeatureForANN(
	const std::vector<double>& v,
	const FeatureWeights& w)
{
	std::vector<double> emb(v.size(), 0.0);
	// reduze extreme z-outliers and scale with w[0]
	for (int i = 0; i < numSingleFeatures && i < static_cast<int>(v.size()); i++)
	{
		double clamped = std::clamp(v[i], -2.0, 4.0);
		emb[i] = (w[0] / 4.0) * clamped;
	}
	
	int histBins = (static_cast<int>(v.size()) > numSingleFeatures)
		? (static_cast<int>(v.size()) - numSingleFeatures) / numHistograms
		: numHistogramBins;

	// group 1 till 5 Hellinger embedding (sqrt per histogram bin times weight)
	for (int h = 0; h < numHistograms; h++)
	{
		int start = numSingleFeatures + h * histBins;
		double weight = w[NUM_SINGLE_GROUPS + h];
		for (int b = 0; b < histBins; b++)
		{
			int idx = start + b;
			if (idx < static_cast<int>(v.size()))
			{
				emb[idx] = weight * std::sqrt(std::max(v[idx], 0.0));
			}
		}
	}
	
	return emb;
}


// Redundant
FeatureData promptNormOrStandFeatures(std::vector<double>& queryFeatures)
{
	std::cout << "Type 1 to match with normalized features, 2 to match with standardized features: ";
	int choice;
	std::cin >> choice;

	if (choice == 1)
	{
		NormalizationData normData = readNormalizationData(normalizationDataFilePath.string());
		normalizeWithData(queryFeatures, normData);
		return readFeatures(fullFeaturesNormFilePath);
	}
	else if (choice == 2)
	{
		StandardizationData stdData = readStandardizationData(standardizationDataFilePath.string());
		standardizeWithData(queryFeatures, stdData);
		normalizeHistogram(queryFeatures);
		return readFeatures(fullFeaturesStandFilePath);
	}
	else
	{
		std::cout << "Invalid input.\n";
		return promptNormOrStandFeatures(queryFeatures);
	}
}

std::vector<std::vector<float>> makeDistanceVector(const std::vector<std::vector<double>>& features, std::vector<double> query)
{
	std::vector<std::vector<float>> distances;
	for (float i = 0; i < features.size(); i++)
	{
		std::vector<float> elem = { i, distanceLp(features[i], query) };
		distances.push_back(elem);
	}

	return distances;
}
