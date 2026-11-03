#pragma once

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

// Non-negative masses from the standardized single features, normalized to sum 1
inline std::array<double, numSingleFeatures> singleMasses(const std::vector<double>& v)
{
	std::array<double, numSingleFeatures> m{};
	double total = 0.0;
	for (int i = 0; i < numSingleFeatures; i++)
	{
		m[i] = std::max(v[i] + SINGLE_MASS_SHIFT, 0.0);
		total += m[i];
	}
	if (total > 1e-9)
		for (double& x : m) x /= total;
	return m;
}

// EMD over all single-value features together (1D ground distance |i-j|, equal total mass)
inline double singleFeaturesEMD(const std::vector<double>& a, const std::vector<double>& b)
{
	auto ma = singleMasses(a);
	auto mb = singleMasses(b);
	double cumA = 0.0, cumB = 0.0, work = 0.0;
	for (int i = 0; i < numSingleFeatures; i++)
	{
		cumA += ma[i];
		cumB += mb[i];
		work += std::fabs(cumA - cumB);
	}
	return work;
}

// Histogram matching (slide): d(H1,H2) = sqrt( sum_i |H1[i]-H2[i]|^2 )
inline double histogramDistance(const std::vector<double>& a, const std::vector<double>& b,
	int start, int bins)
{
	double sq = 0.0;
	for (int i = start; i < start + bins; i++)
	{
		double diff = a[i] - b[i];
		sq += diff * diff;
	}
	return std::sqrt(sq);
}

inline std::array<double, NUM_GROUPS> groupDistances(
	const std::vector<double>& a, const std::vector<double>& b)
{
	std::array<double, NUM_GROUPS> d{};
	d[0] = singleFeaturesEMD(a, b);                              // group 0: all single features
	for (int h = 0; h < numHistograms; h++)
	{
		int start = numSingleFeatures + h * numHistogramBins;
		d[NUM_SINGLE_GROUPS + h] = histogramDistance(a, b, start, numHistogramBins);
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
	auto d = groupDistances(a, b);
	double p = 1;
	double total = 0.0;
	for (size_t g = 0; g < numSingleFeatures; g++)
		total += w[g] * pow(d[g], p);
	total = pow(total, 1.0 / p);

	for (size_t g = numSingleFeatures; g < NUM_GROUPS; g++)
		total += w[g] * d[g];
	return total;
}

// Weight each group inversely to its mean distance over the database
inline FeatureWeights computeWeightsFromDatabase(
	const std::vector<std::vector<double>>& features)
{
	std::array<double, NUM_GROUPS> meanDist{};
	size_t pairs = 0;
	for (size_t i = 0; i < features.size(); i++)
		for (size_t j = i + 1; j < features.size(); j++)
		{
			auto d = groupDistances(features[i], features[j]);
			for (size_t g = 0; g < NUM_GROUPS; g++) meanDist[g] += d[g];
			pairs++;
		}

	FeatureWeights w{};
	for (size_t g = 0; g < NUM_GROUPS; g++)
	{
		double m = pairs ? meanDist[g] / pairs : 0.0;
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



// Redundant
float DistanceLp(std::vector<double> feature, std::vector<double> query)
{
	double p = 2;
	double sum = 0;
	for (int i = 0; i < feature.size(); i++)
	{
		sum += pow(fabs(feature[i] - query[i]), p);
	}

	return pow(sum, 1.0 / p);
}

std::vector<std::vector<float>> makeDistanceVector(const std::vector<std::vector<double>>& features, std::vector<double> query)
{
	std::vector<std::vector<float>> distances;
	for (float i = 0; i < features.size(); i++)
	{
		std::vector<float> elem = { i, DistanceLp(features[i], query) };
		distances.push_back(elem);
	}

	return distances;
}
