#pragma once
#include <vector>
#include <string>

void saveWeights(const FeatureWeights& weights)
{
	std::ofstream file("weights.csv");
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << "weights.csv" << " for writing.\n";
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
	std::ifstream file("weights.csv");
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << "weights.csv" << " for reading.\n";
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

void scoreMatching(
	std::vector<std::string>& paths, 
	std::function<void (Mesh&, int, int)> resample, 
	std::function<void(Mesh&)> normalize, 
	std::function<void(Mesh&)> precompute,
	int k)
{
	//std::vector<std::vector<std::string>> allmatches;
	

	NormalizationData normData = readNormalizationData("normalization_data.csv");
	FeatureData featDataNorm = readFeatures("full_features_norm.csv");

	StandardizationData stdData = readStandardizationData("standardization_data.csv");
	FeatureData featDataStand = readFeatures("full_feats_stand.csv");

	//FeatureWeights weights = computeWeightsFromDatabase(featDataStand.features);
	//saveWeights(weights);
	FeatureWeights weights = readWeights(); // Read the saved weights
	for (size_t g = 0; g < NUM_GROUPS; g++) std::cout << "w[" << g << "]=" << weights[g] << "\n";

	/*
	std::cout << "Calculating score with normalisation";
	double total = 0;
	double points = 0;
	for (int i = 0; i < paths.size(); i++)
	{
		std::cout << ".";
		Parser p;
		Mesh m = p.parseOFF(paths[i]);

		resample(m, 5000, 10000);
		normalize(m);
		precompute(m);

		auto feature = featureExtract(m);

		normalizeWithData(feature, normData);

		auto distances = makeDistanceVector(featDataNorm.features, feature);
		quicksort(distances, 1, 0, distances.size() - 1);

		for (int j = 0; j < k; j++)
		{
			total++;
			std::string matchLabel = featDataNorm.labels[distances[j][0]];
			if (matchLabel == featDataNorm.labels[i])
				points++;
		}	
	}
	std::cout << "\nScore for matching with normalisation (0-100): " << points / total * 100.0 << std::endl;
	*/

	std::cout << "Calculating score with standardization";
	double total = 0;
	double points = 0;
	for (int i = 0; i < paths.size(); i++)
	{
		std::cout << ".";
		Parser p;
		Mesh m = p.parseOFF(paths[i]);

		resample(m, 5000, 10000);
		precompute(m);
		normalize(m);

		auto feature = featureExtract(m);

		standardizeWithData(feature, stdData);
		normalizeHistogram(feature);

		auto distances = makeDistanceVector(featDataStand.features, feature, weights);
		//auto distances = makeDistanceVector(featDataStand.features, feature);
		quicksort(distances, 1, 0, distances.size() - 1);

		for (int j = 0; j < k; j++)
		{
			total++;
			std::string matchLabel = featDataStand.labels[distances[j][0]];
			if (matchLabel == featDataStand.labels[i])
				points++;
		}
	}
	std::cout << "\nScore for matching with standardisation (0-100): " << points / total * 100.0 << std::endl;
}




