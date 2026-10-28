#pragma once
#include <vector>
#include <string>


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
	FeatureData featDataStand = readFeatures("full_features_stand.csv");

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

	std::cout << "Calculating score with standardization";
	total = 0;
	points = 0;
	for (int i = 0; i < paths.size(); i++)
	{
		std::cout << ".";
		Parser p;
		Mesh m = p.parseOFF(paths[i]);

		resample(m, 5000, 10000);
		normalize(m);
		precompute(m);

		auto feature = featureExtract(m);

		standardizeWithData(feature, stdData);


		auto distances = makeDistanceVector(featDataStand.features, feature);
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
