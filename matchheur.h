#pragma once
#include <vector>
#include <string>


void scoreMatching(std::vector<std::string>& paths, int k, int nos)
{
	std::vector<std::vector<std::string>> allmatches;
	int total = 0;

	for (int i; i < paths.size(); i++)
	{
		Parser p;
		Mesh m = p.parseOFF(paths[i]);

		resampleMesh(m);
		normalizeMesh(m);
		precomputeMesh(m);

		auto feature = featureExtract(m);

		if (nos == 0)
		{
			NormalizationData normData = readNormalizationData("normalization_data.csv");
			normalizeWithData(feature, normData);
			FeatureData featData = readFeatures("full_features_norm.csv");

			auto distances = makeDistanceVector(featData.features, feature);
			quicksort(distances, 1, 0, distances.size() - 1);

			for (int j; j < k; j++)
			{
				std::string matchLabel = featData.labels[distances[j][0]];
				allmatches[i].push_back(matchLabel);
			}
				
		}
		else
		{
			StandardizationData stdData = readStandardizationData("standardization_data.csv");
			standardizeWithData(feature, stdData);
			FeatureData featData = readFeatures("full_features_stand.csv");

			auto distances = makeDistanceVector(featData.features, feature);
			quicksort(distances, 1, 0, distances.size() - 1);

			for (int j; j < k; j++)
				allmatches[i].push_back(featData.labels[distances[j][0]]);
		}		
	}
}
