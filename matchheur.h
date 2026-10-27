#pragma once
#include <vector>
#include <string>

/*
void scoreMatching(std::vector<std::string>& paths, int k, int nos)
{
	std::vector<std::vector<std::string>> matches;

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
			FeatureData 
		}

	}
}
*/