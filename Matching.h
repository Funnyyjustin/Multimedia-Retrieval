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

// Distance functions

float euclideanDistance(std::vector<double> feature, std::vector<double> query)
{
	double p = 3;
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
		std::vector<float> elem = { i, euclideanDistance(features[i], query) };
		distances.push_back(elem);
	}

	return distances;
}

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