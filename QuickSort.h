#pragma once
#include <vector>



inline void swap(std::vector<std::vector<float>>& vec, int a, int b)
{
	auto temp = vec[a];

	vec[a] = vec[b];
	vec[b] = temp;
}

// quicksort adapted from https://www.geeksforgeeks.org/cpp/cpp-program-for-quicksort/
inline int partition(std::vector<std::vector<float>>& vec, int index, int low, int high)
{
	int pivot = vec[high][index];

	int i = low - 1;

	for (int j = low; j < high; j++)
	{
		if (vec[j][index] <= pivot)
		{
			i++;
			swap(vec[i], vec[j]);
		}
	}

	swap(vec[i + 1], vec[high]);

	return i + 1;
}

inline void quicksort(std::vector<std::vector<float>>& vec, int index, int low, int high) 
{
	if (low < high)
	{
		int pi = partition(vec, index, low, high);

		quicksort(vec, index, low, pi - 1);
		quicksort(vec, index, pi + 1, high);
	}
}