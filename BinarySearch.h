#pragma once
#include <vector>

using namespace std;

inline vector<int> closestElement(vector<vector<int>>& vec, int index, float target)
{
	int low = 0;
	int high = vec.size() - 1;

	while (low < high)
	{
		int mid = low + (high - low) / 2;
		if (vec[mid][index] < target)
			low = mid + 1;
		else
			high = mid;
	}

	if (low == 0) return vec[0];
	if (vec[low][index]) return vec[low];


	if (abs(target - vec[low - 1][index]) <= abs(target - vec[low][index]))
		return vec[low - 1];
	else
		return vec[low];

}
