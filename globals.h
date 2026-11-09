#pragma once
#include <filesystem>
namespace fs = std::filesystem;

constexpr int numSingleFeatures = 6;
constexpr int numHistograms = 5;
constexpr int numHistogramBins = 32;

constexpr int numFeatures = numSingleFeatures + numHistograms * numHistogramBins;

const fs::path weightsFilePath = "weights.csv";
const fs::path histogramsFilePath = "histograms.csv";
const fs::path normalizationDataFilePath = "normalization_data.csv";
const fs::path standardizationDataFilePath = "standardization_data.csv";
const fs::path fullFeaturesNormFilePath = "full_features_norm.csv";
const fs::path fullFeaturesStandFilePath = "full_features_stand.csv";
const fs::path pathsFilePath = "pathids.csv";
