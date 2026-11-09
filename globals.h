#pragma once
#include <filesystem>
namespace fs = std::filesystem;

constexpr int numSingleFeatures = 6;
constexpr int numHistograms = 5;
constexpr int numHistogramBins = 32;

constexpr int numFeatures = numSingleFeatures + numHistograms * numHistogramBins;

const fs::path weightsFilePath = "weights.csv";
const fs::path histogramsFilePath = "histograms.csv";
const fs::path rawFeaturesFilePath = "raw_features.csv";
const fs::path normalizationDataFilePath = "normalization_data.csv";
const fs::path standardizationDataFilePath = "standardization_data.csv";
const fs::path fullFeaturesNormFilePath = "full_features_norm.csv";
const fs::path fullFeaturesStandFilePath = "full_features_stand.csv";
const fs::path pathsFilePath = "pathids.csv";
const fs::path evalLinearFilePath = "evaluation.csv";
const fs::path evalLinearPerClassFilePath = "evaluation_per_class.csv";
const fs::path evalANNFilePath = "eval_ANN.csv";
const fs::path evalANNPerClassFilePath = "eval_ANN_per_class.csv";
const fs::path tsneEmbeddingFilePath = "tsne_embedding.csv";
