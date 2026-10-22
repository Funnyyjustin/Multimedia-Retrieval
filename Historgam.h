#pragma once

#include <string>
#include <vector>
#include <map>

struct CSVData {
    std::vector<std::string> headers;
    std::map<std::string, std::vector<double>> columns;
};

// Structure representing a single shape's row in the CSV
struct ShapeFeatures {
    int shapeId = -1;
    std::string className;
    std::map<std::string, double> elementary;              // "Surface Area", "Compactness", etc.
    std::map<std::string, std::vector<double>> histograms; // "A3" -> 20 bins, "D1" -> 20 bins, etc.
};

// Full dataset parsed from CSV grouped by class
struct FeatureDatabase {
    bool hasClasses = false;
    std::vector<std::string> classNames;
    std::map<std::string, std::vector<ShapeFeatures>> classes;
    std::vector<std::string> elementaryFeatureNames;
    std::vector<std::string> histogramFeatureNames; // "A3", "D1", "D2", "D3", "D4"
    std::vector<ShapeFeatures> allShapes;
};

std::vector<double> computeFixedHistogram(
    const std::vector<double>& values, 
    size_t numBins, 
    double minVal, 
    double maxVal);

// Reads a semicolon-separated CSV file (supports European comma decimals)
CSVData readCSV(const std::string& filepath);

// Parses a feature CSV file into shapes (with or without class labels)
FeatureDatabase parseFeatureCSV(const std::string& filepath);

// Exports a multi-line plot for a specific class and a 20-bin histogram feature (A3, D1, D2, D3, D4)
bool exportClassHistogramLinePlot(
    const std::string& className,
    const std::string& featureName,
    const std::vector<ShapeFeatures>& shapes,
    const std::string& filename,
    double width = 1000,
    double height = 700
);

// Exports a histogram of a dataset to PNG using pbPlots
bool exportHistogram(
    const std::vector<double>& data,
    const std::string& filename,
    const std::string& title = "Histogram",
    size_t numBins = 20,
    bool normalizeCount = true,
    bool normalizeDataValues = false,
    double width = 800,
    double height = 700
);

// Main interactive mode for exporting histograms and feature charts
void runHistogramMode();
