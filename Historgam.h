#pragma once

#include <string>
#include <vector>
#include <map>

struct CSVData {
    std::vector<std::string> headers;
    std::map<std::string, std::vector<double>> columns;
};

// Reads a semicolon-separated CSV file (supports European comma decimals)
CSVData readCSV(const std::string& filepath);

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

// Interactive console mode for generating histograms from CSV
void runHistogramMode();
