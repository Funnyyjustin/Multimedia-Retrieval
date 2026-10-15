#include "Historgam.h"
#include "pbPlot/pbPlots.hpp"
#include "pbPlot/supportLib.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <cmath>

static std::string formatBound(double val, double binWidth) {
    double absVal = std::abs(val);
    if (absVal < 1e-9) {
        return "0";
    }
    
    // Very large or very small numbers: scientific notation
    if (absVal >= 100000.0 || (absVal < 0.001 && absVal > 0.0)) {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(1) << val;
        return ss.str();
    }
    
    // If bin width is at least 1, display as clean integers
    if (binWidth >= 1.0) {
        return std::to_string(static_cast<long long>(std::round(val)));
    }
    
    // Floating point values with dynamic precision based on binWidth
    std::ostringstream ss;
    if (binWidth >= 0.1) {
        ss << std::fixed << std::setprecision(1) << val;
    } else if (binWidth >= 0.01) {
        ss << std::fixed << std::setprecision(2) << val;
    } else {
        ss << std::fixed << std::setprecision(3) << val;
    }
    
    std::string str = ss.str();
    if (str.find('.') != std::string::npos) {
        str.erase(str.find_last_not_of('0') + 1);
        if (!str.empty() && str.back() == '.') {
            str.pop_back();
        }
    }
    return str;
}

CSVData readCSV(const std::string& filepath) {
    CSVData result;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open CSV file: " << filepath << "\n";
        return result;
    }

    std::string line;
    // Read header line
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string colName;
        while (std::getline(ss, colName, ';')) {
            // clean whitespace
            colName.erase(0, colName.find_first_not_of(" \t\r\n"));
            colName.erase(colName.find_last_not_of(" \t\r\n") + 1);
            if (!colName.empty()) {
                result.headers.push_back(colName);
                result.columns[colName] = std::vector<double>();
            }
        }
    }

    // read rows
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string valStr;
        size_t colIdx = 0;

        while (std::getline(ss, valStr, ';') && colIdx < result.headers.size()) {
            valStr.erase(0, valStr.find_first_not_of(" \t\r\n"));
            valStr.erase(valStr.find_last_not_of(" \t\r\n") + 1);
            // change comma to point for std::stod
            std::replace(valStr.begin(), valStr.end(), ',', '.');

            if (!valStr.empty()) {
                try {
                    double val = std::stod(valStr);
                    result.columns[result.headers[colIdx]].push_back(val);
                } catch (...) {
                    // Non-numeric value encountered (e.g. empty cell)
                }
            }
            colIdx++;
        }
    }

    return result;
}

bool exportHistogram(
    const std::vector<double>& data,
    const std::string& filename,
    const std::string& title,
    size_t numBins,
    bool normalizeCount,
    bool normalizeDataValues,
    double width,
    double height
)
{
    if (data.empty() || numBins == 0) {
        std::cerr << "Error: Data is empty or number of bins is 0" << std::endl;
        return false;
    }
    
    // Copy data to allow optional normalization of input values
    std::vector<double> processedData = data;

    // Get min and max of data
    double minVal = Min(0, *std::min_element(processedData.begin(), processedData.end()));
    double maxVal = *std::max_element(processedData.begin(), processedData.end());
    
    // All data identical
    if (maxVal - minVal < 0.001)
    {
        maxVal += 0.001;
    }
    
    // Normalize data values to [0, 1] if requested
    if (normalizeDataValues && (maxVal > minVal))
    {
        for (double& x : processedData)
        {
            x = (x - minVal) / (maxVal - minVal);
        }
        minVal = 0.0;
        maxVal = 1.0;
    }

    // Divide into bins
    double binWidth = (maxVal - minVal) / static_cast<double>(numBins);
    std::vector<double> bins(numBins, 0.0);
    
    for (double val : processedData)
    {
        int binIndex = static_cast<int>((val - minVal) / binWidth);
        // Ensure last max element is in the last bin
        if (binIndex >= static_cast<int>(numBins))
        {
            binIndex = static_cast<int>(numBins - 1);
        }
        if (binIndex < 0)
        {
            binIndex = 0;
        }
        bins[binIndex]++;
    }
    
    // Normalize bin counts (relative frequencies, sum = 1.0)
    if (normalizeCount)
    {
        double totalCount = static_cast<double>(processedData.size());
        for (double& count : bins)
        {
            count /= totalCount;
        }
    }

    // Prepare range labels for each bin: format "low - high"
    std::vector<std::string> labelTexts(numBins);
    std::vector<std::wstring> wLabelTexts(numBins);
    double maxLabelPixelWidth = 0.0;

    for (size_t i = 0; i < numBins; ++i) {
        double low = minVal + static_cast<double>(i) * binWidth;
        double high = minVal + static_cast<double>(i + 1) * binWidth;
        std::string label = formatBound(low, binWidth) + " - " + formatBound(high, binWidth);
        labelTexts[i] = label;
        wLabelTexts[i] = std::wstring(label.begin(), label.end());

        std::vector<wchar_t>* tempWVec = toVector(wLabelTexts[i].c_str());
        double textWidth = GetTextWidth(tempWVec);
        if (textWidth > maxLabelPixelWidth) {
            maxLabelPixelWidth = textWidth;
        }
        delete tempWVec;
    }
    
    // pbPlots configuration
    BarPlotSeries* series = GetDefaultBarPlotSeriesSettings();
    series->ys = &bins;
    series->color = CreateRGBColor(0.2, 0.5, 0.85);
    
    BarPlotSettings* settings = GetDefaultBarPlotSettings();
    settings->width = width;
    settings->height = height;

    // Set title to feature name
    std::wstring wtitle(title.begin(), title.end());
    settings->title = toVector(wtitle.c_str());

    settings->yLabel = toVector(normalizeCount ? L"Relative Frequency" : L"Count");
    settings->barPlotSeries = new std::vector<BarPlotSeries*>{ series };
    settings->autoBoundaries = true;
    settings->showGrid = true;
    
    // Configure empty xLabels so pbPlots generates the plot without horizontal labels
    settings->autoLabels = false;
    settings->xLabels = new std::vector<StringReference*>();
    for (size_t i = 0; i < numBins; ++i) {
        settings->xLabels->push_back(CreateStringReference(toVector(L"")));
    }

    // Dynamically calculate paddings so vertical labels fit comfortably below the plot
    settings->autoPadding = false;
    double xPadding = std::floor(width * 0.10);
    double yPadding = std::max(height * 0.10, maxLabelPixelWidth + 25.0);
    settings->xPadding = xPadding;
    settings->yPadding = yPadding;

    // Fix spacing: manual group separation ensures bars are properly sized
    settings->autoSpacing = false;
    double availableWidth = width - 2.0 * xPadding;
    double groupSep = 2.0;
    if (numBins > 1) {
        double maxSep = (availableWidth * 0.25) / static_cast<double>(numBins - 1);
        if (groupSep > maxSep) {
            groupSep = maxSep;
        }
    }
    settings->groupSeparation = groupSep;
    settings->barSeparation = 0.0;
    settings->autoColor = false; // Use our series color instead of auto-cycling palette
    settings->barBorder = true;  // Crisp border around each bar
    
    // Generate plot and export as PNG
    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();
    StringReference* errorMessage = new StringReference();

    bool success = DrawBarPlotFromSettings(imageRef, settings, errorMessage);

    if (success) {
        // Draw vertical bin range labels directly onto canvas
        double barWidth = (availableWidth - groupSep * static_cast<double>(numBins - 1)) / static_cast<double>(numBins);
        double yPixelMax = height - yPadding;
        double labelY = yPixelMax + 8.0;

        for (size_t i = 0; i < numBins; ++i) {
            double barCenterX = xPadding + static_cast<double>(i) * (groupSep + barWidth) + 0.5 * barWidth;
            double labelX = std::floor(barCenterX - 13.0 / 2.0); // Font glyph height is 13px

            std::vector<wchar_t>* wtext = toVector(wLabelTexts[i].c_str());
            DrawTextUpwards(imageRef->image, labelX, labelY, wtext, GetGray(0.3));
            delete wtext;
        }

        std::vector<double>* pngData = ConvertToPNG(imageRef->image);
        WriteToFile(pngData, filename);
        delete pngData;
        std::cout << "Histogram saved as: " << filename << std::endl;
    } else {
        std::wcerr << L"Error while drawing histogram: ";
        for (wchar_t c : *errorMessage->string) {
            std::wcerr << c;
        }
        std::wcerr << std::endl;
    }
    
    // Clean memory
    DeleteImage(imageRef->image);
    delete imageRef;
    delete errorMessage;
    delete settings->barPlotSeries;
    for (StringReference* ref : *settings->xLabels) {
        delete ref->string;
        delete ref;
    }
    delete settings->xLabels;
    delete settings;
    delete series;

    return success;
}

void runHistogramMode() {
    std::cout << "\n=== HISTOGRAM GENERATOR MODE ===\n";
    std::cout << "Fill in name/path of the CSV file (example two.csv): ";
    std::string csvPath;
    std::cin >> csvPath;

    CSVData data = readCSV(csvPath);
    if (data.headers.empty()) {
        std::cout << "No valid data found\n";
        return;
    }

    // Filter columns that have actual data
    std::vector<std::string> validColumns;
    for (const auto& header : data.headers) {
        if (!data.columns[header].empty() && header != "Shape id") {
            validColumns.push_back(header);
        }
    }

    std::cout << "\nAvailable features/columns:\n";
    std::cout << "  0. [EXPORT ALL]\n";
    for (size_t i = 0; i < validColumns.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << validColumns[i] 
                  << " (" << data.columns[validColumns[i]].size() << " values)\n";
    }

    std::cout << "\nChoose an option (0 - " << validColumns.size() << "): ";
    int choice;
    std::cin >> choice;

    std::cout << "Normalize Count values? (1 = Yes, 0 = No): ";
    int normChoiceCount;
    std::cin >> normChoiceCount;
    bool normalizeCount = (normChoiceCount == 1);

    std::cout << "Normalize data values? (1 = Yes, 0 = No): ";
    int normChoiceData;
    std::cin >> normChoiceData;
    bool normalizeData = (normChoiceData == 1);
    
    std::cout << "Bin amount (default 20): ";
    int bins;
    std::cin >> bins;
    if (bins <= 0) bins = 20;

    // Helper to clean file names
    auto sanitizeName = [](std::string s) {
        std::replace(s.begin(), s.end(), ' ', '_');
        std::replace(s.begin(), s.end(), '/', '_');
        std::replace(s.begin(), s.end(), '\\', '_');
        return s;
    };

    if (choice == 0) {
        // Batch export for all features
        std::cout << "Exporting all histograms...\n";
        for (const auto& col : validColumns) {
            std::string outFile = "hist_" + sanitizeName(col) + ".png";
            exportHistogram(data.columns[col], outFile, col, bins, normalizeCount, normalizeData);
        }
        std::cout << "Done! All histograms saved as PNG.\n";
    } else if (choice > 0 && choice <= static_cast<int>(validColumns.size())) {
        // Export chosen column
        std::string col = validColumns[choice - 1];
        std::string outFile = "hist_" + sanitizeName(col) + ".png";
        exportHistogram(data.columns[col], outFile, col, bins, normalizeCount, normalizeData);
    } else {
        std::cout << "Invalid choice.\n";
    }
}
