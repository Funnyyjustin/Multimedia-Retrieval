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
#include <cctype>
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct HistogramFeatureInfo {
    double minVal;
    double maxVal;
    std::wstring xLabel;
    std::string description;
};

static HistogramFeatureInfo getHistogramFeatureInfo(const std::string& feature) {
    if (feature == "A3") {
        return { 0.0, M_PI, L"Angle (radians) [0 - PI]", "Angle between 3 random points" };
    } else if (feature == "D1") {
        return { 0.0, 0.90, L"Distance to Barycenter [0 - 0.90]", "Distance from barycenter to random point" };
    } else if (feature == "D2") {
        return { 0.0, 1.75, L"Distance between 2 Points [0 - 1.75]", "Distance between 2 random points" };
    } else if (feature == "D3") {
        return { 0.0, 0.75, L"Square Root of Area [0 - 0.75]", "Square root of triangle area" };
    } else if (feature == "D4") {
        return { 0.0, 0.60, L"Cube Root of Volume [0 - 0.60]", "Cube root of tetrahedron volume" };
    }
    return { 0.0, 1.0, L"Feature Value", "" };
}

static void HSVtoRGB(double h, double s, double v, double& r, double& g, double& b) {
    if (s <= 0.0) {
        r = g = b = v;
        return;
    }
    while (h >= 360.0) h -= 360.0;
    while (h < 0.0) h += 360.0;

    double c = v * s;
    double x = c * (1.0 - std::abs(std::fmod(h / 60.0, 2.0) - 1.0));
    double m = v - c;

    double r1 = 0, g1 = 0, b1 = 0;
    if (h < 60.0) {
        r1 = c; g1 = x; b1 = 0;
    } else if (h < 120.0) {
        r1 = x; g1 = c; b1 = 0;
    } else if (h < 180.0) {
        r1 = 0; g1 = c; b1 = x;
    } else if (h < 240.0) {
        r1 = 0; g1 = x; b1 = c;
    } else if (h < 300.0) {
        r1 = x; g1 = 0; b1 = c;
    } else {
        r1 = c; g1 = 0; b1 = x;
    }

    r = r1 + m;
    g = g1 + m;
    b = b1 + m;
}

static std::string sanitizeName(std::string s) {
    for (char& c : s) {
        if (c == ' ' || c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|') {
            c = '_';
        }
    }
    return s;
}

std::vector<double> computeFixedHistogram(
    const std::vector<double>& values, 
    size_t numBins, 
    double minVal, 
    double maxVal)
{
    std::vector<double> bins(numBins, 0.0);
    if (values.empty() || numBins == 0 || maxVal <= minVal) return bins;

    double binWidth = (maxVal - minVal) / static_cast<double>(numBins);

    for (double val : values)
    {
        if (val <= minVal) {
            bins[0]++;
        } else if (val >= maxVal) {
            bins[numBins - 1]++;
        } else {
            int idx = static_cast<int>((val - minVal) / binWidth);
            if (idx >= static_cast<int>(numBins)) idx = static_cast<int>(numBins - 1);
            bins[idx]++;
        }
    }

    double total = static_cast<double>(values.size());
    for (double& b : bins) {
        b /= total;
    }

    return bins;
}

static std::string formatBound(double val, double binWidth) {
    double absVal = std::abs(val);
    if (absVal < 1e-9) {
        return "0";
    }
    
    if (absVal >= 100000.0 || (absVal < 0.001 && absVal > 0.0)) {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(1) << val;
        return ss.str();
    }
    
    if (binWidth >= 1.0) {
        return std::to_string(static_cast<long long>(std::round(val)));
    }
    
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
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string colName;
        while (std::getline(ss, colName, ';')) {
            colName.erase(0, colName.find_first_not_of(" \t\r\n"));
            colName.erase(colName.find_last_not_of(" \t\r\n") + 1);
            if (!colName.empty()) {
                result.headers.push_back(colName);
                result.columns[colName] = std::vector<double>();
            }
        }
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string valStr;
        size_t colIdx = 0;

        while (std::getline(ss, valStr, ';') && colIdx < result.headers.size()) {
            valStr.erase(0, valStr.find_first_not_of(" \t\r\n"));
            valStr.erase(valStr.find_last_not_of(" \t\r\n") + 1);
            std::replace(valStr.begin(), valStr.end(), ',', '.');

            if (!valStr.empty()) {
                try {
                    double val = std::stod(valStr);
                    result.columns[result.headers[colIdx]].push_back(val);
                } catch (...) {
                }
            }
            colIdx++;
        }
    }

    return result;
}

FeatureDatabase parseFeatureCSV(const std::string& filepath) {
    FeatureDatabase db;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open CSV file: " << filepath << "\n";
        return db;
    }

    std::string line;
    if (!std::getline(file, line)) return db;

    // Parse header line
    std::vector<std::string> headers;
    {
        std::stringstream ss(line);
        std::string col;
        while (std::getline(ss, col, ';')) {
            col.erase(0, col.find_first_not_of(" \t\r\n"));
            col.erase(col.find_last_not_of(" \t\r\n") + 1);
            headers.push_back(col);
        }
    }

    int shapeIdCol = -1;
    int classCol = -1;

    struct HistCol {
        std::string featureName;
        size_t binIndex;
    };
    std::map<size_t, HistCol> histCols;
    std::vector<size_t> elemCols;
    std::map<std::string, bool> histFeatureSeen;

    for (size_t i = 0; i < headers.size(); ++i) {
        const std::string& h = headers[i];
        if (h.empty()) continue;

        if (h == "Shape id" || h == "Shape ID" || h == "Shape_id" || h == "id") {
            shapeIdCol = static_cast<int>(i);
            continue;
        }
        if (h == "Class" || h == "class" || h == "Label" || h == "label") {
            classCol = static_cast<int>(i);
            continue;
        }

        size_t under = h.rfind('_');
        if (under != std::string::npos && under + 1 < h.size()) {
            std::string prefix = h.substr(0, under);
            std::string suffix = h.substr(under + 1);
            bool allDigits = true;
            for (char c : suffix) {
                if (!std::isdigit(static_cast<unsigned char>(c))) {
                    allDigits = false;
                    break;
                }
            }
            if (allDigits && (prefix == "A3" || prefix == "D1" || prefix == "D2" || prefix == "D3" || prefix == "D4")) {
                size_t binIdx = static_cast<size_t>(std::stoul(suffix));
                histCols[i] = { prefix, binIdx };
                if (!histFeatureSeen[prefix]) {
                    histFeatureSeen[prefix] = true;
                    db.histogramFeatureNames.push_back(prefix);
                }
                continue;
            }
        }

        elemCols.push_back(i);
        db.elementaryFeatureNames.push_back(h);
    }

    std::map<std::string, bool> classSeen;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ';')) {
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            std::replace(cell.begin(), cell.end(), ',', '.');
            cells.push_back(cell);
        }

        if (cells.empty()) continue;

        ShapeFeatures shape;
        if (shapeIdCol >= 0 && shapeIdCol < static_cast<int>(cells.size()) && !cells[shapeIdCol].empty()) {
            try { shape.shapeId = std::stoi(cells[shapeIdCol]); } catch (...) {}
        }
        if (classCol >= 0 && classCol < static_cast<int>(cells.size()) && !cells[classCol].empty()) {
            shape.className = cells[classCol];
        } else {
            shape.className = "DefaultClass";
        }

        for (const auto& hName : db.histogramFeatureNames) {
            shape.histograms[hName] = std::vector<double>(20, 0.0);
        }

        for (const auto& pair : histCols) {
            size_t colIdx = pair.first;
            if (colIdx < cells.size() && !cells[colIdx].empty()) {
                try {
                    double val = std::stod(cells[colIdx]);
                    const auto& hc = pair.second;
                    if (hc.binIndex < shape.histograms[hc.featureName].size()) {
                        shape.histograms[hc.featureName][hc.binIndex] = val;
                    }
                } catch (...) {}
            }
        }

        for (size_t colIdx : elemCols) {
            if (colIdx < cells.size() && !cells[colIdx].empty()) {
                try {
                    double val = std::stod(cells[colIdx]);
                    shape.elementary[headers[colIdx]] = val;
                } catch (...) {}
            }
        }

        if (!classSeen[shape.className]) {
            classSeen[shape.className] = true;
            db.classNames.push_back(shape.className);
        }
        db.classes[shape.className].push_back(shape);
    }

    return db;
}

bool exportClassHistogramLinePlot(
    const std::string& className,
    const std::string& featureName,
    const std::vector<ShapeFeatures>& shapes,
    const std::string& filename,
    double width,
    double height
)
{
    if (shapes.empty()) {
        std::cerr << "Error: No shapes provided for class " << className << std::endl;
        return false;
    }

    HistogramFeatureInfo info = getHistogramFeatureInfo(featureName);
    const size_t numBins = 20;
    double binWidth = (info.maxVal - info.minVal) / static_cast<double>(numBins);

    // Compute bin centers in physical domain
    std::vector<double> xs(numBins);
    for (size_t i = 0; i < numBins; ++i) {
        xs[i] = info.minVal + (static_cast<double>(i) + 0.5) * binWidth;
    }

    ScatterPlotSettings* settings = GetDefaultScatterPlotSettings();
    settings->width = width;
    settings->height = height;

    std::string titleStr = className + " - " + featureName + " (" + std::to_string(shapes.size()) + " shapes)";
    std::wstring wtitle(titleStr.begin(), titleStr.end());
    settings->title = toVector(wtitle.c_str());
    settings->xLabel = toVector(info.xLabel.c_str());
    settings->yLabel = toVector(L"Relative Frequency");

    settings->showGrid = true;
    settings->gridColor = CreateRGBColor(0.85, 0.85, 0.85);

    double maxY = 0.05;
    size_t numShapes = shapes.size();

    for (size_t k = 0; k < numShapes; ++k) {
        const auto& shape = shapes[k];
        auto it = shape.histograms.find(featureName);
        if (it == shape.histograms.end() || it->second.empty()) continue;

        const std::vector<double>& bins = it->second;
        for (double val : bins) {
            if (!std::isnan(val) && !std::isinf(val) && val > maxY) {
                maxY = val;
            }
        }

        ScatterPlotSeries* series = GetDefaultScatterPlotSeriesSettings();
        series->linearInterpolation = true;
        series->lineThickness = 1.8;
        series->pointType = toVector(L"pixels");
        series->lineType = toVector(L"solid");

        *series->xs = xs;
        *series->ys = bins;

        // Generate distinct color per shape
        double hue = (numShapes <= 1) ? 210.0 : std::fmod(k * (360.0 / numShapes), 360.0);
        double sat = 0.85;
        double val = 0.85;
        double r, g, b;
        HSVtoRGB(hue, sat, val, r, g, b);

        series->color = CreateRGBColor(r, g, b);

        settings->scatterPlotSeries->push_back(series);
    }

    settings->autoBoundaries = false;
    settings->xMin = info.minVal;
    settings->xMax = info.maxVal;
    settings->yMin = 0.0;
    settings->yMax = std::max(0.05, maxY * 1.15);

    settings->autoPadding = true;

    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();
    StringReference* errorMessage = CreateStringReference(toVector(L""));

    bool success = DrawScatterPlotFromSettings(imageRef, settings, errorMessage);

    if (success) {
        std::vector<double>* pngData = ConvertToPNG(imageRef->image);
        WriteToFile(pngData, filename);
        std::cout << "[Saved Line Chart] " << filename << " (" << shapes.size() << " lines)\n";
    } else {
        std::wcerr << L"Error while drawing scatter plot: ";
        if (errorMessage && errorMessage->string) {
            for (wchar_t c : *errorMessage->string) {
                std::wcerr << c;
            }
        }
        std::wcerr << std::endl;
    }

    FreeAllocations();

    return success;
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
    
    std::vector<double> processedData = data;

    double minVal = *std::min_element(processedData.begin(), processedData.end());
    double maxVal = *std::max_element(processedData.begin(), processedData.end());
    
    if (maxVal - minVal < 0.001)
    {
        maxVal += 0.001;
    }
    
    if (normalizeDataValues && (maxVal > minVal))
    {
        for (double& x : processedData)
        {
            x = (x - minVal) / (maxVal - minVal);
        }
        minVal = 0.0;
        maxVal = 1.0;
    }

    double binWidth = (maxVal - minVal) / static_cast<double>(numBins);
    std::vector<double> bins(numBins, 0.0);
    
    for (double val : processedData)
    {
        int binIndex = static_cast<int>((val - minVal) / binWidth);
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
    
    if (normalizeCount)
    {
        double totalCount = static_cast<double>(processedData.size());
        for (double& count : bins)
        {
            count /= totalCount;
        }
    }

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
    }
    
    BarPlotSeries* series = GetDefaultBarPlotSeriesSettings();
    *series->ys = bins;
    series->color = CreateRGBColor(0.2, 0.5, 0.85);
    
    BarPlotSettings* settings = GetDefaultBarPlotSettings();
    settings->width = width;
    settings->height = height;

    std::wstring wtitle(title.begin(), title.end());
    settings->title = toVector(wtitle.c_str());

    settings->yLabel = toVector(normalizeCount ? L"Relative Frequency" : L"Count");
    settings->barPlotSeries->push_back(series);
    settings->autoBoundaries = true;
    settings->showGrid = true;
    
    settings->autoLabels = false;
    for (size_t i = 0; i < numBins; ++i) {
        settings->xLabels->push_back(CreateStringReference(toVector(L"")));
    }

    settings->autoPadding = false;
    double xPadding = std::floor(width * 0.10);
    double yPadding = std::max(height * 0.10, maxLabelPixelWidth + 25.0);
    settings->xPadding = xPadding;
    settings->yPadding = yPadding;

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
    settings->autoColor = false;
    settings->barBorder = true;
    
    RGBABitmapImageReference* imageRef = CreateRGBABitmapImageReference();
    StringReference* errorMessage = CreateStringReference(toVector(L""));

    bool success = DrawBarPlotFromSettings(imageRef, settings, errorMessage);

    if (success) {
        double barWidth = (availableWidth - groupSep * static_cast<double>(numBins - 1)) / static_cast<double>(numBins);
        double yPixelMax = height - yPadding;
        double labelY = yPixelMax + 8.0;

        for (size_t i = 0; i < numBins; ++i) {
            double barCenterX = xPadding + static_cast<double>(i) * (groupSep + barWidth) + 0.5 * barWidth;
            double labelX = std::floor(barCenterX - 13.0 / 2.0);

            std::vector<wchar_t>* wtext = toVector(wLabelTexts[i].c_str());
            DrawTextUpwards(imageRef->image, labelX, labelY, wtext, GetGray(0.3));
        }

        std::vector<double>* pngData = ConvertToPNG(imageRef->image);
        WriteToFile(pngData, filename);
        std::cout << "[Saved Bar Chart] " << filename << std::endl;
    } else {
        std::wcerr << L"Error while drawing histogram: ";
        if (errorMessage && errorMessage->string) {
            for (wchar_t c : *errorMessage->string) {
                std::wcerr << c;
            }
        }
        std::wcerr << std::endl;
    }
    
    FreeAllocations();

    return success;
}

void runHistogramMode() {
    std::filesystem::create_directories("images");

    std::cout << "\n=== HISTOGRAM & FEATURE EXPORT MODE ===\n";
    std::cout << "Enter CSV filename or path (e.g. full_featuress_3.csv): ";
    std::string csvPath;
    std::cin >> csvPath;

    if (!csvPath.empty() && (csvPath.front() == '"' || csvPath.front() == '\'')) {
        csvPath = csvPath.substr(1, csvPath.size() - 2);
    }

    FeatureDatabase db = parseFeatureCSV(csvPath);

    if (db.classNames.empty()) {
        std::cout << "\nNo class data found in CSV. Checking for simple column CSV...\n";
        CSVData legacyData = readCSV(csvPath);
        if (legacyData.headers.empty()) {
            std::cout << "Error: Could not read valid data from " << csvPath << "\n";
            return;
        }

        std::vector<std::string> validColumns;
        for (const auto& header : legacyData.headers) {
            if (!legacyData.columns[header].empty() && header != "Shape id") {
                validColumns.push_back(header);
            }
        }

        std::cout << "\nAvailable features/columns:\n";
        std::cout << "  0. [EXPORT ALL]\n";
        for (size_t i = 0; i < validColumns.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << validColumns[i] 
                      << " (" << legacyData.columns[validColumns[i]].size() << " values)\n";
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

        if (choice == 0) {
            std::cout << "Exporting all histograms...\n";
            for (const auto& col : validColumns) {
                std::string outFile = "images/hist_" + sanitizeName(col) + ".png";
                exportHistogram(legacyData.columns[col], outFile, col, bins, normalizeCount, normalizeData);
            }
            std::cout << "Done! All histograms saved as PNG in images/ folder.\n";
        } else if (choice > 0 && choice <= static_cast<int>(validColumns.size())) {
            std::string col = validColumns[choice - 1];
            std::string outFile = "images/hist_" + sanitizeName(col) + ".png";
            exportHistogram(legacyData.columns[col], outFile, col, bins, normalizeCount, normalizeData);
        }
        return;
    }

    int totalShapes = 0;
    for (const auto& c : db.classNames) {
        totalShapes += static_cast<int>(db.classes[c].size());
    }

    std::cout << "\nSuccessfully loaded " << db.classNames.size() << " classes (" 
              << totalShapes << " total shapes) from " << csvPath << ".\n";

    // 1. Select Class
    std::cout << "\n--- SELECT CLASS ---\n";
    std::cout << "  0. [ALL CLASSES] (" << db.classNames.size() << " classes)\n";
    for (size_t i = 0; i < db.classNames.size(); ++i) {
        std::cout << "  " << (i + 1) << ". " << db.classNames[i] 
                  << " (" << db.classes[db.classNames[i]].size() << " shapes)\n";
    }

    std::cout << "\nChoose class (0 - " << db.classNames.size() << "): ";
    int classChoice;
    std::cin >> classChoice;

    std::vector<std::string> selectedClasses;
    if (classChoice == 0) {
        selectedClasses = db.classNames;
    } else if (classChoice > 0 && classChoice <= static_cast<int>(db.classNames.size())) {
        selectedClasses.push_back(db.classNames[classChoice - 1]);
    } else {
        std::cout << "Invalid class choice.\n";
        return;
    }

    // 2. Select Feature
    std::cout << "\n--- SELECT FEATURE ---\n";
    std::vector<std::pair<std::string, bool>> featureList; // (featureName, isHistogram)

    if (!db.histogramFeatureNames.empty()) {
        std::cout << "Shape Distribution Histograms (Multi-line chart, 1 line per object):\n";
        for (const auto& hName : db.histogramFeatureNames) {
            HistogramFeatureInfo info = getHistogramFeatureInfo(hName);
            featureList.push_back({ hName, true });
            std::cout << "  " << featureList.size() << ". " << hName 
                      << " (" << info.description << ")\n";
        }
    }

    if (!db.elementaryFeatureNames.empty()) {
        std::cout << "\nElementary Features (Bar chart distribution across objects in class):\n";
        for (const auto& eName : db.elementaryFeatureNames) {
            featureList.push_back({ eName, false });
            std::cout << "  " << featureList.size() << ". " << eName << "\n";
        }
    }

    std::cout << "\nBatch Options:\n";
    std::cout << "  0. [ALL FEATURES] (Export both histogram line charts & elementary bar charts)\n";
    if (!db.histogramFeatureNames.empty()) {
        std::cout << "  " << (featureList.size() + 1) << ". [ALL HISTOGRAM LINE CHARTS ONLY] (A3, D1, D2, D3, D4)\n";
    }

    std::cout << "\nChoose feature option (0 - " << (featureList.size() + (db.histogramFeatureNames.empty() ? 0 : 1)) << "): ";
    int featChoice;
    std::cin >> featChoice;

    std::vector<std::pair<std::string, bool>> selectedFeatures;
    if (featChoice == 0) {
        selectedFeatures = featureList;
    } else if (!db.histogramFeatureNames.empty() && featChoice == static_cast<int>(featureList.size() + 1)) {
        for (const auto& item : featureList) {
            if (item.second) selectedFeatures.push_back(item);
        }
    } else if (featChoice > 0 && featChoice <= static_cast<int>(featureList.size())) {
        selectedFeatures.push_back(featureList[featChoice - 1]);
    } else {
        std::cout << "Invalid feature choice.\n";
        return;
    }

    // Check if any elementary features are selected for bar chart options
    bool hasElementary = false;
    for (const auto& f : selectedFeatures) {
        if (!f.second) { hasElementary = true; break; }
    }

    int elemBins = 20;
    bool normCount = true;
    bool normData = false;

    if (hasElementary) {
        std::cout << "\n[Bar Chart Options for Elementary Features]\n";
        std::cout << "Normalize Count values? (1 = Yes, 0 = No, default 1): ";
        int nc = 1;
        std::cin >> nc;
        normCount = (nc == 1);

        std::cout << "Normalize data values to [0, 1]? (1 = Yes, 0 = No, default 0): ";
        int nd = 0;
        std::cin >> nd;
        normData = (nd == 1);

        std::cout << "Number of bins (default 20): ";
        int nb = 20;
        std::cin >> nb;
        if (nb > 0) elemBins = nb;
    }

    std::cout << "\nExporting charts to images/ folder...\n";
    int exportedCount = 0;

    for (const auto& clsName : selectedClasses) {
        const auto& shapes = db.classes[clsName];
        for (const auto& feat : selectedFeatures) {
            const std::string& featName = feat.first;
            bool isHist = feat.second;

            std::string outFile = "images/hist_" + sanitizeName(clsName) + "_" + sanitizeName(featName) + ".png";

            if (isHist) {
                if (exportClassHistogramLinePlot(clsName, featName, shapes, outFile)) {
                    exportedCount++;
                }
            } else {
                std::vector<double> vals;
                for (const auto& s : shapes) {
                    auto it = s.elementary.find(featName);
                    if (it != s.elementary.end()) {
                        vals.push_back(it->second);
                    }
                }
                std::string title = clsName + " - " + featName;
                if (exportHistogram(vals, outFile, title, elemBins, normCount, normData)) {
                    exportedCount++;
                }
            }
        }
    }

    std::cout << "\nDone! Successfully exported " << exportedCount << " chart(s) to images/ directory.\n";
}
