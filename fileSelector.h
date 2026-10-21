#pragma once

#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

namespace fs = std::filesystem;

inline fs::path selectModelFile(const fs::path& baseDir = "ObjectFiles")
{
    if (!fs::exists(baseDir))
    {
        std::cerr << "Error: Map '" << baseDir.string() << "' does not exist.\n";
        return {};
    }
    else if (!fs::is_directory(baseDir))
    {
        std::cerr << "Error: Map '" << baseDir.string() << "' is not a directory.\n";
        return {};
    }

    // Get all subfolders
    std::vector<fs::path> classes;
    for (const auto& entry : fs::directory_iterator(baseDir))
    {
        if (entry.is_directory())
        {
            classes.push_back(entry.path());
        }
    }

    if (classes.empty())
    {
        std::cerr << "Error: No category found in " << baseDir.string() << "!\n";
        return {};
    }

    // Sort alphabetically
    std::sort(classes.begin(), classes.end());

    // Show all classes by number
    std::cout << "\n=== Choose a category ===\n";
    for (size_t i = 0; i < classes.size(); ++i)
    {
        std::cout << (i + 1) << ". " << classes[i].filename().string() << "\n";
    }

    int classChoice = 0;
    while (true)
    {
        std::cout << "\nType in the number of the class: (1-" << classes.size() << "): ";
        if (std::cin >> classChoice && classChoice >= 1 && classChoice <= static_cast<int>(classes.size()))
        {
            break;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid choice, try again.\n";
    }

    fs::path selectedClass = classes[classChoice - 1];

    // 3. Get all files (.obj or .off) within directory
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(selectedClass))
    {
        if (entry.is_regular_file())
        {
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".obj" || ext == ".off")
            {
                files.push_back(entry.path());
            }
        }
    }

    if (files.empty())
    {
        std::cerr << "No .obj or .off files found in " << selectedClass.string() << "!\n";
        return {};
    }

    std::sort(files.begin(), files.end());

    // Show all files, numbered
    std::cout << "\n=== Choose a file in " << selectedClass.filename().string() << " ===\n";
    for (size_t i = 0; i < files.size(); ++i)
    {
        std::cout << (i + 1) << ". " << files[i].filename().string() << "\n";
    }

    int fileChoice = 0;
    while (true)
    {
        std::cout << "\nType in the number of the file: (1-" << files.size() << "): ";
        if (std::cin >> fileChoice && fileChoice >= 1 && fileChoice <= static_cast<int>(files.size()))
        {
            break;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Invalid choice, try again.\n";
    }

    return files[fileChoice - 1];
}

inline fs::path selectModelFile(int id)
{
    fs::path baseDir = "ObjectFiles";
    if (!fs::exists(baseDir))
    {
        std::cerr << "Error: Map '" << baseDir.string() << "' does not exist.\n";
        return {};
    }
    else if (!fs::is_directory(baseDir))
    {
        std::cerr << "Error: Map '" << baseDir.string() << "' is not a directory.\n";
        return {};
    }

    // Get all subfolders
    std::vector<fs::path> classes;
    for (const auto& entry : fs::directory_iterator(baseDir))
    {
        if (entry.is_directory())
        {
            classes.push_back(entry.path());
        }
    }

    // Sort alphabetically
    std::sort(classes.begin(), classes.end());

    for (int i = 0; i < classes.size(); i++)
    {
        fs::path path = classes[i];
        for (const auto& entry : fs::directory_iterator(path))
        {
            if (entry.is_regular_file())
            {
                if (id <= 0)
                {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".obj" || ext == ".off")
                    {
                        return entry.path();
                    }
                }
                id--;
            }
        }
    }
}