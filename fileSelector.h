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
    if (!fs::exists(baseDir) || !fs::is_directory(baseDir))
    {
        std::cerr << "Fout: Map '" << baseDir.string() << "' bestaat niet of is geen directory.\n";
        return {};
    }

    // 1. Verzamel alle submappen (classes)
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
        std::cerr << "Geen categorieen gevonden in " << baseDir.string() << "!\n";
        return {};
    }

    // Sorteer alfabetisch voor overzichtelijkheid
    std::sort(classes.begin(), classes.end());

    // 2. Toon alle classes genummerd
    std::cout << "\n=== Kies een categorie ===\n";
    for (size_t i = 0; i < classes.size(); ++i)
    {
        std::cout << (i + 1) << ". " << classes[i].filename().string() << "\n";
    }

    int classChoice = 0;
    while (true)
    {
        std::cout << "\nVoer het nummer van de gewenste categorie in (1-" << classes.size() << "): ";
        if (std::cin >> classChoice && classChoice >= 1 && classChoice <= static_cast<int>(classes.size()))
        {
            break;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Ongeldige keuze, probeer opnieuw.\n";
    }

    fs::path selectedClass = classes[classChoice - 1];

    // 3. Verzamel bestanden (.obj / .off) binnen de gekozen categorie
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
        std::cerr << "Geen .obj of .off bestanden gevonden in " << selectedClass.string() << "!\n";
        return {};
    }

    std::sort(files.begin(), files.end());

    // 4. Toon alle bestanden genummerd
    std::cout << "\n=== Kies een bestand in " << selectedClass.filename().string() << " ===\n";
    for (size_t i = 0; i < files.size(); ++i)
    {
        std::cout << (i + 1) << ". " << files[i].filename().string() << "\n";
    }

    int fileChoice = 0;
    while (true)
    {
        std::cout << "\nVoer het nummer van het bestand in (1-" << files.size() << "): ";
        if (std::cin >> fileChoice && fileChoice >= 1 && fileChoice <= static_cast<int>(files.size()))
        {
            break;
        }
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Ongeldige keuze, probeer opnieuw.\n";
    }

    return files[fileChoice - 1];
}
