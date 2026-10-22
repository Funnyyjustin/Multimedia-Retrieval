#pragma once

fs::path promptPathOrSelect()
{
	std::cout << "Type 1 to select a file, 2 to enter a path: ";
	int choice;
	std::cin >> choice;
	if (choice == 1)
	{
		return selectModelFile("ObjectFiles");
	}
	else if (choice == 2)
	{
		std::cout << "Enter the path: ";
		std::string path;
		std::cin >> path;
		return fs::path(path);
	}
	else
	{
		std::cout << "Invalid choice. Please try again.\n";
		return promptPathOrSelect();
	}
}

