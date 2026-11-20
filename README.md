# Multimedia Retrieval

This repository contains a project for the course INFOMR Multimedia Retrieval from Utrecht University.
We use openGL

## How to compile and run
Clone or download this repository and open the solution "Multimedia Retrieval.sln" in an IDE (e.g. Visual Studio Community).
After opening the solution, build and run the solution. A console window should open prompting you to choose one of several modes.

## How to use
The console window will prompt you with a couple options, for manual querying, select 5. for the showcase from the presentations, select either 0 or 8.

## How to use showcase application
After selecting 8 or 0, a window will open. On the left you can select classes and objects. The classes can be ordered by Name or AUC. 
At the top there are 5 tabs: 
1. 3D Query Studio: Select an object on the left and it will query that object.
2. 3D Preprocessing & Outliers: shows histograms with some processing data and a table with outliers in the bottom-right.
3. Features & Histograms: shows single value and histogram features for the selected object.
4. 2D t-SNE Projection: shows t-SNE projection
5. System Evalutation & ROC Curves: shows overall ROC curve and ROC curve for selected class

## Apple warning
We used an external implementation of quickhull that might not work on Apple devices.
Quickhull is only used in feature extraction so the rest of the application should run like normal.

# In case running from IDE does not work
Alternatively, if that does not work, then follow these steps:
1. Open a command or Powershell prompt.
2. Navigate to where the root folder of the project is located on your device.
3. Create a new directory called "build" within the root folder of the project.
4. Navigate to this "build" folder inside your prompt, and type in the command "cmake ..." (without the "). On some devices, this will not work - you should use the command "cmake -G "Visual Studio 17 2022" .." (again, without the (outer) ") instead. This will build a VS Community solution that you can then open. This solution will be located in the "build" folder. Open it with your IDE.
5. Then, you can hit "Run" at the top of the screen, and the console should pop up.
