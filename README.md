# Genre Classification
Approach to genre classification using a Convolutional Neural Network to analyze Mel-Frequency Cepstral Coeffiecients.
This program utilizes the names of folders in order to build the list of genres for classifications, for this reason: <br />
***DATA MUST BE PLACED USING THE SAME FILE STRUCTURE AS SHOWN IN THIS REPOSITORY*** <br />
```
*This program has currently only been testing on Windows, errors could occur using another operating system*
```
## Required Libraries
- numpy
- pandas
- matplotlib
- librosa
- tensorflow
- sklearn

## Instructions for Running Program
1. Download program
   - If necessary download the data files which can also be found on Kaggle here: <br />
      https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification <br />
   - The only required data are the .wav files
2. Ensure the data is placed using the same file structure shown in the repository
   > ex. *Program Location*//Data//genres_original//*Genre Folders*
   - For clarity: The "Data" folder is located in the directory containing the program
3. Open and Run program 
   - **Currently trains model on every run**
   - Approximately 10 minute run time depending on machine
4. The output will occur as listed below:
```
- Display the current training progress
- Display a plot of the training and validation accuracy/error
- Print the overall Testing accuracy
- Display the confusion Matix
```
