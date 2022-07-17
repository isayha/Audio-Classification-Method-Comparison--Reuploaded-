# Audio Classification Methodology Comparison
A Python-based audio classification project for CPSC 473 - Introduction to Data Mining

## Instructions

### Dependencies
- This program requires:
  - Librosa, a Python package "for music and audio analysis" (developed using Librosa 0.8.1)
  - Pandas, a Python package for data analysis (developed using Pandas 1.3.4)
  - Scikit-Learn, a Python package for data mining and machine learning (developed using Scikit-Learn 1.0.1)
- All of the packages list above, as well as all of their dependencies, can be installed using the package manager `pip` by simply executing the following shell command: `pip install -r required_packages.txt`
  - Note that the shell instance in question must be navigated to the root directory of this project (i.e. where `required_packages.txt` is found)

### How to run
- To run the program, simply run `main.py`, either using the command `python` on the command line (while in the same directory as the source files), or the IDE of your choice
  - Instructions regarding choice of pre-processing techniques, etc. are all provided in-program
  - **If you wish to skip preprocessing the .wav files** (A very lengthy process as per `.\preprocess_times.txt`), preprocessing has already been performed using all 3 methodologies, and the outputs have been saved
    - Simply copy and paste the files found within `.\wavDataPP0\` (or `\wavDataPP1\`, or `\wavDataPP2\`) into `.\wavData\`, where the X in `PPX` indicates which of the 3 preprocessing methodologies the contents were generated using (see the results section in the project paper - `.\CPSC473_project_paper.pdf` - for more clarification)
  - Alternatively, if you would like to preprocess the .wav files yourself, they can be downloaded via the URL found in `.\wav_source.txt`
    - The .wav source used is the UrbanSound8k dataset; all compilation credit goes to Justin Salamon, Christopher Jacoby and Juan Pablo Bello. The .wav files included originate from www.freesound.org.
    - After filling out a brief download form, you should be provided with a tarball. Within, there should be folders containing .wav files named `fold1` through `fold10` (`.\UrbanSound8K\audio\fold1 ... fold10\`). Place these folders directly into the project subdirectory `.\wavs\`.
      - Thus, the project subdirectory `.\wavs\` should, in turn, have subdirectories `.\wavs\fold1 ... fold10\`

### Output
- Once preprocessed, data (CSVs) for each .wav for each of the ten (10) folds of .wavs can be found in `.\wavData\` in files `.\wavData\fold1Data.txt ... fold10Data.txt`
- Expected classifications (ordered) for each .wav within the designated test fold can be found in `.\ExpectedResults.txt` (following a run of the program)
- KNN and Logistic Regression model output (classifications) for each .wav within the designated test fold can be found in `.\KNNOutput.txt` and `.\LogRegOutput.txt`, respectively (following a run of the program)