# Barbell Exercise Recognition using Sensor Data

This project aims to build a machine learning model for recognizing different types of barbell exercises using accelerometer and gyroscope data from wearable devices. The data is processed, cleaned, and temporally abstracted to extract meaningful features for training and testing machine learning models.

## Project Structure
```bash
|-- Code/
| |-- pycache/
| |-- DataTransformation.py # Script for data transformation
| |-- FrequencyAbstraction.py # Script for frequency domain feature extraction
| |-- LearningAlgorithms.py # Script for implementing learning algorithms
| |-- TemporalAbstraction.py # Script for temporal abstraction of the time-series data
| |-- count_repetitions.py # Script for counting exercise repetitions
| |-- features.ipynb # Jupyter notebook for feature extraction and analysis
| |-- make_dataset.py # Script to process and merge accelerometer & gyroscope data
| |-- remove_outlier.py # Script to clean data by removing outliers
| |-- train_model.ipynb # Jupyter notebook for training the machine learning model
|Files/
| |-- cleaned_feature.csv # CSV with extracted features
| |-- cleaned_file.csv # Cleaned sensor data
| |-- cleaned_file_outlier.csv # Cleaned data after outlier removal
| |-- README.md # Project documentation (this file)
```

## Setup and Installation

To run this project, you'll need to have Python installed, along with the necessary dependencies:

1. Clone the repository:

    ```bash
    git clone https://github.com/kushpatel16112/Machine-Learning.git
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The dataset consists of time-series data collected from MetaMotion sensors, including accelerometer and gyroscope readings for barbell exercises. The data is stored in the `data/raw/` directory.

- **Data Columns:**
  - `acc_x`, `acc_y`, `acc_z`: Accelerometer readings along the x, y, and z axes.
  - `gyr_x`, `gyr_y`, `gyr_z`: Gyroscope readings along the x, y, and z axes.
  - `participant`: Identifier for the participant.
  - `label`: The label indicating the type of barbell exercise.
  - `category`: A more generalized category of the exercise.
  - `set`: Set number for the exercise repetition.

## Data Processing

### 1. Merging and Cleaning Data

The `make_dataset.py` script is used to read multiple CSV files, merge the accelerometer and gyroscope data, and resample the data into consistent time intervals (200ms). This is stored in the `data/interim/` folder.

**Steps to Run:**
```bash
python scripts/make_dataset.py
```

- **Input**: Raw sensor data (data/raw/MetaMotion/*.csv)
- **Output**: Cleaned and merged data (data/interim/cleaned_file.csv)

## 2. Outlier Detection and Removal

The `remove_outlier.py` script is used to detect and remove outliers from the sensor data based on Z-scores. Outliers are replaced by the mean of the respective columns. The script ensures that data quality is maintained before further processing or analysis.

### How It Works:
- **Input**: The script takes in a cleaned sensor data file (`data/interim/cleaned_file.csv`).
- **Outlier Detection**: Outliers are detected using Z-scores for each sensor column (e.g., `acc_x`, `acc_y`, `acc_z`, `gyr_x`, `gyr_y`, `gyr_z`).
- **Outlier Replacement**: Detected outliers are replaced by the mean of the respective column.
- **Output**: The resulting data is saved to a new file (`data/interim/cleaned_file_outlier.csv`).

### Steps to Run the Script:

1. Navigate to the project's root directory.
2. Execute the following command in your terminal:

   ```bash
   python scripts/remove_outlier.py
   ```

## 3. Temporal Abstraction

The `TemporalAbstraction.py` script performs temporal abstraction on the numerical data by applying various aggregation functions (mean, max, min, etc.) over a rolling window. This is an essential step in preparing the data for machine learning models by summarizing information in time windows.

### Example Usage:

```python
from TemporalAbstraction import NumericalAbstraction

# Create an instance of the NumericalAbstraction class
numerical_abstraction = NumericalAbstraction()

# Abstract the numerical data
data_abstracted = numerical_abstraction.abstract_numerical(
    data_table=df_cleaned,
    cols=['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'],
    window_size=5,
    aggregation_function='mean'
)
```
## Model Training

The `model_training.py` script (to be developed) will use the processed data to train a machine learning model for classifying different types of barbell exercises. This script will take in preprocessed data and generate a trained model, which will be saved for future use.

### Steps to Run:

#### 1. Run the data processing pipeline to clean and prepare the data:
Before training the model, ensure that the data is cleaned and outliers are removed:

```bash
python scripts/make_dataset.py
python scripts/remove_outlier.py
```

#### 2. Optionally, perform temporal abstraction on the data:
```bash
python scripts/temporal_abstraction.py
```

#### 3. Train the machine learning model using:
```bash
python scripts/model_training.py
```
## Input: 
Processed data (`data/interim/cleaned_file_outlier.csv`)

## Output: 
Trained model saved in the `models/` directory.

## Future Work:
- Implement machine learning models for exercise recognition.
- Tune hyperparameters to improve model accuracy.
- Explore additional feature extraction techniques for better performance.

## License:
This project is licensed under the MIT License. See the `LICENSE` file for more details.

