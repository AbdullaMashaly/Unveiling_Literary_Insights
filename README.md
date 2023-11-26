# Goodreads Book Reviews Dataset Exploration

## Team Members
- Abdulla Mashaly [@AbdullaMashaly](https://github.com/AbdullaMashaly)
- Jeremy Magee [@JpMageeGitHub](https://github.com/JpMageeGitHub)
- Akhil Karandikar [@KarandikarA](https://github.com/KarandikarA)
- Lori Girton [@lorigirton](https://github.com/lorigirton)

## Overview

This GitHub repository contains the code and process documentation for the exploration and analysis of the "Goodreads Book Reviews" dataset sourced from Kaggle. The dataset provides comprehenive information about book reviews including details about authors, ratings, formats, and more. Our team used this data to create a machine learning model that could predict book review ratings.

## Dataset Source
The original dataset can be found on Kaggle: [Goodreads Book Reviews](https://www.kaggle.com/datasets/pypiahmad/goodreads-book-reviews1).

## Data Cleaning Process

1. **Data Loading with Spark:**
    - The dataset was initially loaded using Spark to efficiently handle the large file. 

2. **Column Selection:**
    - We narrowed down the dataset by keeping only the essential columns, including authors, average rating, country code, format, ebook information, ISBN13, number of pages, popular shelves, publication year, ratings count, series information, text reviews count, title, and author ID.

3. **Data Filtering**
    - Formats were further filtered to include only paperback, ebooks, and hardcover.
    - Duplicate records with the same ISBN13 were dropped.
    - Records with null values were removed to ensure data quality.

4. **Series Encoding:**
    - A binary column, 'series', was created to indicate whether a book is part of a series (1) or not (0).

5. **Pandas DataFrame and CSV Export:**
    - The cleaned data was converted to a Pandas DataFrame, making it more manageable. 
    - The final cleaned dataset was exported to a CSV file and shared with the team.

## Data Visualization

1. **Initial Exploration:**
    - Visualizations were created to explore the distribution of key features, such as average rating, number of pages, and publication year.

2. **Outlier Detection:**
    - Identified outliers in the 'num_pages' and 'publication_year' columns, allowing for a more focused analysis.

## Machine Learning Model

1. **Neural Network Model:**
    - Due to the diverse set of features and the objective of predicting a single variable from various inputs, a neural network model was implemented.

## Repository Structure

-**/Code:** Contains the Python scripts used for data cleaning, exploration, and modeling.
-**/Resources:** Includes the original dataset and the cleaned CSV file.
-**//Visualizations:** Holds visualizations created during the exploratory data analysis.

## How to Use This Repository

1. **Clone the Repository:**
    '''bash
    git clone https://github.com/AbdullaMashaly/Project_4.git
    cd goodreads-book-reviews

2. **Explore the Code:**
    - Navigate to the '/code' directory to access Python scripts for data cleaning, exploration, and modeling.

3. **Review the Data:**
    - Explore the '/Resources' directory to find the cleaned CSV file.
4. **Visualizations:**
    - Visit the '/Visualizations' directory to view visual insights generated during the exploratory data analysis.

5. Run the Code:
    - Execute the Python scripts using a Jupyter Notebook or your preferred Python environment.

## Credits


