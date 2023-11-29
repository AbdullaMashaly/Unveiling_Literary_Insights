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
    -`Data_Import.ipynb` file utilized Spark to read in the large dataset, create a dataframe, repartition the DataFrame into smaller chunks and write them into separate files. We then read these files back in and dropped columns we did not think would be helpful (`['asin','country_code', 'book_id','description', 'isbn', 'similar_books', 'title_without_series','is_ebook', 'work_id' , 'link', 'image_url', 'url', 'edition_information', 'kindle_asin','language_code', 'publication_day', 'publication_month', 'publisher', 'title']`). 
    -Next rows with emply strings were filtered out as well as rows with empty strings in any column. 
    - We dropped duplicate ‘isbn13’. A new column was created named “to_read_count” was created after exploding the “popular_shelves” array and capturing the number of people who had the book in their “To read” cart. After the creation, we dropped the “popular_shelves” column. 
    -We then joined the original DataFrame with the aggregated DataFrame on the “isbn13” column, made the column ‘series’ binary. We then made `df_authors_exploded` DataFrame by selecting only the “authors”, and “isbn13” columns from the main df. We exploded the “author” column and created “unique_author_id” column and dropped the “author” column. We then created a CSV ("books_cleaned_sample") out of the final column. 

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

## Data Exploration

1. **Initial Exploration:**
    - Visualizations were created to explore the distribution of key features, such as average rating, number of pages, and publication year in "EDA_sample".

2. **Outlier Detection:**
    - Identified outliers in the 'num_pages' and 'publication_year' columns, allowing for a more focused analysis.

3. **Further Data Cleaning to Optimize Models:**
    - **Data Cleaning Iterations:**
    - We cleaned the data after beginning the model two additional times ("books_cleaned_sample2", and "books_cleaned_sample3").
    - We filtered out outliers within the publication_year column so that we dropped records before 1900 or after 2023
    -The `books_cleaned_sample3` was created to add in the genres df, extract the fields, and create new columns ('children', 'comics, graphic', 'fantasy, paranormal', 'fiction', 'history, historical fiction, biography', 'mystery, thriller, crime', 'non-fiction', 'poetry', 'romance', 'young-adult'). 
    - After the creation, we gave a numerical value to each column as 0 (if the book did not contain this genre keyword) or a 1 (the book did contain this genre keyword). 
    - We then joined this dataframe with the previous data frame on “book_id."


## Machine Learning Model

1. **Neural Network Model:**
    - Due to the diverse set of features and the objective of predicting a single variable from various inputs, a neural network model was implemented. 
    - The machine learning model must demonstrate meaningful predictive power of at least 75% classification accuracy or 0.80 R-squared.
    - After multiple tries, we were only able to get a R-squared value from .048 (first try) to .0536 (last try).

2. **Linear Regression Model:**
    - We then tried creating a linear regression model since the EDA showed that some of the features were linear with the average rating data. We also used an imputer in this model to account for outliers. The first try returned a .513 R-squared value with the original cleaned data. We tried changing the features used as well as including the more clean sample 3 data with no improvement in R-squared value.

3. **Random Forest Model:**

    -Since we were not able to improve on either of the first two regression models, we decided to bin our data into the following bins: `[0, 1, 2, 3, 4, 5]`, labels `[0, 1, 2, 3, 4]`, `include_lowest=True` to make the data ready for a classification model. We chose the Random Forest model because it is commonly used for classification tasks where the goal is to predict the class or category of an input based on its features.




## Repository Structure

-**/EDA:** Contains the Python scripts used for data exploration, and modeling.
-**/ML Models:** Includes the script for the different models we explored.
-**//Resources:** Holds the files with the data import and data cleaning.
    -**//panda_df:** Contains the data cleaning files.


## How to Use This Repository

1. **Clone the Repository:**
    '''bash
    git clone https://github.com/AbdullaMashaly/Project_4.git
    cd goodreads-book-reviews

2. **Explore the Code:**
    - Navigate to the '/code' directory to access Python scripts for data cleaning, exploration, and modeling.

3. **Review the Data:**
    - Explore the '/Resources' directory to find the most cleaned CSV file "books_cleaned_sample3.csv."
4. **Visualizations:**
    - Visit the '/Visualizations' directory to view visual insights generated during the exploratory data analysis.

5. Run the Code:
    - Execute the Python scripts using a Jupyter Notebook or your preferred Python environment.

## Citations
Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18. [bibtex]
Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19. [bibtex]
Wan, Mengting.(2023).goodreads.GitHub.https://github.com/MengtingWan/goodreads
Ahmad. (2023, October). Goodreads Book Reviews, Version 1. Retrieved November 22, 2023 from https://www.kaggle.com/datasets/pypiahmad/goodreads-book-reviews1

