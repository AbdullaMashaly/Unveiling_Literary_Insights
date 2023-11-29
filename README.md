# Unveiling Literacy Insights: Harnessing The Power of Data for Predictive Modeling of Book Ratings

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

1. **Data Loading with Spark:** ![Alt Text](https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/PySpark.png)
    - The dataset was initially loaded using Spark to efficiently handle the large file. 
    -`Data_Import.ipynb` file utilized Spark to read in the large dataset, create a dataframe, repartition the DataFrame into smaller chunks and write them into separate files. We then read these files back in and dropped columns we did not think would be helpful (`['asin','country_code', 'book_id','description', 'isbn', 'similar_books', 'title_without_series','is_ebook', 'work_id' , 'link', 'image_url', 'url', 'edition_information', 'kindle_asin','language_code', 'publication_day', 'publication_month', 'publisher', 'title']`) and were left with the following columns:
    ![Alt Text](https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/Column%20Names.png)
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
    - Visualizations were created to explore the distribution of key features, such as average rating, number of pages, and publication year in "EDA_sample". There were several records with year published after 2023 (scatter plot 1). After taking these out, there were still other published years too far back (scatter plot 2). We also noticed the column with number of pages had over half the records with 0 pages (bar graph 1). Identifying these outliers in the 'num_pages' and 'publication_year' columns, allowed for a more focused analysis.

    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/EDA1a.png" alt="scatter plot 1" width="200" height="200">
    </p>

    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/EDA1b.png" alt="scatter plot 2" width="200" height="200">
    </p>

    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/EDA1c.png" alt="bar graph 1" width="200" height="200">
    </p>


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
    - After multiple tries, we were only able to get a R-squared value from .048 (first try) to .0536 (last try). The visuals for distribution 1 shows the distribution of the actual vs. predicted ratings and distribution 2 shows how close the predictions were to the actual ratings. Distribution 3 is another way of looking at the actual versus predicted ratings in our Neural Network model.
    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/lori/Visuals/NNetwork1.png" alt="distribution 1" width="200" height="200">
    </p>
    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/lori/Visuals/NNetwork2.png" alt="distribution 2" width="200" height="200">
    </p>
    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/lori/Visuals/NNetwork3.png" alt="distribution 3" width="200" height="200">
    </p>


2. **Linear Regression Model:**
    - We noticed in the initial EDA that there was a linear relationship between average rating and author average rating (scatter plot 3) and decided to look at other features (scatter plots 4 & 5) which may have linear relationships to try a Linear Regression Model.  We also used an imputer in this model to account for outliers. The first try returned a .513 R-squared value with the original cleaned data. We tried changing the features used as well as including the more clean sample 3 data with no improvement in R-squared value. This model performed with a MSE 0.12 and R-Squared value of 0.52 (image 1). The second model where we plugged in a cleaner dataset provided a less reliable model with a R-Squared value of 0.48 (image 2). We believe this could be due to the imputer being used to plug in the mean for data with outliers and once those outliers were removed, it was less predictive.
   <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/EDA3a.png" alt="scatter plot 3" width="200" height="200">
    </p>
    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/EDA3b.png" alt="scatter plot 4" width="200" height="200">
    </p>
    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/EDA3c.png" alt="scatter plot 5" width="200" height="200">
    </p>
    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/LR1.png" alt="image 1" width="200" height="200">
    </p>
    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/LR2.png" alt="image 2" width="200" height="200">
    </p>
    

3. **Random Forest Model:**

    -Since we were not able to improve on either of the first two regression models, we decided to bin our data into the following bins: `[0, 1, 2, 3, 4, 5]`, labels `[0, 1, 2, 3, 4]`, `include_lowest=True` to make the data ready for a classification model. We chose the Random Forest model because it is commonly used for classification tasks where the goal is to predict the class or category of an input based on its features.

    <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/RF1.png" alt="image 3" width="200" height="200">
    </p>
   <p align="center">
        <img src="https://github.com/AbdullaMashaly/Project_4/blob/main/Visuals/RF2.png" alt="image 4" width="200" height="200">
    </p>



## Repository Structure

-**/EDA:** Contains the Python scripts used for data exploration.
-**/ML Models:** Includes the script for the different models we explored.
    -**Notebooks:** For Linear Regression Model 1 & 2, Random Forest and the original Neural Network Tuner
    -**NN_Tuner_Trial2:** Includes the second data cleaning files and the second Neural Network Tuner
    -**NN_Tuner_Whole_Data:** Include the files for the final data cleaning and the final Neural Network Tuner
-**//Resources:** Holds the csv files created after data cleaning.
-**//Visuals:** Contains the png files used in Presentation and README.
-**No folder:** Contains, gitignor, data preprocessing, and README files.


## Citations
Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18. [bibtex]
Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19. [bibtex]
Wan, Mengting.(2023).goodreads.GitHub.https://github.com/MengtingWan/goodreads
Ahmad. (2023, October). Goodreads Book Reviews, Version 1. Retrieved November 22, 2023 from https://www.kaggle.com/datasets/pypiahmad/goodreads-book-reviews1

