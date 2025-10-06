# Myers-Briggs-Personality-Prediction

This project predicts a person’s Myers-Briggs Type Indicator (MBTI) personality type based on the text they write using machine learning.

Features
1. Cleans and preprocesses raw text data using BeautifulSoup for HTML tag removal and regex for link and symbol cleaning
2. Explores dataset distribution with Seaborn and Matplotlib visualizations of MBTI personality type frequencies
3. Converts text data into numerical features using TF-IDF Vectorization and CountVectorizer
4. Applies TruncatedSVD for dimensionality reduction and visualization of latent text patterns
5. Trains multiple machine learning models like ExtraTreesClassifier, Multinomial Naive Bayes, and Logistic Regression for MBTI personality classification
6. Evaluates models using Stratified 5-Fold Cross Validation with metrics: Accuracy, F1-score, and Log Loss
7. Tunes Logistic Regression (balanced class weights, regularization) achieving >65% accuracy on validation folds
8. Supports real-time personality prediction from user-input text via trained TF-IDF and Logistic Regression pipeline
