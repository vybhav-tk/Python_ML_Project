# Loan Default Prediction using Machine Learning

Predicting whether a borrower will repay their loan or default is a critical task for financial institutions. This project utilizes machine learning techniques to predict loan repayment outcomes using historical loan data from LendingClub.

### Objective
The goal is to build a predictive model that can determine the likelihood of a borrower repaying their loan based on historical data. This model can assist lenders in making informed decisions about approving or rejecting loan applications.

### Dataset Description

The Dataset used here is a subset of the LendingClub DataSet obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LoanStatNew</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>term</td>
      <td>The number of payments on the loan. Values are in months and can be either 36 or 60.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>int_rate</td>
      <td>Interest Rate on the loan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>installment</td>
      <td>The monthly payment owed by the borrower if the loan originates.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>grade</td>
      <td>LC assigned loan grade</td>
    </tr>
    <tr>
      <th>5</th>
      <td>sub_grade</td>
      <td>LC assigned loan subgrade</td>
    </tr>
    <tr>
      <th>6</th>
      <td>emp_title</td>
      <td>The job title supplied by the Borrower when applying for the loan.*</td>
    </tr>
    <tr>
      <th>7</th>
      <td>emp_length</td>
      <td>Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.</td>
    </tr>
    <tr>
      <th>8</th>
      <td>home_ownership</td>
      <td>The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER</td>
    </tr>
    <tr>
      <th>9</th>
      <td>annual_inc</td>
      <td>The self-reported annual income provided by the borrower during registration.</td>
    </tr>
    <tr>
      <th>10</th>
      <td>verification_status</td>
      <td>Indicates if income was verified by LC, not verified, or if the income source was verified</td>
    </tr>
    <tr>
      <th>11</th>
      <td>issue_d</td>
      <td>The month which the loan was funded</td>
    </tr>
    <tr>
      <th>12</th>
      <td>loan_status</td>
      <td>Current status of the loan</td>
    </tr>
    <tr>
      <th>13</th>
      <td>purpose</td>
      <td>A category provided by the borrower for the loan request.</td>
    </tr>
    <tr>
      <th>14</th>
      <td>title</td>
      <td>The loan title provided by the borrower</td>
    </tr>
    <tr>
      <th>15</th>
      <td>zip_code</td>
      <td>The first 3 numbers of the zip code provided by the borrower in the loan application.</td>
    </tr>
    <tr>
      <th>16</th>
      <td>addr_state</td>
      <td>The state provided by the borrower in the loan application</td>
    </tr>
    <tr>
      <th>17</th>
      <td>dti</td>
      <td>A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.</td>
    </tr>
    <tr>
      <th>18</th>
      <td>earliest_cr_line</td>
      <td>The month the borrower's earliest reported credit line was opened</td>
    </tr>
    <tr>
      <th>19</th>
      <td>open_acc</td>
      <td>The number of open credit lines in the borrower's credit file.</td>
    </tr>
    <tr>
      <th>20</th>
      <td>pub_rec</td>
      <td>Number of derogatory public records</td>
    </tr>
    <tr>
      <th>21</th>
      <td>revol_bal</td>
      <td>Total credit revolving balance</td>
    </tr>
    <tr>
      <th>22</th>
      <td>revol_util</td>
      <td>Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.</td>
    </tr>
    <tr>
      <th>23</th>
      <td>total_acc</td>
      <td>The total number of credit lines currently in the borrower's credit file</td>
    </tr>
    <tr>
      <th>24</th>
      <td>initial_list_status</td>
      <td>The initial listing status of the loan. Possible values are – W, F</td>
    </tr>
    <tr>
      <th>25</th>
      <td>application_type</td>
      <td>Indicates whether the loan is an individual application or a joint application with two co-borrowers</td>
    </tr>
    <tr>
      <th>26</th>
      <td>mort_acc</td>
      <td>Number of mortgage accounts.</td>
    </tr>
    <tr>
      <th>27</th>
      <td>pub_rec_bankruptcies</td>
      <td>Number of public record bankruptcies</td>
    </tr>
  </tbody>
</table>

----

### Exploratory Data Analysis
#### Target Variable Distribution
The target variable loan_status indicates whether a loan is "Fully Paid" or "Charged Off" (defaulted). A count plot was used to visualize the distribution of loan_status.
- Observation: The dataset contains more loans that are fully paid than those that are charged off, indicating an imbalance that may need to be addressed during modeling.

#### Loan Amount Distribution
Analyzing the distribution of loan amounts helps in understanding the range and common values of loans issued.
- Observation: Loan amounts vary widely, with a concentration around certain amounts, suggesting standard loan offerings or common borrower needs.

#### Correlation Analysis
Examining correlations between numerical features using a heatmap helps identify relationships that can inform feature selection and engineering.
- Observation: There is a high correlation (95%) between installment and loan_amnt, indicating that as loan amount increases, the installment payments also increase proportionally.

#### Feature Relationships
- Installment vs. Loan Amount: A scatter plot showed a strong positive linear relationship between installment amounts and loan amounts.
- Loan Status vs. Loan Amount: A box plot indicated that loans which were charged off tend to have slightly higher loan amounts compared to those that were fully paid.
- Grade and Subgrade Analysis: Count plots revealed that loans with lower grades (indicating higher risk) have higher rates of default, confirming the reliability of the grading system used by LendingClub.


### Data Preprocessing

#### Handling Missing Values
- Empirical Analysis of Missing Data: Calculated the percentage of missing values for each column to identify features with significant missing data.
- Columns with Significant Missing Values:
  - emp_title: Missing values due to diverse employment titles.
  - emp_length: Missing values possibly due to incomplete employment history data.
  - title: Redundant information with the purpose column.
  - mort_acc: Missing values that could be related to borrowers without mortgage accounts.
- Dropping Irrelevant Columns
  - emp_title: Dropped due to the high number of unique values making it impractical for effective encoding.
  - title: Dropped because it provides redundant information already captured in the purpose column.
- Filling Missing Values
  - mort_acc: Since mort_acc is correlated with total_acc, missing values were filled using the average number of mortgage accounts for borrowers with the same total_acc. This method leverages the relationship between total accounts and mortgage accounts to estimate missing values.
  - Remaining Missing Values: The remaining missing data accounted for less than 0.1% of the dataset. These rows were dropped to maintain data integrity without significantly impacting the dataset size.

#### Feature Engineering

Encoding Categorical Variables
- Term: Converted the term column from string format to integer by extracting the numeric value, simplifying the model's ability to interpret the loan term length.
- Grade and Subgrade:
  - Dropped the grade column as it is directly related to sub_grade and would introduce redundancy.
  - Applied one-hot encoding to the sub_grade column, creating dummy variables for each subgrade to allow the model to learn from the ordinal nature of loan grades.
- Home Ownership:
  - Combined rare categories (NONE and ANY) into a single category OTHER to reduce sparsity.
  - One-hot encoded the home_ownership column to numerically represent home ownership status.
- Other Categorical Features: One-hot encoded verification_status, application_type, initial_list_status, and purpose to convert categorical text data into numerical format suitable for modeling.
- Issue Date: Dropped the issue_d column to prevent data leakage, as this feature would not be available at the time of loan application and could artificially inflate model performance.
- Earliest Credit Line: Extracted the year from the earliest_cr_line column to represent the length of the borrower's credit history numerically, providing insight into creditworthiness.
- Address:
  - Extracted the zip code from the address column to include geographical information.
  - One-hot encoded the zip codes to capture any regional effects on loan repayment.
- Feature Scaling: Applied Min-Max scaling to normalize the numerical features, ensuring they all fall within the same range. This is essential for algorithms like Random Forest that are sensitive to feature scaling, as it prevents features with larger scales from dominating the model.


### Model Building

#### Data Splitting

- Features and Labels:
  - Separated the dataset into features (X) by dropping the target variable loan_repaid.
  - The target variable (y) was set as the loan_repaid column, which indicates whether the loan was repaid or not.
- Train-Test Split: Split the data into training and testing sets using an 80/20 split. This allows the model to be trained on the majority of the data while reserving a portion for unbiased evaluation.

#### Random Forest Classifier

- Model Initialization and Training:
  - Initialized a Random Forest Classifier with 350 decision trees (n_estimators=350) to build a robust model through ensemble learning.
  - Trained the model using the training data, allowing it to learn patterns associated with loan repayment.

- Predictions: Generated predictions on the test data using the trained model to assess its performance on unseen data.

### Model Evaluation

- Classification Report:
  - Produced a classification report to evaluate the model's precision, recall, F1-score, and support for each class (loan repaid or defaulted).
  - Precision: Indicates the accuracy of positive predictions.
  - Recall: Measures the model's ability to identify all positive instances.
  - F1-Score: The harmonic mean of precision and recall, providing a balance between the two.

- Confusion Matrix:
  - Constructed a confusion matrix to visualize true positives, true negatives, false positives, and false negatives.
  - This helps in understanding the types of errors the model is making.

- Metrics Interpretation:
  - Accuracy: The overall percentage of correct predictions.
  - High Precision and Recall: Essential for financial applications where false positives (approving a risky loan) and false negatives (rejecting a safe loan) have significant consequences.


The Random Forest Classifier demonstrated satisfactory performance in predicting loan repayment. Key takeaways include:
- Effective Feature Engineering: Encoding categorical variables and scaling numerical features contributed to the model's performance.
- Handling Missing Data: Strategic imputation and dropping of irrelevant columns improved data quality.

Recommendations for Improvement:

- Hyperparameter Tuning: Adjusting parameters like tree depth, minimum samples split, and others could enhance model performance.
- Algorithm Exploration: Trying different algorithms such as Gradient Boosting Machines or XGBoost may capture complex nonlinear relationships better.
- Addressing Class Imbalance: If the dataset is imbalanced, techniques like SMOTE (Synthetic Minority Over-sampling Technique) or class weighting can be applied.
- Feature Selection: Further analysis to identify and select the most impactful features can reduce model complexity and overfitting.
- Cross-Validation: Implementing k-fold cross-validation to ensure the model's robustness across different subsets of data.
