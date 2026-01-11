# IIT-KANPUR-ML-PROJECT-PREDICT-SMOKERS-

<h1>Predicting Smokers and non-smokers via Vital Signs ( Machine Learning Project)</h1>

<p>
This is a <strong>machine learning project</strong> focused on predicting whether a person is a smoker or a non-smoker based on provided biological and health-related factors.
The project demonstrates end-to-end ownership of a supervised ML pipeline, from data preprocessing and visualization to model evaluation and selection.
</p>

<hr>

<h2>Dataset Overview</h2>
<ul>
  <li><strong>Total Rows:</strong> 55,692</li>
  <li><strong>Total Columns:</strong> 27</li>
</ul>

<p>
The dataset was provided by the institution and consists of structured demographic and biological features representing individual health attributes.
</p>

<hr>

<h2>Approach Used</h2>
<p>
The overall machine learning workflow followed these major steps:
</p>
<ol>
  <li>Data collection</li>
  <li>Data preprocessing and visualization</li>
  <li>Model comparison and selection</li>
  <li>Model training</li>
  <li>Model testing and evaluation</li>
</ol>

<hr>

<h2>1. Data Collection</h2>
<p>
The dataset was provided by the institution itself. As mentioned above, the dataset size is <strong>55,692 × 27</strong>.
No external data sources were used.
</p>

<hr>

<h2>2. Data Preprocessing and Visualization</h2>

<p>
Data preprocessing was performed to improve data quality and model performance. 
Irrelevant and non-contributing features were identified and removed to reduce noise in the dataset.
</p>

<p>
Two separate DataFrames were created:
</p>
<ul>
  <li><strong>df1</strong> for input features (<code>X_train</code>)</li>
  <li><strong>df2</strong> for target labels (<code>Y_train</code>)</li>
</ul>

<p>
The <strong>ID</strong> column was set as the index for both DataFrames, as it uniquely represented individuals but did not contribute to prediction.
</p>

<p>
Non-numerical categorical features were converted into numerical form (0/1 encoding) to make them suitable for machine learning models.
</p>

<p>
The categorical features encoded include:
</p>
<ul>
  <li>Gender</li>
  <li>Tartar</li>
  <li>Oral</li>
</ul>

<p>
The distribution of smokers and non-smokers was visualized using a bar graph to analyze class balance.
</p>

<p>
The feature DataFrame (<code>df1</code>) and target DataFrame (<code>df2</code>) were then merged into a single DataFrame (<code>df</code>) to perform correlation analysis and understand relationships between features and the target variable.
</p>

<hr>

<h2>3. Model Selection and Evaluation</h2>

<p>
Multiple supervised machine learning models were trained and evaluated to determine the best-performing classifier:
</p>

<ul>
  <li>Decision Tree Classifier</li>
  <li>Gaussian Naive Bayes</li>
  <li>K-Nearest Neighbors (with feature scaling)</li>
  <li>Random Forest Classifier</li>
</ul>

<p>
All models were evaluated using <strong>5-fold cross-validation</strong> to ensure reliable and unbiased performance comparison.
</p>

<p>
The <strong>Random Forest Classifier</strong> achieved the best performance with an average accuracy of approximately <strong>75.5%</strong>, outperforming the other models.
</p>

<hr>

<h2>4. Model Training and Testing</h2>

<p>
The Random Forest model was trained on the full training dataset and tested on unseen test data.
Model performance was evaluated using:
</p>

<ul>
  <li>Accuracy score</li>
  <li>Confusion matrix</li>
  <li>Classification report</li>
</ul>

<p>
Feature importance analysis was also performed to understand which biological factors contributed most to the prediction.
</p>

<hr>

<h2>5. Model Persistence</h2>

<p>
The final trained Random Forest model was serialized using <code>pickle</code> to allow future reuse, testing, or deployment without retraining.
</p>

<hr>

<h2>Technologies Used</h2>
<ul>
  <li>Python</li>
  <li>pandas</li>
  <li>NumPy</li>
  <li>scikit-learn</li>
  <li>Matplotlib</li>
</ul>

<hr>

<h2>Conclusion</h2>
<p>
This project demonstrates practical application of supervised machine learning techniques on a large real-world dataset, with emphasis on data preprocessing, model comparison, and evaluation.
Being an unguided project, it reflects independent problem-solving and end-to-end ML pipeline development.
</p>

