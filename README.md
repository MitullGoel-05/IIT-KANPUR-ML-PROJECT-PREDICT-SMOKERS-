# IIT-KANPUR-ML-PROJECT-PREDICT-SMOKERS-

<h2>DESCRIPTION</h2>
<p>This is an unguided project on predicting a person as a smoker or non-smoker based on the biological factors provided as the data</p>
<br>
<p>The size of the dataset for the ML model is: </p>
<ul>
    <li> Total Rows: 55692 </li>
    <li> Total Columns: 27 </li>
</ul>

<br>
<h2>APPROACH USED: </h2>

<p>The basic steps involved are: </p>

<ol>
    <li> Collecting the data (provided by the institution) </li>
    <li> Pre-processing and visualising the data </li>
    <li> Checking out the best model </li>
    <li> Training the data </li>
    <li> Testing the data </li>
    
</ol>

<br>
<h2>(1.) DATA COLLECTION</h2>
<p>The data file was provided by the institution itself. As mentioned above the dataset is of size (55692 x 27). </p>

<h2>(2.) PRE-PROCESSING AND DATA VISUALISATION</h2>
<p>In the process of data pre-processing, the data was cleaned by reducing the features and removing the useless data (which was not contributing in the prediction).</p>
<p>DataFrames were created 'df1' and 'df2' for 'X_train' and 'Y-train' datasets respectively.</p>
<p>The indexes were set as IDs for the df1 and df2 dataframes, as ID were not playing any role for predicting but were representing different people as it is a unique entity.</p>
<p>Non-numerical data was converted into numerical data (0/1) for easy training and testing of the model.</p>
<p>The features which were converted into numerical data are: </p>
<ol>
    <li>Gender</li>
    <li>Tartar</li>
    <li>Oral</li>
</ol>
<p>Then, visualisation of the distribution of smokers and non-smokers is shown in the form of bar graph. </p>
<p>The two datasets 'df1' and 'df2', are joined together as 'df' for the correlation graph.</p>
