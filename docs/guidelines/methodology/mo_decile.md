---
layout: default
title: M&O - Decile Analysis
parent: XRAI Methodology
grand_parent: Guidelines
nav_order: 5
---

# XRAI Methodology - Model & Output (M&O) - Decile Analysis
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

In the context of machine learning, **decile analysis** can be used to evaluate the performance of a predictive model, particularly in binary classification tasks. It involves dividing the predicted probabilities or scores generated by the model into ten equal parts (deciles) and analyzing how well the model's predictions align with the actual outcomes within each decile. Decile analysis provides insights into the calibration and discrimination capabilities of the model. 

Here's an example of decile analysis in the context of a binary classification problem: suppose you have developed a machine learning model to predict whether a credit applicant will default on a loan or not. The model generates predicted probabilities for each applicant. You want to assess how well the model's predictions align with the actual outcomes 

## Step 1: Data and Predictions 
For each applicant, you have both their actual loan outcome (default or not) and the predicted probability of default generated by the model. Let's assume you have a dataset with the following information (simplified for illustration): 

| Applicant | Actual Outcome | Predicted Probability | 
|-----------|----------------|-----------------------| 
| 1         | Default        | 0.25                  | 
| 2         | Not Default    | 0.12                  | 
| ...       | ...            | ...                   | 

## Step 2: Decile Calculation  
1. Sort the dataset based on the predicted probabilities in ascending order.
   
    ```python
    # Assuming that the table above is stored in df as a pandas.DataFrame
    df = df.sort_values("Predicted Probability", ascending = False)
    ```

2. Divide the dataset into ten equal parts (deciles), with each decile containing approximately 10% of the data.

    ```python
    # Assuming that the pandas package is already imported as pd 
    df["Decile"] = pd.qcut(df["Predicted Probability"], q = 10, labels = list(range(10, 0, -1)))

    # Plot the deciles using seaborn as sns
    sns.countplot(x = "Decile", data = df, order = range(1, 11), palette = 'Blues_r')
    ```  

## Step 3: Analysis 
For each decile, calculate various metrics to evaluate the model's performance: 
- **Actual Default Rate**: Proportion of actual defaults within the decile.

```python
# Count the actual default per decile
actual_default_count = df[["Actual Outcome", "Decile"]].groupby("Decile").count("Actual Outcome").reset_index()
actual_default_pivot = actual_default_count.pivot(index='Decile', columns='Actual Outcome', values='Count')

# Calculate the proportion of actual defaults within each decile
actual_default_pivot['Proportion Default'] = actual_default_pivot['default'] / (actual_default_pivot['default'] + actual_default_pivot['not default'])

# Plot the results using Seaborn
sns.barplot(x=actual_default_pivot.index, y='Proportion Default', data=actual_default_pivot, order = range(1, 11), palette = 'Blues_r')
```

- **Predicted Default Rate**: Proportion of predicted defaults (based on a threshold) within the decile. For the code implementation, simply do the implementation from the Actual Outcome to the Predicted Outcome.

- **Average Predicted Probability**: Average predicted probability within the decile.

```python
# Average the predicted probability of the data per decile
ave_prob = df[["Predicted Probability", "Decile"]].groupby("decile").average("Predicted Probabiltity").reset_indeX()

# Plots the decile x profit graph
sns.barplot(x = "Decile", y = "Predicted Probability", data = ave_prob, order = range(1, 11), palette = 'Blues_r')
```

- **Lift**: Ratio of the actual default rate in the decile to the overall default rate in the dataset.

```python
# Use the function below to plot the lift
def plot_lift(y_real, y_proba, ax = None, color = 'b', title = 'Lift Curve'):
    # Prepare the data
    aux_df = pd.DataFrame()
    aux_df['y_real'] = y_real
    aux_df['y_proba'] = y_proba
    # Sort by predicted probability
    aux_df = aux_df.sort_values('y_proba', ascending = False)
    # Find the total positive ratio of the whole dataset
    total_positive_ratio = sum(aux_df['y_real'] == 1) / aux_df.shape[0]
    # For each line of data, get the ratio of positives of the given subset and calculate the lift
    lift_values = []
    for i in aux_df.index:
        threshold = aux_df.loc[i]['y_proba']
        subset = aux_df[aux_df['y_proba'] >= threshold]
        subset_positive_ratio = sum(subset['y_real'] == 1) / subset.shape[0]
        lift = subset_positive_ratio / total_positive_ratio
        lift_values.append(lift)
    # Plot the lift curve
    if ax == None:
        ax = plt.axes()
    ax.set_xlabel('Proportion of sample')
    ax.set_ylabel('Lift')
    ax.set_title(title)
    sns.lineplot(x = [x/len(lift_values) for x in range(len(lift_values))], y = lift_values, ax = ax, color = color)
    ax.axhline(1, color = 'gray', linestyle = 'dashed', linewidth = 3)
``` 

By analyzing these metrics across deciles, you can assess how well the model's predicted probabilities align with actual outcomes. A well-calibrated model would have predicted probabilities that closely match the actual default rates. Additionally, a model with good discrimination capabilities would exhibit higher lift in higher deciles (indicating a higher concentration of defaults in the higher-risk deciles). 

Decile analysis helps you identify whether the model is overestimating or underestimating risks and whether its predictions are consistently aligned with the actual outcomes across different risk segments. 