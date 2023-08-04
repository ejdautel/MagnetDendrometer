#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
from pandas.tseries.offsets import DateOffset
import matplotlib.dates as mdates

def load_and_prepare_data(file_name):
    df = pd.read_csv(file_name, header=2)

    color_dict = {'Green': 0, 'Yellow': 1, 'Red': 2, 'Error':3}
    df['alignment_mapped'] = df['Alignment'].map(color_dict)
    df['time_local'] = pd.to_datetime(df['time_local'])

    device_name = df['name'].iloc[0]  # get device name from the 'name' column
    return df, device_name

def calculate_differences(df):
    df['Difference'] = df['um'].diff()
    df['Absolute Difference'] = df['Difference'].abs()
    
    return df

def calculate_and_print_stats(df):
    mean_difference = df['Difference'].mean()
    std_dev_difference = df['Difference'].std()

    print(df['Difference'].describe())

    # quartiles and IQR
    Q1 = df['Difference'].quantile(0.25)
    Q3 = df['Difference'].quantile(0.75)
    IQR = Q3 - Q1

    # range for outliers
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR

    # Count of the outliers
    outliers = df[(df['Difference'] < lower_bound) | (df['Difference'] > upper_bound)]

    # fraction of outliers
    fraction_of_outliers = len(outliers) / len(df)

    print(f"Fraction of outliers in original box plot: {fraction_of_outliers}, where number of outliers is {len(outliers)} and total number of values is {len(df)}")

    # Total number of 'Difference' values
    total_values = df['Difference'].count()

    # Number of zero 'Difference' values
    zero_values = (df['Difference'] == 0).sum()

    # Number of outliers
    outliers = ((df['Difference'] - df['Difference'].mean()).abs() > 3*df['Difference'].std()).sum()

    # Ratios
    ratio_zeros = zero_values / total_values
    ratio_outliers = outliers / total_values

    print(f'The number of outliers is: {outliers}')
    print(f'Ratio of zero differences: {ratio_zeros:.2f}')
    print(f'Ratio of outliers: {ratio_outliers:.2f}')
 


def plot_displacement(df, device_name):
    min_date = df['time_local'].min().strftime('%Y-%m-%d')
    max_date = df['time_local'].max().strftime('%Y-%m-%d')
    instance_name = df['instance'].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 10))
    # plot displacement
    ax.plot(df['time_local'], df['um'], label='Displacement')

    # plot alignment
    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(), where=(df['alignment_mapped'] == 0),
                    color='green', alpha=0.5, label='Green Alignment')
    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(), where=(df['alignment_mapped'] == 1),
                    color='yellow', alpha=1, label='Yellow Alignment')
    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(), where=(df['alignment_mapped'] == 2),
                    color='red', alpha=1, label='Red Alignment')
    ax.fill_between(df['time_local'], df['um'].max(), df['um'].min(), where=(df['alignment_mapped'] == 3),
                    color='purple', alpha=1, label='Error Alignment')
    # format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))  # tick every week
    plt.gcf().autofmt_xdate()  # rotate the x labels
    ax.set_title(f'{device_name}: {instance_name} from {min_date} to {max_date}')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_histogram(df, device_name):
    instance_name = df['instance'].iloc[0]
    min_date = df['time_local'].min().strftime('%Y-%m-%d')
    max_date = df['time_local'].max().strftime('%Y-%m-%d')

    plt.figure(figsize=(15, 6))
    sns.histplot(df['Difference'].dropna(), bins=50)
    plt.title(f'{device_name}: {instance_name}. Data from {min_date} to {max_date}')
    plt.show()
    
    plt.figure(figsize=(15, 6))
    sns.boxplot(x=df['Difference'])
    plt.title(f'{device_name}:{instance_name}. Data from {min_date} to {max_date}')
    plt.show()

def outliers_analysis(df):
    Q1 = df['Difference'].quantile(0.25)
    Q3 = df['Difference'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    outliers = df[(df['Difference'] < lower_bound) | (df['Difference'] > upper_bound)]
    fraction_of_outliers = len(outliers) / len(df)
    print(f"Fraction of outliers: {fraction_of_outliers}, where number of outliers is {len(outliers)} and total number of values is {len(df)}")

def plot_cleaned_difference(df, device_name):
    mean_difference = df['Difference'].mean()
    std_dev_difference = df['Difference'].std()
    df = df[np.abs(df['Difference'] - mean_difference) <= 2*std_dev_difference]
    instance_name = df['instance'].iloc[0]
    min_date = df['time_local'].min().strftime('%Y-%m-%d')
    max_date = df['time_local'].max().strftime('%Y-%m-%d')
    
    plt.figure(figsize=(15, 6))
    sns.histplot(df['Difference'].dropna(), bins=20)
    plt.title(f'{device_name}: {instance_name}. Data from {min_date} to {max_date} (excluding outliers)')
    plt.show()
   
    plt.figure(figsize=(15, 6))
    sns.boxplot(x=df['Difference'])
    plt.title(f'{device_name}: {instance_name}. Data from {min_date} to {max_date} (excluding outliers)')
    plt.show()

def identify_and_adjust_outliers(df):
    # Calculate the differences and identify the outliers
    df['Difference'] = df['um'].diff()
    Q1 = df['Difference'].quantile(0.25)
    Q3 = df['Difference'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_indices = df[(df['Difference'] < Q1 - 3.0 * IQR) | (df['Difference'] > Q3 + 3.0 * IQR)].index

    # new column for the adjustment values
    df['Adjustment'] = 0.0

    # Iterate over the outlier indices
    for idx in outlier_indices:
        if idx != 0:  # we can't do this for the first index
            # The adjustment is the difference at the current index
            adjustment = df.loc[idx, 'Difference']

            # Subtract the adjustment from the current and all following displacement values
            df.loc[idx:, 'Adjustment'] += adjustment

    # The cleaned displacement values are the original displacement minus the adjustment
    df['Cleaned Displacement'] = df['um'] - df['Adjustment']

    return df
   

'''def identify_and_adjust_outliers(df):
    # Calculate the differences and identify the outliers
    df['Difference'] = df['um'].diff()
    mean_difference = df['Difference'].mean()
    std_dev_difference = df['Difference'].std()
    outlier_indices = df[np.abs(df['Difference'] - mean_difference) > 2 * std_dev_difference].index

    # new column for the adjustment values 
    df['Adjustment'] = 0.0

    # Iterate over the outlier indices
    for idx in outlier_indices:
        if idx != 0:  # we can't do this for the first index
            # The adjustment is the difference at the current index
            adjustment = df.loc[idx, 'Difference']
            
            # Subtract the adjustment from the current and all following displacement values
            df.loc[idx:, 'Adjustment'] += adjustment

    # The cleaned displacement values are the original displacement minus the adjustment
    df['Cleaned Displacement'] = df['um'] - df['Adjustment']
    
    return df'''


def plot_cleaned_displacement(df, device_name):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].plot(df['time_local'], df['um'])
    ax[0].set_title('Original Displacement')
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Displacement')

    ax[1].plot(df['time_local'], df['Cleaned Displacement'])
    ax[1].set_title('Cleaned Displacement')
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Displacement')

    plt.tight_layout()
    plt.show()


    
'''def main():
    df, device_name = load_and_prepare_data('BB2.csv')  # also get device_name
    df = calculate_differences(df)
    
    calculate_and_print_stats(df)

    plot_displacement(df, device_name)
    plot_histogram(df, device_name)
    outliers_analysis(df)
    plot_cleaned_difference(df, device_name)
    identify_and_adjust_outliers(df)
    plot_cleaned_displacement(df, device_name)
    

if __name__ == "__main__":
    main()
'''


# In[2]:


df, device_name = load_and_prepare_data('H1.csv')  # also get device_name
df = calculate_differences(df)
    
calculate_and_print_stats(df)



# In[3]:


plot_displacement(df, device_name)


# In[4]:


plot_histogram(df, device_name)
    


# In[5]:


outliers_analysis(df)
plot_cleaned_difference(df, device_name)


# In[6]:


identify_and_adjust_outliers(df)
plot_cleaned_displacement(df, device_name)


# In[ ]:




