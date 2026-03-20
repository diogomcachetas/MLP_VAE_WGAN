import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy import stats
import numpy as np
from matplotlib.ticker import FuncFormatter

def box_and_whisker(data, title, ylabel, xticklabels):
    """
    Create a box-and-whisker plot with significance bars and display statistics (max, min, std, mean).
    """
    ax = plt.axes()
    bp = ax.boxplot(data, widths=0.7, patch_artist=True)
    
    # Graph title with larger font size
    ax.set_title(title, fontsize=22)
    
    # Label y-axis with larger font size
    ax.set_ylabel(ylabel, fontsize=18)
    
    # Label x-axis ticks with larger font size
    new_xticklabels = [label.split(' ', 1)[0] + '\n' + label.split(' ', 1)[1] if ' ' in label else label for label in xticklabels]
    ax.set_xticklabels(new_xticklabels, fontsize=16)
    
    # Hide x-axis major ticks
    ax.tick_params(axis='x', which='major', length=0)
    
    # Show x-axis minor ticks
    xticks = [0.5] + [x + 0.5 for x in ax.get_xticks()]
    ax.set_xticks(xticks, minor=True)
    
    # Clean up the appearance
    ax.tick_params(axis='y', labelsize=16)
    #ax.tick_params(axis='x', which='minor', length=3, width=1)
    ax.tick_params(axis='x', which='major', length=0, pad=10) 

    # Change the colour of the boxes
    #colors = sns.color_palette('viridis')
    palette = ["#4CC9F0", "#7209B7"]
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)

    # Colour of the median lines
    plt.setp(bp['medians'], color='k')  

    '''# Calculate and annotate the statistics (max, min, std, mean)
    for i, dataset in enumerate(data):
        max_value = np.max(dataset)
        min_value = np.min(dataset)
        std_value = np.std(dataset)
        mean_value = np.mean(dataset)
        
        # Position of the annotation
        y_pos = np.max(dataset) + (np.ptp(dataset) * 0.05)  # Adjusting for clear visibility
        
        # Display the statistics near each box
        ax.text(i + 1, y_pos + 0.2, f'Max: {max_value:.2f}%\nMin: {min_value:.2f}%\nSD: {std_value:.2f}%\nMean: {mean_value:.2f}%',
                ha='center', fontsize=9, color='black')'''

    # Check for statistical significance
    significant_combinations = []
    ls = list(range(1, len(data) + 1))
    combinations = [(ls[x], ls[x + y]) for y in reversed(ls) for x in range((len(ls) - y))]
    for c in combinations:
        data1 = data[c[0] - 1]
        data2 = data[c[1] - 1]
        U, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        if p < 0.05:
            significant_combinations.append([c, p])

    # Get info about y-axis
    bottom, top = 75, 90
    yrange = 180 - bottom

    # Significance bars
    for i, significant_combination in enumerate(significant_combinations):
        x1 = significant_combination[0][0]
        x2 = significant_combination[0][1]
        level = len(significant_combinations) - i
        bar_height = (yrange * 0.08 * level) + top - 12
        bar_tips = bar_height - (yrange * 0.01)
        plt.plot([x1, x1, x2, x2], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
        
        p = significant_combination[1]
        if p < 0.001:
            sig_symbol = '***'
        elif p < 0.01:
            sig_symbol = '**'
        elif p < 0.05:
            sig_symbol = '*'
        text_height = bar_height + (yrange * 0.01)
        plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k', fontsize=16)

    # Adjust y-axis
    ax.set_ylim(75, top)

    # Annotate sample size with larger font size
    '''for i, dataset in enumerate(data):
        sample_size = len(dataset)
        ax.text(i + 1, bottom + 0.2, fr'n = {sample_size}', ha='center', size=16)'''

    # Add percentage signs to y-axis labels
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    plt.show()

# Load data from Excel file
file_path = "C:/Users/diogo/Desktop/mlp_accs.xlsx"  # Replace this with your actual file path
df = pd.read_excel(file_path)

# Convert percentage strings to float (remove '%' and convert to decimal)
columns_to_convert = df.columns[1:]  # All columns except the first 'Random_State'
df[columns_to_convert] = df[columns_to_convert].replace({',': '.'}, regex=True)
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce') * 100

# Extract the data for plotting
data = [df[column].dropna().values for column in columns_to_convert]

# Apply the box_and_whisker function
box_and_whisker(data, title='5-Fold Stratified Cross Validation', ylabel='Average Accuracy', xticklabels=columns_to_convert)


# Font size settings
title_font = 24
label_font = 20
tick_font = 18
legend_font = 18

# Load the CSV file
#file_path = "C:/Users/diogo/Desktop/final_models/mlp/mlp_average_curves_across_seeds.csv"
file_path = "C:/Users/diogo/Desktop/final_models/new/curves_across_seeds.csv"

df = pd.read_csv(file_path, sep=";")


# Plot the loss curves
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'].to_numpy(), df['Train_CrossEntropy'].to_numpy(), label='Training Loss', color='#4CC9F0', linewidth=2)
plt.plot(df['Epoch'].to_numpy(), df['Val_CrossEntropy'].to_numpy(), label='Validation Loss', color='#7209B7', linewidth=2)

#plt.plot(df['Epoch'].to_numpy(), df['Train_Accuracy'].to_numpy(), label='Training Accuracy', color='#4CC9F0', linewidth=2)
#plt.plot(df['Epoch'].to_numpy(), df['Val_Accuracy'].to_numpy(), label='Validation Accuracy', color='#7209B7', linewidth=2)


# Customize the plot
plt.ylim(0, 2.6)
#plt.ylim(0, 1.0)
#plt.title("MLP-VAE-WGAN: Average Accuracy", fontsize=title_font)
plt.xlabel("Epoch", fontsize=label_font)
plt.ylabel("Loss", fontsize=label_font)
plt.legend(loc='lower right', fontsize=legend_font)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=tick_font)
plt.yticks(fontsize=tick_font)
plt.axvline(x=404, color='gray', linestyle='--', linewidth=1.5)
plt.tight_layout()

# Show the plot
#plt.show()
plt.savefig("C:/Users/diogo/Desktop/images_paper/mlpww_ce_1.png", dpi=1000)

