import pandas as pd
from scipy import stats

# Load the data
df = pd.read_csv('results.csv')

# Extract the relevant columns
with_gestures = df['How engaged were you during scenes that included Pepper’s arm gestures?  (1: Not engaged → 5: Highly engaged)']
without_gestures = df['How engaged were you during scenes without Pepper’s arm gestures?  (1: Not engaged → 5: Highly engaged)']

# Remove any rows with missing values
clean_data = pd.DataFrame({
    'with_gestures': with_gestures,
    'without_gestures': without_gestures
}).dropna()

# Calculate differences
differences = clean_data['with_gestures'] - clean_data['without_gestures']

# Shapiro-Wilk test for normality
shapiro_stat, shapiro_p = stats.shapiro(differences)
print(f"\nShapiro-Wilk test for normality of differences: p = {shapiro_p:.4f}")
if shapiro_p < 0.05:
    print("Warning: Differences may not be normally distributed. Consider Wilcoxon signed-rank test.")

# Perform paired t-test (one-tailed since we're testing "higher" engagement)
t_stat, p_value = stats.ttest_rel(
    clean_data['with_gestures'],
    clean_data['without_gestures'],
    alternative='greater'  # One-tailed test for "greater than"
)

# Calculate descriptive statistics
mean_with = clean_data['with_gestures'].mean()
mean_without = clean_data['without_gestures'].mean()
std_with = clean_data['with_gestures'].std()
std_without = clean_data['without_gestures'].std()
n = len(clean_data)

# Print results
print(f"\nPaired t-test results (one-tailed):")
print(f"Sample size: {n}")
print(f"Mean engagement with gestures: {mean_with:.2f} (SD = {std_with:.2f})")
print(f"Mean engagement without gestures: {mean_without:.2f} (SD = {std_without:.2f})")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject the null hypothesis - engagement is significantly higher with arm gestures")
else:
    print("\nConclusion: Fail to reject the null hypothesis - no significant difference in engagement")

# Perform Wilcoxon signed-rank test (non-parametric)
wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
    clean_data['with_gestures'],
    clean_data['without_gestures'],
    alternative='greater'
)
print(f"\nWilcoxon signed-rank test (one-tailed): p = {wilcoxon_p:.4f}")
    