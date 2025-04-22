## "🏏📊 ODI Cricket Head-to-Head Analysis (1987–2023) 🔍" ##
#  🐍 ─────────────────────────────────────── ⋆⋅☆⋅⋆ ─────────────────────────────────────── 🐍 #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, ttest_1samp
import warnings
warnings.filterwarnings("ignore")

# ╰┈➤ 1. Load and Clean Data
df = pd.read_csv("odidata.csv")

# Rename unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df.rename(columns={'Unnamed: 0': 'match_id'}, inplace=True)

# Add winner column
df['winner'] = np.where(df['winning_team'] == 1, df['team1'], df['team2'])

# ╰┈➤ 2. Basic Dataset Info
print("\n📋 Dataset Info:\n")
print(df.info())
print("\n🔍 First 5 Rows:\n", df.head())


#🗐 Check for NULL and Unique values:
print("\n🔧 Null Values:\n", df.isna().sum())
print("\n🧬 Unique Winners:", df['winner'].nunique())



# ╰┈➤ 3. Total Wins Per Team
team_wins = df['winner'].value_counts().reset_index()
team_wins.columns = ['Team', 'Total Wins']

# Bar Chart: Top 10 teams
top10 = team_wins.head(10)
plt.figure(figsize=(10, 6))
sns.barplot(data=top10, x="Team", y="Total Wins", palette="crest")
plt.title("🏏 Top 10 ODI Teams by Wins (1987–2023)")
plt.xlabel("Team")
plt.ylabel("Total Wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie Chart: Share of Wins (Top 8)
top8 = team_wins.head(8)
plt.figure(figsize=(8, 8))
plt.pie(top8["Total Wins"], labels=top8["Team"], autopct="%1.1f%%", startangle=140, colors=sns.color_palette("Set3"))
plt.title("🥧 Top 8 Teams' Share of Total ODI Wins")
plt.axis("equal")
plt.tight_layout()
plt.show()
# ╰┈➤ 4. Head-to-Head Matrix
head_to_head = df.groupby(['winner', 'team1']).size().unstack(fill_value=0) + \
               df.groupby(['winner', 'team2']).size().unstack(fill_value=0)
head_to_head = head_to_head.fillna(0).astype(int)
plt.figure(figsize=(10, 7))
sns.heatmap(head_to_head, cmap="YlGnBu", linewidths=0.5)
plt.title("🔥 Head-to-Head Wins Heatmap")
plt.xlabel("Defeated Team")
plt.ylabel("Winning Team")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ╰┈➤ 5. Yearly Wins for Top 5 Teams
top5 = top10['Team'].tolist()
yearly = df[df['winner'].isin(top5)].groupby(['year', 'winner']).size().unstack().fillna(0)

plt.figure(figsize=(10, 6))
yearly.plot(marker='o')
plt.title("📈 Yearly ODI Wins for Top 5 Teams")
plt.xlabel("Year")
plt.ylabel("Number of Wins")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ╰┈➤ 6. Count of Teams by Win Ranges
win_bins = pd.cut(team_wins["Total Wins"], bins=[0, 50, 100, 200, 300, 500, 1000])
plt.figure(figsize=(10, 5))
sns.countplot(x=win_bins, data=team_wins, palette="coolwarm")
plt.title("🏆 Distribution of Teams by Win Count Ranges")
plt.xlabel("Win Ranges")
plt.ylabel("Number of Teams")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ╰┈➤ 7. Stats and Hypothesis Testing
mean_wins = team_wins['Total Wins'].mean()
std_wins = team_wins['Total Wins'].std()

print("\n📊 Statistics Summary:")
print("Mean Wins per Team:", round(mean_wins, 2))
print("Standard Deviation:", round(std_wins, 2))

# One-Sample t-Test (H0: Mean = 100)
t_stat, p_value = ttest_1samp(team_wins['Total Wins'], 100)
print("\n🧪 One-Sample t-Test on Total Wins:")
print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {p_value:.4f}")
print("Result:", "✅ Reject H0" if p_value < 0.05 else "❌ Fail to Reject H0")

# ╰┈➤ 8. Line Chart: Wins against India
if "India" in head_to_head.columns:
    plt.figure(figsize=(14, 6))
    sns.lineplot(data=head_to_head["India"].sort_values(ascending=False).reset_index(), x='winner', y='India', marker='o', color='crimson')
    plt.title("🆚 Wins by Teams Against India")
    plt.xlabel("Team")
    plt.ylabel("Wins Against India")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
