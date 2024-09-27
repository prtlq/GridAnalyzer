import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


path1 = r'diff/2019_12_09/sm_w/mx_m.csv'

df1 = pd.read_csv(path1, header=None, delimiter=';')
df1.columns = ['X', 'Y', 'Z', 'Interpolated_Values']
df1[['X', 'Y', 'Z', 'Interpolated_Values']] = df1[['X', 'Y', 'Z', 'Interpolated_Values']].apply(pd.to_numeric, errors='coerce')

path2 = r'diff/2019_12_09/ap_d/ga.csv'

df2 = pd.read_csv(path2, header=None, delimiter=',')
df2.columns = ['X', 'Y', 'Z', 'Interpolated_Values']
df2[['X', 'Y', 'Z', 'Interpolated_Values']] = df2[['X', 'Y', 'Z', 'Interpolated_Values']].apply(pd.to_numeric, errors='coerce')

# Adjust based on your data and desired resolution
x_bin_size = 10
y_bin_size = 10
z_bin_size = 10

def aggregate_data(df, x_bin_size, y_bin_size, z_bin_size):
    # bins
    df['x_bin'] = np.floor(df['X'] / x_bin_size) * x_bin_size
    df['y_bin'] = np.floor(df['Y'] / y_bin_size) * y_bin_size
    df['z_bin'] = np.floor(df['Z'] / z_bin_size) * z_bin_size
    
    aggregated = df.groupby(['x_bin', 'y_bin', 'z_bin'])['Interpolated_Values'].mean().reset_index()
    return aggregated

aggregated1 = aggregate_data(df1, x_bin_size, y_bin_size, z_bin_size)
aggregated2 = aggregate_data(df2, x_bin_size, y_bin_size, z_bin_size)

comparison = pd.merge(aggregated1, aggregated2, on=['x_bin', 'y_bin', 'z_bin'], suffixes=('_1', '_2'))

comparison['diff'] = comparison['Interpolated_Values_1'] - comparison['Interpolated_Values_2']

print(comparison)


xy_diff = comparison.groupby(['x_bin', 'y_bin'])['diff'].mean().reset_index()

pivot_table = xy_diff.pivot(index='y_bin', columns='x_bin', values='diff')

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, annot=True, cmap='coolwarm', fmt=".2f")  # annot=True displays the values
plt.title('Mean Differences in Cumulative Moment for 1 week in the XY Plane, 2019-01-30 2102 - Bl 34, Bin Size:50')
plt.xlabel('X Bin')
plt.ylabel('Y Bin')
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = comparison['x_bin']
# y = comparison['y_bin']
# z = comparison['z_bin']
# diff = comparison['diff']

# img = ax.scatter(x, y, z, c=diff, cmap=plt.hot())
# ax.set_xlabel('X Bin')
# ax.set_ylabel('Y Bin')
# ax.set_zlabel('Z Bin')
# fig.colorbar(img)
# plt.title('3D Scatter Plot of Differences')
# plt.show()
