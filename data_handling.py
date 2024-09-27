import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from configmanifest import DataManager

# Calculated Parameters 

def add_energy_index_column(data):    #note: ENERGY INDEX IS GENERAL, BUT SHALL BE DEFINE BASED ON BLOCK 

    data['Energy Index'] = data.iloc[:, 6] / np.exp(1.26 * np.log10(data.iloc[:, 5]) - 9.28)

    # data['Energy Index'].replace([np.inf, -np.inf], np.nan, inplace=True)  # Optional
    # data.dropna(subset=['Energy Index'], inplace=True)  # Optional: Drop rows with NaN in 'Energy Index'
    
    return data

# CSV handling
 
def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    root.destroy()
    return file_path

def read_csv(file_path):
    try:
        column_names = [
            "DateTime", "X", "Y", "Z", "ML", "Seismic Moment", 
            "TotalRadiatedEnergy", "Apparent Stress", "Residual", "Potency", 
            "LogP", "Max Displacement Source", "EsEp", "Static Stress Drop", 
            "SourceRadius", "SWaveCornerFrequency", "SensorsHit", "SensorsUsed",
            "Apparent Volume", "Comment", "UserTag", "ImportedTag", 
            "DistanceToClosestGeometry", "DistanceToClosestDevelopment"
        ]
        
        data = pd.read_csv(file_path, header=0, names=column_names, encoding='ISO-8859-1')
        data = add_energy_index_column(data)  
        
        data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
        data['Epoch'] = data['DateTime'].apply(lambda x: x.timestamp() if not pd.isna(x) else None)

        print("Column headers in the DataFrame:")
        print(data.columns)
        return data
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# Analytics and Visualization 
class DataProcessor:
    def __init__(self):
        self.df = None
        self.cut_min_value = None
        self.cut_max_value = None
        
    def histogram_tab(self, plot_type, data, data_manager, selected_column, start_datetime, end_datetime):

        data['DateTime'] = pd.to_datetime(data['DateTime'])
        
        min_x = float(data_manager.get_value('X_min'))
        max_x = float(data_manager.get_value('X_max'))
        min_y = float(data_manager.get_value('Y_min'))
        max_y = float(data_manager.get_value('Y_max'))
        min_z = float(data_manager.get_value('Z_min'))
        max_z = float(data_manager.get_value('Z_max'))
        

        filtered_data = data[(data['X'] >= min_x) & (data['X'] <= max_x) &
                            (data['Y'] >= min_y) & (data['Y'] <= max_y) &
                            (data['Z'] >= min_z) & (data['Z'] <= max_z) &
                            (data['DateTime'] >= start_datetime) & (data['DateTime'] <= end_datetime)]

        fig, ax1 = plt.subplots()

        bins = int(data_manager.get_value('bins_value'))
        alpha = float(data_manager.get_value('Alpha_value'))
        print(f"Bins: {bins}, alpha: {alpha}")

        if plot_type == 'histogram' and selected_column in filtered_data.columns:
            data_column = np.log10(filtered_data[selected_column].dropna())

            n, bins, patches = ax1.hist(data_column, bins=bins, alpha=alpha, color='blue')

            sorted_data = np.sort(data_column)
            cdf = np.searchsorted(sorted_data, sorted_data, side='right') / len(sorted_data)
            ax2 = ax1.twinx()
            ax2.plot(sorted_data, cdf, color='red')
            ax2.set_ylabel('Cumulative Distribution')


            ax1.set_xlabel(f'Log of {selected_column}')
            ax1.set_ylabel('Frequency')

        elif plot_type == 'plot2':
            ax1.plot([1, 2, 3], [3, 2, 1]) 

        return fig
    def block_tab(self, data, data_manager, selected_column, start_datetime, end_datetime, interpolation_method='average', update_view=None, selected_view=None, order=3):

        if not selected_column or selected_column not in data.columns:
            print("Selected column is not valid or not provided.")
            return None
        
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        
        min_x = float(data_manager.get_value('X_min'))
        max_x = float(data_manager.get_value('X_max'))
        min_y = float(data_manager.get_value('Y_min'))
        max_y = float(data_manager.get_value('Y_max'))
        min_z = float(data_manager.get_value('Z_min'))
        max_z = float(data_manager.get_value('Z_max'))
        
        grid_spacing = int(data_manager.get_value('Grid Spacing_value'))

        Rmin = int(data_manager.get_value('R_min')) * grid_spacing
        Rmax = int(data_manager.get_value('R_max')) * grid_spacing
        N = int(data_manager.get_value('N_value'))

        # Apply filters
        filtered_data = data[(data['X'] >= min_x) & (data['X'] <= max_x) &
                            (data['Y'] >= min_y) & (data['Y'] <= max_y) &
                            (data['Z'] >= min_z) & (data['Z'] <= max_z) &
                            (data['DateTime'] >= start_datetime) & (data['DateTime'] <= end_datetime)]

        coordinates = filtered_data.iloc[:, 1:4].values
        values = filtered_data[selected_column].dropna().values

        # coordinates = data.iloc[:, 1:4].values
        # values = data[selected_column].dropna().values

        events = np.column_stack((coordinates, values))

        source_radius = filtered_data.iloc[:, 14].values
        source_radius = source_radius.reshape(-1, 1)

        x_vals = np.arange(min_x, max_x + 1, grid_spacing)
        y_vals = np.arange(min_y, max_y + 1, grid_spacing)
        z_vals = np.arange(min_z, max_z + 1, grid_spacing)
        meshgrid = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1, 3)

        safe_selected_column = selected_column.replace(" ", "_").replace("/", "_").lower() 
        output_file = f'output_{interpolation_method}_{safe_selected_column}_grid.csv'
        
        def weighted_average_interpolation(coordinates, values, min_x, max_x, min_y, max_y, min_z, max_z, Rmax):

            mask_x = (coordinates[:, 0] >= min_x - Rmax*1.2) & (coordinates[:, 0] <= max_x + Rmax*1.2)
            mask_y = (coordinates[:, 1] >= min_y - Rmax*1.2) & (coordinates[:, 1] <= max_y + Rmax*1.2)
            mask_z = (coordinates[:, 2] >= min_z - Rmax*1.2) & (coordinates[:, 2] <= max_z + Rmax*1.2)

            threshold_distance = 90
            minimum_events_within_threshold = 10

            combined_mask = mask_x & mask_y & mask_z

            # Apply the combined mask to filter events
            coordinates = coordinates[combined_mask]
            values = values[combined_mask]

            interpolated_values = []
            interpolated_points = []
            
            for point in meshgrid:
                distances = np.linalg.norm(coordinates - point, axis=1)

                within_radius = distances <= Rmin
                if np.sum(within_radius) < N: # will not include points beyond Rmin if already reached N
                    within_radius = distances <= Rmax # will include points as far as Rmax to probably reaches N

                neighbors_distances_all = distances[within_radius]
                neighbors_values_all = values[within_radius]

                neighbors_indices_selective = np.argsort(neighbors_distances_all)[:N]
                neighbors_distances_selective = neighbors_distances_all[neighbors_indices_selective]
                
                # Density-based quality check
                index_of_greater_than_threshold_distance = np.searchsorted(neighbors_distances_selective, threshold_distance, side='right')
                if index_of_greater_than_threshold_distance < minimum_events_within_threshold:
                    continue

                thetas = neighbors_distances_selective / neighbors_distances_selective[-1] # pick the highest distance value as "Search Radius" to apply kernel func.
                weights = np.where(thetas >= 1, 0, (1 - thetas ** 2) ** order) # Apply the kernel function
                weighted_values = neighbors_values_all[neighbors_indices_selective] * weights
                sum_weights = np.sum(weights)
                if sum_weights == 0:
                    continue
                else:
                    interpolated_value = np.sum(weighted_values) / sum_weights

                interpolated_values.append(interpolated_value)
                interpolated_points.append(point)

            interpolated_points = np.array(interpolated_points)
            interpolated_values = np.array(interpolated_values)

            # Take the logarithm of the results, handling zeros/negatives
            mask = interpolated_values <= 0
            interpolated_values[mask] = np.nan  # Assigning NaN to zero/negative values
            interpolated_values = np.log10(interpolated_values)

            # Creating DataFrame with X, Y, Z, and interpolated values
            df = pd.DataFrame({
                'X': interpolated_points[:, 0],
                'Y': interpolated_points[:, 1],
                'Z': interpolated_points[:, 2],
                'Interpolated_Values': interpolated_values
            })

            #print(f'DF: {df}')
            df.to_csv(output_file, index=False)

            return df

        def weighted_cumulative_interpolation(events, meshgrid, source_radius):

            all_kernels_of_thetas = []
            factors = []

            involved_events = events
            # Create a boolean array to exclude non-involved events later
            mask = np.ones(len(events), dtype=bool)

            for index, event in enumerate(events):
                event_coords = event[:3]
                distances = np.linalg.norm(meshgrid[:, np.newaxis] - event[:3], axis=2)
                thetas = distances / source_radius[index]

                kernels_of_thetas = np.where(thetas >= 1, 0, (1 - thetas ** 2) ** order)  # Apply the kernel function
                factor = sum(kernels_of_thetas)
                if factor == 0:
                    # Update the boolean array, mask
                    mask[index] = False
                else:
                    # Append factor to factors.
                    factors.append(factor)
                    # Append kernels_of_thetas to all_kernels_of_thetas. Rows are events and columns are grid points.
                    # We will need all_kernels_of_thetas for interpolation later, as K(thetas(i,j))
                    all_kernels_of_thetas.append(kernels_of_thetas)

            # Extract the involved events for cumulative interpolation
            # involved_events is going to be used as "All events" in equation 1 in the Wesseloo et. al paper (2014)
            involved_events = events[mask]


            total = sum(row[3] for row in involved_events)  # Summation of property values of involved event
            # Extract the excluded events,  just in case we need them for future developments
            reversed_mask = [not value for value in mask]
            not_involved_events = events[reversed_mask]

            # remove one extra dimension that the loop made!
            all_kernels_of_thetas = np.squeeze(all_kernels_of_thetas)
            # Convert the all_kernels_of_thetas to a numpy array and transpose it. Easier to be used later!
            all_kernels_of_thetas = np.array(all_kernels_of_thetas).T
            # Now Rows are grid points and columns are events!

            factors = np.array(factors)  # Convert the factors to a numpy array
            factors = np.squeeze(factors)  # convert it to a 1-dimensional array

            interpolated_grid = np.zeros((len(meshgrid), 4))  # Initialize an array to store results
            number_of_events = np.zeros((len(meshgrid), 1))  # Initialize an array to store number of events

            for index, point in enumerate(meshgrid):
                interpolated_value = np.sum(
                    (involved_events[:, 3] * all_kernels_of_thetas[index, :]) / factors)
                interpolated_grid[index, :3] = point  # Set the first three elements as the points for output
                interpolated_grid[index, 3] = interpolated_value  # Set the last element as the interpolated_value for output

            sum_grids = sum(interpolated_grid[:, 3])

            # Set the condition for rows where the 4th value is zero
            condition = interpolated_grid[:, 3] != 0

            # Use boolean indexing to get the rows where the condition is True
            interpolated_grid = interpolated_grid[condition]

            # Take the logarithm of the results, handling zeros/negatives
            mask2 = interpolated_grid[:, 3] <= 0
            interpolated_grid[:, 3][mask2] = np.nan  # Assigning NaN to zero/negative values
            interpolated_grid[:, 3] = np.log10(interpolated_grid[:, 3])

            df = pd.DataFrame({
                'X': interpolated_grid[:, 0],
                'Y': interpolated_grid[:, 1],
                'Z': interpolated_grid[:, 2],
                'Interpolated_Values': interpolated_grid[:, 3]
            })

            df.to_csv(output_file, index=False)

                    ## Here we are extracting the number of events to be saved as another output
            # Find rows in all_kernels_of_thetas with at least one non-zero element
            non_zero_indices = np.any(all_kernels_of_thetas != 0, axis=1)
            # Extract rows from meshgrid based on non-zero indices
            new_list = meshgrid[non_zero_indices]
            # Count non-zero elements in each row of all_kernels_of_thetas
            non_zero_counts = np.count_nonzero(all_kernels_of_thetas, axis=1)
            # Add non-zero counts as the fourth column to new_list
            new_list_with_counts = np.column_stack((new_list, non_zero_counts[non_zero_indices].reshape(-1, 1)))

            # Creating DataFrame with X, Y, Z, and number of events
            df_events_num = pd.DataFrame({
                'X': new_list_with_counts[:, 0],
                'Y': new_list_with_counts[:, 1],
                'Z': new_list_with_counts[:, 2],
                'Events_Number': new_list_with_counts[:, 3]
            })

            # Save DataFrame to a CSV file
            df_events_num.to_csv(f'_number_of_events_{output_file}', index=False)

            return df, df_events_num
        
        if interpolation_method == 'average':
            df = weighted_average_interpolation(coordinates, values, min_x, max_x, min_y, max_y, min_z, max_z, Rmax)
        elif interpolation_method == 'cumulative':
            df, df_events_num = weighted_cumulative_interpolation(events, meshgrid, source_radius)

        if df is not None:    
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(df['X'], df['Y'], df['Z'], c=df['Interpolated_Values'], cmap='rainbow', marker='s', s=50)

            cbar = plt.colorbar(sc, ax=ax, fraction=0.02, pad=0.05)
            cbar.set_label('Interpolated Values', fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            fig.suptitle(f"Interpolation Results: {selected_column}", fontsize=9)
            fig.text(0.5, 0.01, f"Start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')} - End: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}", ha='center', fontsize=7)

            ax.set_xlabel('X', fontsize=7)
            ax.set_ylabel('Y', fontsize=7)
            ax.set_zlabel('Z', fontsize=7)

            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            ax.tick_params(axis='z', labelsize=6)

            if update_view is not None:
                update_view(ax, selected_view)

            filename = f"{selected_column}_{start_datetime.strftime('%Y-%m-%d')}_{end_datetime.strftime('%Y-%m-%d')}.png"
            filename = filename.replace(':', '-').replace(' ', '_')
            fig.savefig(filename, dpi=800)

            plt.close()

        else:
            raise ValueError(f"DataFrame is not defined. interpolation method: {interpolation_method}")

        self.df=df

        return fig, self.df, ax

    def plane_tab(self, data_manager, selected_axis, cut_step=None, cols=3, point_marker_size=15, cmap='rainbow', png_output_file_name='plot_2d_slices_.png',
                    title='The title of plot', min_scale=None, max_scale=None):
        
        if hasattr(self, 'cut_min_value') and hasattr(self, 'cut_max_value'):

            return   #waiting for new update

            cut_step = int(data_manager.get_value('Cut step_value'))
            min_cut = int(data_manager.get_value('Min cut_value'))
            max_cut = int(data_manager.get_value('Max cut_value'))
            print(f"Selected Axis: {selected_axis}, Cut Step: {cut_step}, Min Cut: {min_cut}, Max Cut: {max_cut}")

            coordinates = self.df.iloc[:, :3].copy()
            values = self.df.iloc[:, 3].copy()

            if selected_axis == 'X':
                cut_axis_inx = 0
                h_axis_label = 'Y'; v_axis_label = 'Z'; h_axis = 1; v_axis = 2
            elif selected_axis == 'Y':
                cut_axis_inx = 1
                h_axis_label = 'X'; v_axis_label = 'Z'; h_axis = 0; v_axis = 2
            elif selected_axis == 'Z':
                cut_axis_inx = 2
                h_axis_label = 'Y'; v_axis_label = 'X'; h_axis = 1; v_axis = 0
            else:
                print("Error! The cut_axis should be 'x', 'y' or 'z'")
                quit

            min_h = coordinates.iloc[:, h_axis].min()
            max_h = coordinates.iloc[:, h_axis].max()
            min_v = coordinates.iloc[:, v_axis].min()
            max_v = coordinates.iloc[:, v_axis].max()

            # if min_scale and max_scale is not given, determine the overall min and max
            # the same range (scale) for color map of property values to be used for
            if min_scale is None:
                overall_min_diff = min(values.min(), 0)
                min_scale = overall_min_diff + 0.2 * overall_min_diff
            if max_scale is None:
                overall_max_diff = max(values.max(), 0)
                max_scale = overall_max_diff - 0.2 * overall_max_diff

            if max_cut - min_cut < cut_step:
                num_slices = 2
                rows = 1
                figsize = (int(np.ceil((max_h-min_h)/100)), int(np.ceil((max_v-min_v)/100)))
                fig, axs = plt.subplots(rows, cols, figsize=figsize )
            else:
                num_slices = int(np.ceil((max_cut - min_cut) / cut_step))
                rows = int(np.ceil(num_slices / cols))
                figsize = (int(np.ceil((max_h-min_h)/100)) * cols+10, int(np.ceil((max_v-min_v)/100)) * rows)
                fig, axs = plt.subplots(rows, cols, figsize=figsize )

            # Iterate over each slice
            for i, s in enumerate(range(min_cut, max_cut + 1, cut_step)):
                # Filter coordinates and values for the current s value
                slice_coordinates = coordinates[coordinates.iloc[:, cut_axis_inx] == s]
                slice_values = values[coordinates.iloc[:, cut_axis_inx] == s]
                # Plot the slice if there are data points
                if not slice_coordinates.empty:
                    row = i // cols
                    col = i % cols
                    ax = axs[row, col] if rows > 1 else axs[col]

                    scatter = ax.scatter(slice_coordinates.iloc[:, h_axis], slice_coordinates.iloc[:, v_axis], c=slice_values,
                                        cmap=cmap, vmin=min_scale, vmax=max_scale, marker='s', s=point_marker_size)  # 's' for square marker

                    ax.set_title(f"{selected_axis} = {s}")
                    ax.set_xlabel(f"{h_axis_label} location")
                    ax.set_ylabel(f"{v_axis_label} location")

                    # Set the same ranges for horizontal and vertical axes
                    ax.set_xlim(min_h, max_h)
                    ax.set_ylim(min_v, max_v)

                    # Add color bar legend
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label(property_name)

            # Hide empty subplots
            for i in range(num_slices, rows * cols):
                row = i // cols
                col = i % cols

                
                axs[row, col].axis('off')

            plt.suptitle(title, fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust layout to make space for the title
            # plt.tight_layout()
            # Save the plot in good quality with a randomly numbered name
            plt.savefig(png_output_file_name, dpi=300, bbox_inches='tight')
            plt.show()

        else:
            print("Min and Max cut values are not set.")
