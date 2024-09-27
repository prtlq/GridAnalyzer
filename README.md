
## How to Use the App
---

![alt text](manual.png)

### Left Panel: Data Selection and Range Definition

1. **Selecting a File:**
   - Click on the **"Select the File"** button to open Windows Explorer.
   - Choose a CSV file containing your data.

2. **Loading Data:**
   - Observe the message box on the lower left-hand side of the app.
   - A message "Read data from..." confirms that the data is loaded.

3. **Setting Temporal and Spatial Ranges:**
   - The temporal sliders update to show the maximum and minimum values from your database.
   - Adjust the time range by scrolling the handles or typing directly into the time boxes in the format YYYY-MM-DD HH:MM:SS.
   - Modify the spatial range to focus on the specific block.

4. **Adjusting Parameters and Ranges:**
   - Use the **Marker** dropdown menu to select the desired parameter, which updates the marker range.
   - Adjust the R range and N values as needed.

5. **Executing Tasks:**
   - The **Execute** button performs different tasks based on the active tab.
     - **Histogram Tab**: Clicking execute generates a histogram and a CDF curve based on the defined marker range.
     - **Block Tab**: Executes grid analysis for the selected block. The type of analysis (average or cumulative) depends on the marker. The completion of analytics updates the Block tab and displays a plot in iso view.

### Middle Panel: Display Options


1. **Tabs Overview:**
   - The panel includes three main tabs: **Histogram**, **Block**, and **Plane**. 
   - Note the blue line under the active tab, which indicates which tab is currently selected. This selection will dictate the operations performed by the **Execute** button when pressed.

### Right Panel: Settings and Batch Processing

1. **Histogram Settings:**
   - Set bin size and alpha value for histograms in the histogram settings section.

2. **Block and Plane Settings:**
   - Adjust grid spacing and view settings. Begin with a higher grid spacing for large datasets and adjust as necessary. Cumulative calculations may be time-consuming for large datasets.
   - Set parameters for plane cuts, including step size, axis selection, and the min/max cut values. 

3. **Batch Processing:**

   - Enable batch processing by checking the "Batch Opt" box.
   - Set the event time in the format YYYY-MM-DD HH:MM:SS. Define the backward time and interval for batch processing (e.g., 7 or 28 days).
   - Click the **Execute** button to start batch operations. Monitor the message box for updates and completion messages.
   - Access the resulting plots and output CSV file in the app folder.
---
### Release Note  

#### Version 514
- **Compatibility Enhancements**: Improved compatibility for X64 systems to ensure better performance and stability.

#### Version 311
- **Plane Tab Disabled**: The Plane tab has been temporarily disabled to focus on other critical improvements.
- **General Enhancements**: Various optimizations and minor fixes for the nightly release, enhancing overall functionality.

#### Version 301
- **Naming Convention Update**: Changed the naming convention to use dates, making it easier to track versions.
- **Batch Operation**: Implemented batch operation capabilities to streamline processes and improve efficiency.
- **Block Tab View**: Introduced plane view for the Block tab

#### Version 1.5
- **Settings Update**: Added settings options for the Histogram, Block and Plane tabs.

#### Version 1.4
- Added the Plane tab

#### Version 1.3
- Introduced cumulative parameter analytics

#### Version 1.2
- **Block Tab Addition**: Added the Block tab, for average parameter analytics.
- **General Enhancements**: Various improvements to overall functionality and performance.

#### Version 1.1
- **File Reading and Range Pickers**: Enhanced file reading capabilities and range pickers.
- **Histogram Tab**: Added the Histogram tab to provide better data visualization and analysis tools.