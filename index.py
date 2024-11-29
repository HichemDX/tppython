import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xarray as xr
import rioxarray
from scipy.stats import zscore

# Constants
CLIMATE_DATA_PATH = "Climate-DATA"
OUTPUT_FOLDER = "Output/"
COUNTRY_FILE = "Country/Country.shp"
SOIL_DATA_FILE = "soil_dz_allprops.csv"
EXISTING_DATASET_PATH = f"{OUTPUT_FOLDER}FullDataSet.csv"
CLIMATE_VARIABLES = ["PSurf", "Qair", "Rainf", "Snowf", "Tair", "Wind"]
MONTHS = [f"{i:02d}" for i in range(1, 13)]

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Helper functions for dataset handling
def validate_and_clean_csv(file_path):
    """Validate and clean CSV, handling parsing errors."""
    try:
        return pd.read_csv(file_path)
    except pd.errors.ParserError:
        st.warning("CSV format issues detected. Attempting repair...")
        df = pd.read_csv(file_path, on_bad_lines="skip", engine="python")
        df.to_csv(file_path, index=False)  # Overwrite with cleaned file
        return df


def load_or_generate_dataset():
    """Load existing dataset or generate one if it doesn't exist."""
    if os.path.exists(EXISTING_DATASET_PATH):
        st.sidebar.success("Existing dataset found.")
        return validate_and_clean_csv(EXISTING_DATASET_PATH)
    else:
        st.sidebar.warning("Existing dataset not found. Generating dataset...")
        generate_dataset()
        return validate_and_clean_csv(EXISTING_DATASET_PATH)


def generate_dataset():
    """Generate dataset by processing climate and soil data."""
    algeria_polygon = load_algeria_polygon()
    process_climate_data(algeria_polygon)
    merge_climate_csv_files()
    process_soil_data()


def load_algeria_polygon():
    """Load Algeria polygon from the country shapefile."""
    country_data = gpd.read_file(COUNTRY_FILE)
    algeria_polygon = country_data[country_data["CNTRY_NAME"] == "Algeria"]
    algeria_polygon = algeria_polygon.to_crs("EPSG:4326")
    st.sidebar.success("Loaded Algeria polygon.")
    return algeria_polygon


def process_climate_data(algeria_polygon):
    """Process climate NetCDF files and save to individual CSV files."""
    for variable in CLIMATE_VARIABLES:
        all_data = []
        for month in MONTHS:
            file_path = os.path.join(CLIMATE_DATA_PATH, f"{variable}_WFDE5_CRU_2019{month}_v2.1.nc")
            try:
                dataset = xr.open_dataset(file_path)
                dataset = dataset.rio.write_crs("EPSG:4326")
                clipped = dataset.rio.clip(algeria_polygon.geometry, dataset.rio.crs)
                df = clipped[[variable]].to_dataframe().reset_index()
                df = df[["time", "lon", "lat", variable]]
                all_data.append(df)
            except FileNotFoundError:
                st.warning(f"File not found: {file_path}")
            except Exception as e:
                st.error(f"Error processing {file_path}: {e}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(f"{OUTPUT_FOLDER}FullData_{variable}.csv", index=False)
            st.success(f"Processed and saved data for {variable}.")
        else:
            st.warning(f"No valid data for variable {variable}.")


def merge_climate_csv_files():
    """Merge all climate CSV files into a single file."""
    all_data = []
    for variable in CLIMATE_VARIABLES:
        file_path = os.path.join(OUTPUT_FOLDER, f"FullData_{variable}.csv")
        if os.path.exists(file_path):
            all_data.append(pd.read_csv(file_path))
        else:
            st.warning(f"File not found: {file_path}")

    if all_data:
        merged_data = pd.concat(all_data, ignore_index=True)
        merged_data = merged_data.groupby(["time", "lon", "lat"], as_index=False).first()
        merged_data.to_csv(f"{OUTPUT_FOLDER}FullDataClimate.csv", index=False)
        st.success("Merged climate data successfully.")
    else:
        st.error("No climate data files to merge.")


def process_soil_data():
    """Merge climate data with soil data."""
    try:
        soil_data = pd.read_csv(SOIL_DATA_FILE)
        soil_data["geometry"] = soil_data["geometry"].apply(lambda geom: gpd.GeoSeries.from_wkt([geom])[0])
        soil_gdf = gpd.GeoDataFrame(soil_data, geometry="geometry")

        climate_data_file = f"{OUTPUT_FOLDER}FullDataClimate.csv"
        if not os.path.exists(climate_data_file):
            st.error("Climate data file not found. Please process climate data first.")
            return

        climate_data = pd.read_csv(climate_data_file)
        climate_gdf = gpd.GeoDataFrame(
            climate_data, geometry=gpd.points_from_xy(climate_data["lon"], climate_data["lat"])
        )

        combined_gdf = gpd.sjoin(climate_gdf, soil_gdf, how="left", predicate="intersects")
        combined_gdf.drop(columns=["geometry", "index_right"], inplace=True)
        combined_gdf.to_csv(EXISTING_DATASET_PATH, index=False)
        st.success("Soil and climate data merged successfully.")
    except Exception as e:
        st.error(f"Error processing soil data: {e}")


# Streamlit UI
st.title("Dataset Management and Analysis")

# Add a button to regenerate the dataset overview
if st.button("Regenerate Dataset Overview"):
    # Load or generate dataset again (reload the dataset)
    df = load_or_generate_dataset()
    st.success("Dataset Overview has been regenerated.")
else:
    # Load or generate dataset if not already loaded
    if 'df' not in locals():
        df = load_or_generate_dataset()

if not df.empty:
    # Dataset Overview Section
    st.header("Dataset Overview")

    # Display dataset size
    st.subheader("General Information")
    st.write(f"**Number of Rows:** {df.shape[0]}")
    st.write(f"**Number of Columns:** {df.shape[1]}")

    # Display general information about each attribute
    st.subheader("Attribute Information")

    # Create a dictionary for attribute statistics
    attribute_info = {
        "Attribute": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Unique Values": [df[col].nunique() for col in df.columns],
        "Missing Values": [df[col].isna().sum() for col in df.columns],
        "Mean": [df[col].mean() if df[col].dtype in ["float64", "int64"] else "N/A" for col in df.columns],
        "Median": [df[col].median() if df[col].dtype in ["float64", "int64"] else "N/A" for col in df.columns],
        "Mode": [df[col].mode().iloc[0] if not df[col].mode().empty else "N/A" for col in df.columns],
        "Standard Deviation": [df[col].std() if df[col].dtype in ["float64", "int64"] else "N/A" for col in df.columns],
        "Skewness": [df[col].skew() if df[col].dtype in ["float64", "int64"] else "N/A" for col in df.columns],
    }

    # Convert the dictionary to a DataFrame for display
    attribute_info_df = pd.DataFrame(attribute_info)

    # Format numeric values only; leave non-numeric values as they are
    st.dataframe(attribute_info_df.style.format(
        {
            "Unique Values": "{:.0f}",
            "Missing Values": "{:.0f}",
            "Mean": lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x,
            "Median": lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x,
            "Mode": lambda x: f"{x}" if pd.notna(x) else "N/A",
            "Standard Deviation": lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x,
            "Skewness": lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int, float)) else x,
        }
    ))

    # Display the first few rows of the dataset
    st.subheader("Sample Data")
    st.write(df.head())

    # Data manipulation
    st.subheader("Data Manipulation")
    if st.checkbox("Edit a value"):
        index = st.number_input("Row index to edit", min_value=0, max_value=len(df) - 1)
        column = st.selectbox("Column to edit", df.columns)
        new_value = st.text_input(f"New value for {column}")
        if st.button("Update"):
            df.at[index, column] = new_value
            st.success("Value updated.")
            df.to_csv(EXISTING_DATASET_PATH, index=False)

    if st.checkbox("Delete a row"):
        index = st.number_input("Row index to delete", min_value=0, max_value=len(df) - 1)
        if st.button("Delete"):
            df.drop(index, inplace=True)
            st.success("Row deleted.")
            df.to_csv(EXISTING_DATASET_PATH, index=False)

    # New section: Handle missing data
    st.subheader("Handle Missing Data")
    impute_method = st.selectbox("Select Imputation Method",
                                 ["Imputation par moyenne", "Imputation par médiane", "Suppression"])

    if st.button("Apply Imputation"):
        # Impute or drop missing values
        numeric_df = df.select_dtypes(include=["float64", "int64"])

        if impute_method == "Imputation par moyenne":
            # Replace NaN with mean
            df[numeric_df.columns] = numeric_df.fillna(numeric_df.mean())
        elif impute_method == "Imputation par médiane":
            # Replace NaN with median
            df[numeric_df.columns] = numeric_df.fillna(numeric_df.median())
        elif impute_method == "Suppression":
            # Drop rows with NaN values
            df = df.dropna()

        # Show updated dataset
        st.write("Updated Dataset")
        st.write(df.head())

        # Option to download updated data
        df.to_csv(EXISTING_DATASET_PATH, index=False)
        st.success(f"Dataset updated with {impute_method}.")

        # Export the updated dataset
        with open(EXISTING_DATASET_PATH, "rb") as file:
            st.download_button(
                label="Download Updated Dataset",
                data=file,
                file_name="UpdatedData.csv",
                mime="text/csv"
            )

    # Attribute analysis using pandas methods
    st.subheader("Attribute Analysis")
    column = st.selectbox("Select an attribute", df.columns)

    if st.button("Analyze Attribute"):
        # Calculate statistics using pandas
        stats = {
            "Statistic": ["Mean", "Median", "Standard Deviation", "Skewness", "Missing Values", "Unique Values"],
            "Value": [
                df[column].mean(),  # Mean
                df[column].median(),  # Median
                df[column].std(),  # Standard Deviation
                df[column].skew(),  # Skewness (pandas handles missing values)
                df[column].isna().sum(),  # Missing Values
                df[column].nunique()  # Unique Values
            ]
        }

        # Create a DataFrame for display
        stats_df = pd.DataFrame(stats)

        # Display as a styled table
        st.subheader(f"Summary Statistics for {column}")
        st.dataframe(stats_df.style.format({"Value": "{:.2f}" if stats_df["Value"].dtype == "float" else "{}"}))

        # Visualizations
        st.subheader("Visualizations")
        fig, ax = plt.subplots()
        df[column].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        df.boxplot(column=column, ax=ax)
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    # Correlation analysis
    st.subheader("Correlation Analysis")
    if st.button("Show Correlation Matrix"):
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        st.write(numeric_df.corr())

    # Data reduction
    st.subheader("Data Reduction")
    if st.button("Aggregate by Season"):
        df["time"] = pd.to_datetime(df["time"])
        df["season"] = df["time"].dt.month % 12 // 3 + 1
        aggregated = df.groupby("season").mean()
        st.write(aggregated)

    # Normalization
    st.subheader("Normalization")
    method = st.selectbox("Select Normalization Method", ["Min-Max", "Z-Score"])
    if st.button("Normalize Data"):
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        if method == "Min-Max":
            normalized = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
        else:
            normalized = numeric_df.apply(zscore)
        st.write("Normalized Data Sample:")
        st.write(normalized.head(100))

        # Download normalized data
        if st.button("Download Normalized Data"):
            normalized.to_csv("NormalizedData.csv", index=False)
            st.success("Normalized dataset saved as 'NormalizedData.csv'.")
            with open("NormalizedData.csv", "rb") as file:
                st.download_button(
                    label="Download Normalized Dataset",
                    data=file,
                    file_name="NormalizedData.csv",
                    mime="text/csv"
                )

    # Discretization
    st.subheader("Discretization")
    bins = st.number_input("Number of bins", min_value=2, max_value=10, value=4)
    column = st.selectbox("Select column for discretization", df.columns)
    if st.button("Discretize"):
        df[f"{column}_binned"] = pd.cut(df[column], bins=bins)
        st.write(df[[column, f"{column}_binned"]])

        # Export discretized data
        with open(EXISTING_DATASET_PATH, "rb") as file:
            st.download_button(
                label="Download Discretized Dataset",
                data=file,
                file_name="DiscretizedData.csv",
                mime="text/csv"
            )

    # Data reduction (eliminate redundancies)
    st.subheader("Data Reduction")

    # Horizontal (Remove duplicate rows)
    if st.button("Remove Duplicate Rows (Horizontal)"):
        initial_row_count = df.shape[0]
        df_no_duplicates_rows = df.drop_duplicates()
        final_row_count = df_no_duplicates_rows.shape[0]
        st.write(f"Removed {initial_row_count - final_row_count} duplicate rows.")
        st.write(df_no_duplicates_rows.head())  # Display first few rows of reduced data
        df = df_no_duplicates_rows  # Update the main DataFrame with the reduced dataset
        df.to_csv(EXISTING_DATASET_PATH, index=False)  # Save the reduced dataset

        # Export reduced dataset
        with open(EXISTING_DATASET_PATH, "rb") as file:
            st.download_button(
                label="Download Reduced Dataset",
                data=file,
                file_name="ReducedData.csv",
                mime="text/csv"
            )

    # Vertical (Remove duplicate columns)
    if st.button("Remove Duplicate Columns (Vertical)"):
        initial_column_count = df.shape[1]
        df_no_duplicates_cols = df.loc[:, ~df.columns.duplicated()]
        final_column_count = df_no_duplicates_cols.shape[1]
        st.write(f"Removed {initial_column_count - final_column_count} duplicate columns.")
        st.write(df_no_duplicates_cols.head())  # Display first few rows of reduced data
        df = df_no_duplicates_cols  # Update the main DataFrame with the reduced dataset
        df.to_csv(EXISTING_DATASET_PATH, index=False)  # Save the reduced dataset

        # Export reduced dataset
        with open(EXISTING_DATASET_PATH, "rb") as file:
            st.download_button(
                label="Download Reduced Dataset",
                data=file,
                file_name="ReducedData.csv",
                mime="text/csv"
            )

    # Export Final Dataset (General Export Button)
    if st.button("Export Final Dataset"):
        df.to_csv(EXISTING_DATASET_PATH, index=False)
        with open(EXISTING_DATASET_PATH, "rb") as file:
            st.download_button(
                label="Download Final Dataset",
                data=file,
                file_name="FinalData.csv",
                mime="text/csv"
            )

else:
    st.error("Dataset is empty. Please ensure the dataset is generated correctly.")
