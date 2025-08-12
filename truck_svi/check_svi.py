from zensvi.download import GSVDownloader
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
import os 

# load environment variables
load_dotenv()
GSV_API_KEY = os.getenv("GSV_API_KEY")
DIR_OUTPUT = "data/processed/svi_images"
os.makedirs(DIR_OUTPUT, exist_ok=True)

# load .csv in data/raw/GPS Data and concatenate to pandas dataframe
def load_gps_data():
    # Load all CSV files from the directory
    df_list = [pd.read_csv(f"data/raw/GPS Data/{file}") for file in os.listdir("data/raw/GPS Data") if file.endswith('.csv')]

    # Concatenate all dataframes into one
    gps_data = pd.concat(df_list, ignore_index=True)
    
    return gps_data

# convert this df to geodataframe
def convert_to_geodataframe(gps_data):
    # Convert the DataFrame to a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        gps_data,
        geometry=gpd.points_from_xy(gps_data['Longitude'], gps_data['Latitude']),
        crs="EPSG:4326"  # Assuming the coordinates are in WGS84
    )
    return gdf

def main():
    # Load GPS data
    gps_data = load_gps_data()
    
    # save to a csv file
    gps_data.to_csv('data/processed/gps_data.csv', index=False)
    
    print("Now downloading Street View images for each centroid...")
    downloader = GSVDownloader(
        gsv_api_key=GSV_API_KEY
    )
    downloader.download_svi(
        dir_output=DIR_OUTPUT,
        input_csv_file="data/processed/gps_data.csv",
        metadata_only=True,
        augment_metadata=True
    )

if __name__ == "__main__":
    main()