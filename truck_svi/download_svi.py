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

def main():
    print("Now downloading Street View images for each centroid...")
    
    city_list = [
        "cuttack",
        "kanpur"
    ]
    
    for city in city_list:
        DIR_OUTPUT = f"data/processed/svi_images/{city}"
        os.makedirs(DIR_OUTPUT, exist_ok=True)
        downloader = GSVDownloader(
            gsv_api_key=GSV_API_KEY
        )
        downloader.download_svi(
            dir_output=DIR_OUTPUT,
            input_shp_file=f"data/processed/{city}_speeds.geojson",
            metadata_only=False,
            augment_metadata=True
        )

if __name__ == "__main__":
    main()