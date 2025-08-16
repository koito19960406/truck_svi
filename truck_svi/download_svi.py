from zensvi.download import GSVDownloader
import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
import os 

# load environment variables
load_dotenv()
GSV_API_KEY = os.getenv("GSV_API_KEY")
DIR_PROCESSED = "data/processed"
DIR_SVI_IMAGES = os.path.join(DIR_PROCESSED, "svi_images")
os.makedirs(DIR_SVI_IMAGES, exist_ok=True)

def main():
    print("Now downloading Street View images for each centroid...")
    
    city_list = [
        "cuttack",
        "kanpur"
    ]
    
    for city in city_list:
        city_svi_dir = os.path.join(DIR_SVI_IMAGES, city)
        os.makedirs(city_svi_dir, exist_ok=True)

        input_geojson = f"data/raw/geojson/{city}_speeds.geojson"
        
        print(f"Reading geojson and calculating centroids for {city}...")
        gdf = gpd.read_file(input_geojson)
        gdf['geometry'] = gdf.geometry.centroid
        gdf = gdf.to_crs(4326)
        
        gdf['lon'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        
        output_csv_path = os.path.join(DIR_PROCESSED, f"{city}_centroids.csv")
        print(f"Saving centroids to {output_csv_path}...")
        df_for_csv = gdf.drop(columns='geometry')
        df_for_csv.to_csv(output_csv_path, index=False)

        downloader = GSVDownloader(
            gsv_api_key=GSV_API_KEY
        )
        downloader.download_svi(
            dir_output=city_svi_dir,
            input_csv_file=output_csv_path,
            metadata_only=True,
            augment_metadata=False
        )

        print(f"Finding nearest SVI points for {city}...")
        svi_pids_path = os.path.join(city_svi_dir, "gsv_pids.csv")
        if not os.path.exists(svi_pids_path):
            print(f"Warning: {svi_pids_path} not found. Skipping nearest point calculation for {city}.")
            continue
        
        svi_df = pd.read_csv(svi_pids_path)
        svi_gdf = gpd.GeoDataFrame(
            svi_df, geometry=gpd.points_from_xy(svi_df.lon, svi_df.lat), crs="EPSG:4326"
        )
        
        # Estimate UTM zone
        utm_crs = gdf.estimate_utm_crs()
        print(f"Projecting coordinates to {utm_crs} for distance calculation...")

        gdf_utm = gdf.to_crs(utm_crs)
        svi_gdf_utm = svi_gdf.to_crs(utm_crs)
        
        merged_gdf = gpd.sjoin_nearest(gdf_utm, svi_gdf_utm, how="left", distance_col="dist")

        # Merge back with original gdf to get wgs84 coordinates for centroids
        final_df = merged_gdf.drop(columns=['geometry', 'index_right'])

        final_columns = [
            'panoid', 'id', 'osm_id', 'deviceSpeed', 'lat', 'lon',
            'year', 'month', 'input_latitude', 'input_longitude', 'dist'
        ]
        
        # Ensure all columns exist, fill with None if not
        for col in final_columns:
            if col not in final_df.columns:
                final_df[col] = None
        
        final_df = final_df[final_columns]
        
        nearest_csv_path = os.path.join(DIR_PROCESSED, f"svi_images/{city}/nearest_svi.csv")
        print(f"Saving nearest SVI points to {nearest_csv_path}...")
        final_df.to_csv(nearest_csv_path, index=False)
        
        downloader.download_svi(
            path_pid = nearest_csv_path,
            dir_output=city_svi_dir,
            metadata_only=False,
            augment_metadata=True,
        )
        
        print(f"Finished processing for {city}.")

if __name__ == "__main__":
    main()