from zensvi.cv import Segmenter
import os

def main():
    segmenter = Segmenter(dataset = "mapillary")
    
    city_list = [
        "cuttack",
        "kanpur"
    ]
    
    for city in city_list:
        # find the number of batch folder in data/processed/svi_images/cuttack/gsv_panorama
        base_path = f"data/processed/svi_images/{city}"
        batch_list = os.listdir(os.path.join(base_path, "gsv_panorama"))
        for batch in batch_list:
            # Set the input and output directories
            dir_input = os.path.join(base_path, "perspective", batch, "perspective")

            # segment images
            dir_image_output = os.path.join(base_path, "segmented", batch)
            os.makedirs(dir_image_output, exist_ok=True)
            segmenter.segment(dir_input=dir_input, 
                              dir_image_output=dir_image_output,
                              dir_summary_output=base_path,
                              save_format="csv",
                              csv_format="wide")

if __name__ == "__main__":
    main()