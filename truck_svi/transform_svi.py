from zensvi.transform import ImageTransformer
import os

def main():
    
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
            dir_input = os.path.join(base_path, "gsv_panorama", batch)
            dir_output = os.path.join(base_path, "perspective", batch)

            # Start transforming images
            # Set the parameters for the transformation, field of view (FOV), angle of view (theta), angle of view (phi), aspect ratio, size of the image to show (show_size), use_upper_half
            image_transformer = ImageTransformer(dir_input=dir_input, dir_output=dir_output)
            image_transformer.transform_images(
                style_list="perspective",  # list of projection styles in the form of a string separated by a space
                FOV=90,  # field of view
                theta=90,  # angle of view (horizontal)
                phi=0,  # angle of view (vertical)
                aspects=(9, 16),  # aspect ratio
                show_size=100, # size of the image to show (i.e. scale factor)
                use_upper_half=False, # if True, only the upper half of the image is used for transformation. Use this for fisheye images to estimate sky view.
            ) 

if __name__ == "__main__":
    main()