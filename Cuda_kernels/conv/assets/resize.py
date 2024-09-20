import os
from PIL import Image

# Specify the directory containing the images
directory = '/home/tesla/exp/Notes/Cuda_kernels/conv'

# Specify the new size (width, height)
new_size = (1000, 1000)  # Change to your desired size

# Loop through each file in the directory
for filename in os.listdir(directory):
    # Check if the file is a .png or .jpg file
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Open the image
        img_path = os.path.join(directory, filename)
        with Image.open(img_path) as img:
            # Resize the image
            img_resized = img.resize(new_size)
            
            # Save the resized image, optionally you can save with a new name
            img_resized.save(img_path)

print("Resizing complete!")
