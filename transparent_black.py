import os
from PIL import Image

def make_black_pixels_transparent(input_folder, output_folder):
    """
    For each PNG in input_folder, make all black pixels transparent and save to output_folder with the same name.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for fname in os.listdir(input_folder):
        if fname.lower().endswith('.png'):
            in_path = os.path.join(input_folder, fname)
            out_path = os.path.join(output_folder, fname)
            img = Image.open(in_path).convert('RGBA')
            data = img.getdata()
            new_data = []
            for item in data:
                # item is (R, G, B, A)
                if item[0] == 0 and item[1] == 0 and item[2] == 0:
                    # Make black pixel transparent
                    new_data.append((0, 0, 0, 0))
                else:
                    new_data.append(item)
            img.putdata(new_data)
            img.save(out_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python png_make_black_transparent.py <input_folder> <output_folder>")
    else:
        make_black_pixels_transparent(sys.argv[1], sys.argv[2])
