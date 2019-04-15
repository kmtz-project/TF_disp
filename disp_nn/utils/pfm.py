import numpy as np
import re
import struct

def readPfm(file):
        with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
            type = f.readline().decode('latin-1')
            if "PF" in type:
                channels = 3
            elif "Pf" in type:
                channels = 1
            else:
                sys.exit(1)

            # Line 2: width height
            line = f.readline().decode('latin-1')
            width, height = re.findall('\d+', line)
            width = int(width)
            height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
            line = f.readline().decode('latin-1')
            BigEndian = True
            if "-" in line:
                BigEndian = False
            # Slurp all binary data
            samples = width * height * channels;
            buffer = f.read(samples * 4)
            # Unpack floats with appropriate endianness
            if BigEndian:
                fmt = ">"
            else:
                fmt = "<"
            fmt = fmt + str(samples) + "f"
            img = struct.unpack(fmt, buffer)

        img = np.array(img)

        img[img == float("Inf")] = 0
        img = np.round(np.reshape(img, (height, width)))
        img = np.fliplr([img])[0]

        return img