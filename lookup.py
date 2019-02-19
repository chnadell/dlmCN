import os

def gridGen(data_dir, grid_dir):
    # get data to be read
    file_paths = os.listdir(data_dir)

    # define output file
    outFile = os.path.join(grid_dir, 'gridDataOut.txt')

    # open files
    with open(outFile) as f_out:
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                # read the data
                line = f.readline()

                # write the data to predictions file




if __name__=="__main__":
    gridGen(os.path.join('.','dataIn','eval'))
