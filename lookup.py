import os

def gridGen(data_dir, grid_dir):

    # get data to be read
    file_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

    # define output file
    outFile = os.path.join(grid_dir, 'gridDataOut.txt')

    # define data generator
    def get_geom(file_path):
        with open(file_path) as file:
            line = file.readline()
            yield line

    # open files and get geometry
    with open(os.path.join(grid_dir, 'gridFile.csv'), 'w+') as gridFile:
        for file_path in file_paths:
            print('file path is {}'.format(file_path))
            iter_lines = iter(get_geom(file_path))
            for line in iter_lines:
                geom = line[:8]
                gridFile.write(geom)







if __name__=="__main__":
    gridGen(os.path.join('.','dataIn','eval'), os.path.join('.', 'dataGrid'))
