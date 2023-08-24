from utils.data_utils import data_prep, test_data_prep
from utils.train_utils import set_seed

set_seed(6_345_789)
# Wilson Pickett - 634-5789 https://www.youtube.com/watch?v=TSGuaVAufV0

root_folder = "data/multi_shelf_bounce/"
data_folders = ["variable_angle/", "variable_shelfheight/", "variable_time/",
                "variable_time_angle/", "variable_angle_shelfheight/", "variable_time_angle_shelfheight/"]
train_test_split = [0.8, 0.1, 0.1]

for data_folder in data_folders:
    data_prep(folder_loc=root_folder + data_folder,
              file_name='minimal_text.txt', line_delimiter='\n', ans_delimiter=' ans: ', split=train_test_split,
              save_indices=True, split_method='train_val_test')

    test_data_prep(folder_loc=root_folder + data_folder, file_name='oos_minimal_text.txt',
                   line_delimiter='\n', ans_delimiter=' ans: ')
