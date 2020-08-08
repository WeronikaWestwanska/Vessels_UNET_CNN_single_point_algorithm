data_params = {
    "width"                            : 3504,
    "height"                           : 2336,
    "labelled_train_db"                : 'data/db/vessels.train.db',
    "labelled_train_dir"               : 'data/images/',
    "labelled_validate_db"             : 'data/db/vessels.validate.db',
    "labelled_validate_dir"            : 'data/images/',
    "segmented_coarse_dir"             : 'data/segmented_coarse/',
    "segmented_fine_dir"               : 'data/segmented_fine/',
    'fine_segmented_partial_file_name' : '_segmented_binary.png'
}

model_params = {
    "num_classes" : 2,
    "batch_size"  : 150,
    "epochs"      : 50
}

hyper_params = {
    "max_training_images_count"           : 100,     # if value is -1 then take all training images
    "max_testing_images_count"            : -1,      # if value is -1 then take all training images
    "l2_regularisation"                   : 0.0005,
    "dropout"                             : 0.50,
    "learning_rate"                       : 0.0001,
    "default_background_probability"      : 0.80,
    "default_vessel_probability"          : 0.20,
    "percentage_train"                    : 90,
    "windows_per_image_on_average_coarse" : 120,
    "windows_per_image_on_average_fine"   : 1200,
    "window_size"                         : 50,
    "vessel_radius"                       : 20,
    "min_vessel_prob"                     : 0.90,
    "max_vessel_prob"                     : 1.00,
    "vessel_class_index"                  : 1,
    "background_radius"                   : 50,
    "min_background_prob"                 : 0.80,
    "max_background_prob"                 : 1.00,
    "background_class_index"              : 0,
    "min_vessel_prct_window"              : 45.0,
    "min_background_prct_window"          : 50.0,
    "vessel_window_percentage"            : 50,
    "background_window_percentage"        : 50,
    "filters_count"                       : 32,
    "kernel_size"                         : 3,
    "padding_to_remove"                   : 15,
    "sliding_window_step"                 : 15,
    "binarisation_threshold"              : 0.58,
    "rectangle_searcher_window_size"      : 100
}
