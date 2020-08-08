import os

#-----------------------------------
# creates a parent directory
# for a given file name
# file_name - directory name
#-----------------------------------
def empty_or_create_directory(file_name):
        
    # setup empty directory
    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))

        # Guard against race condition
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
