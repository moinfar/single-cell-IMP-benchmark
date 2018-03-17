import errno
import os


def make_sure_dir_exists(dire_name):
    try:
        os.makedirs(dire_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def download_file(url, file_path):
    from six.moves.urllib.request import urlretrieve

    urlretrieve(url, filename=file_path)
