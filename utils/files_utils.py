import os
import sys
import tarfile
import zipfile
import requests
from urllib.request import urlretrieve
from pathlib import Path


class FilesUtils:
    @staticmethod
    def download(url, target_path):
        if url.startswith('ftp://'):
            urlretrieve(url, target_path, FilesUtils.reporthook)
        else:
            with open(target_path, "wb") as f:
                print("Downloading {}".format(url))
                response = requests.get(url, stream=True)
                total_length = response.headers.get('content-length')

                if total_length is None:  # no content length header
                    raise Exception('Failed to find "content-length" in url header.')
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                        sys.stdout.flush()

                    print('')

    @staticmethod
    def extract(file_path, sub_path=None):
        print("Extracting '{}' with subpath '{}'".format(file_path, sub_path))
        parent_dir = Path(file_path).parent
        if zipfile.is_zipfile(file_path):
            file = zipfile.ZipFile(file_path, 'r')
        elif file_path.endswith("tar.gz"):
            file = tarfile.open(file_path, "r:gz")
        else:
            file = tarfile.open(file_path, "r:")

        if sub_path is not None:
            subdir_and_files = []
            members = file.getmembers()

            for p in sub_path:
                for t in members:
                    if t.name.startswith(p):
                        subdir_and_files.append(t)

            file.extractall(path=parent_dir, members=subdir_and_files)

        else:
            file.extractall(path=parent_dir)

        file.close()

    @staticmethod
    def validate_path(path):
        if not os.path.exists(path):
            raise Exception('Path "{}" does not exists.'.format(path))

    @staticmethod
    def reporthook(block_num, block_size, total_size):
        readsofar = block_num * block_size
        if total_size > 0:
            percent = readsofar * 1e2 / total_size
            s = "\r%5.1f%% %*d / %d" % (
                percent, len(str(total_size)), readsofar, total_size)
            sys.stderr.write(s)
            if readsofar >= total_size:
                sys.stderr.write("\n")
        else:
            sys.stderr.write("read %d\n" % (readsofar,))