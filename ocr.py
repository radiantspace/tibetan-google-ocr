#!/usr/local/bin/python3

from __future__ import print_function
import httplib2
import os
import io
import time

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from functools import partial

import sys
import codecs
import logging
from Wylie import Wylie

from multiprocessing import Pool, Manager, Value, Array
from ctypes import c_char_p


class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    @property
    def value(self):
        return self.val.value


# class Errors(object):
#     def __init__(self):
#         self.val = Array(c_char_p, 255)
#         self.i = 0

#     def add(self, error):
#         with self.val.get_lock():
#             self.val[self.i] = str(error)
#             self.i += 1

#     @property
#     def value(self):
#         return self.val


class OCR(object):
    # Setup logger
    from logging import config
    config.fileConfig('logging.conf')
    logging.getLogger('googleapiclient.discovery').setLevel(logging.CRITICAL)

    # If modifying these scopes, delete your previously saved credentials
    # at ~/.credentials/drive-python-quickstart.json
    SCOPES = 'https://www.googleapis.com/auth/drive'
    CLIENT_SECRET_FILE = 'client_secret_1029773793109-tm33do2mms9t9j42hck7vmc1ks7pgnre.apps.googleusercontent.com.json'
    APPLICATION_NAME = 'dw-toolbox-python'

    # Initialize multithread safe counters
    # errors = Errors()
    img_files_counter = Counter()
    txt_files_counter = Counter()
    wyl_files_counter = Counter()

    # Default properties
    foldername = ""
    credentials = None

    def get_credentials(self):
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.

        Returns:
            Credentials, the obtained credential.
        """
        logger = logging.getLogger(__name__)
        credential_path = os.path.join("./", 'drive-python-quickstart.json')
        store = Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(
                self.CLIENT_SECRET_FILE, self.SCOPES)
            flow.user_agent = self.APPLICATION_NAME
            if flags:
                credentials = tools.run_flow(flow, store, flags)
            else:  # Needed only for compatibility with Python 2.6
                credentials = tools.run(flow, store)
            logger.debug('Storing credentials to ' + credential_path)
        return credentials

    def add_wylie(self, txtfilename):
        logger = logging.getLogger(__name__)
        with codecs.open(txtfilename, 'r', 'utf-8', 'replace') as fin:
            tibetan = fin.read()

        with open(txtfilename[:-4] + '.wyl', 'w') as fout:
            warns = []
            fout.write(Wylie().toWylieOptions(
                tibetan, warns, not flags.skip_nontibetan))
            if flags.warnings:
                if len(warns) > 0:
                    warnings = "\n\n\nWarnings:\n" + '\n'.join(warns)
                    logger.debug('Warnings: ' + warns)
                    fout.write(warnings)

    def ocr_file(self, file):
        logger = logging.getLogger(__name__)
        supported_extentions = ['.png', '.jpg']

        file_name, file_ext = os.path.splitext(file)
        if not file_ext in supported_extentions:
            return

        full_file_name_without_ext = os.path.join(self.foldername, file_name)
        imgfilename = full_file_name_without_ext + file_ext
        txtfilename = full_file_name_without_ext + '.txt'

        # OCR image if there is no txt file yet
        if not os.path.isfile(txtfilename) or os.stat(txtfilename).st_size == 0:
            try:
                mime = 'application/vnd.google-apps.document'
                http = self.credentials.authorize(httplib2.Http())
                service = discovery.build(
                    'drive', 'v3', http=http, cache_discovery=False)
                res = service.files().create(
                    body={
                        'name': imgfilename,
                        'mimeType': mime
                    },
                    media_body=MediaFileUpload(
                        imgfilename, mimetype=mime, resumable=True)
                    #, ocrLanguage='bo'
                ).execute()

                downloader = MediaIoBaseDownload(
                    io.FileIO(txtfilename, 'wb'),
                    service.files().export_media(
                        fileId=res['id'], mimeType="text/plain")
                )
                done = False
                while done is False:
                    status, done = downloader.next_chunk()

                if os.stat(txtfilename).st_size == 0:
                    logger.error('Zero size file: ' + txtfilename)
                    raise Exception('Zero size!')

                service.files().delete(fileId=res['id']).execute()
                self.img_files_counter.increment()
            except Exception as e:
                exception = "Failed OCR: " +\
                    imgfilename + ", error: " + str(e)
                logger.error(exception)

                try:
                    os.remove(txtfilename)
                except OSError:
                    pass

                raise e
            else:
                self.txt_files_counter.increment()
                logger.info("OCRed: " + file)
        else:
            logger.debug("Skipped OCR: " + file)

        # Convert to Wylie
        if os.path.isfile(txtfilename):
            if not os.path.isfile(txtfilename[:-4] + '.wyl') or os.stat(txtfilename[:-4] + '.wyl').st_size == 0:
                try:
                    self.add_wylie(txtfilename)
                except Exception as e:
                    exception = "Failed converting to Wylie: " +\
                        txtfilename + ", error: " + str(e)
                    logger.error(exception)
                    raise exception
                else:
                    self.wyl_files_counter.increment()
                    logger.info("Converted to Wylie: " + txtfilename)
            else:
                logger.debug("Skipped converting to Wylie: " +
                             txtfilename + " was already converted")
        else:
            logger.debug("Skipped converting to Wylie: " +
                         txtfilename + " was not there")

    def ocr_file_with_retry(self, file):
        try:
            logger = logging.getLogger(__name__)
            for retry in range(10):
                try:
                    self.ocr_file(file)
                    break
                except Exception as e:
                    logger.debug('Retrying: ' + file +
                                 ', times: ' + str(retry + 1))
                    time.sleep(2)
        except e:
            print("Unexpected error: " + str(e))
            pass

    def unzip_file(self, path_to_zip_file):
        import zipfile
        zip_handle = zipfile.ZipFile(path_to_zip_file, 'r')
        zip_handle.extractall(path_to_zip_file[:-4])
        zip_handle.close()

    def zip_file_with_ext(self, folder, ext):
        import zipfile
        archive_name = '../' + \
            os.path.basename(os.path.normpath(folder)) + ext + '.zip'
        zip_handle = zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED)

        for root, dirs, files in os.walk(folder):
            for file in files:
                if os.path.splitext(file)[1] == ext:
                    zip_handle.write(os.path.join(root, file))

        zip_handle.close()

    def start(self, flags):
        logger = logging.getLogger(__name__)
        for foldername in flags.folder:
            if os.path.isfile(foldername) and os.path.splitext(foldername)[1] == '.zip':
                logger.info('Unzipping: ' + foldername)
                self.unzip_file(foldername)
                self.foldername = os.path.splitext(foldername)[0]
            else:
                self.foldername = foldername

            files = os.listdir(self.foldername)

            if not files:
                logger.info("Folder is empty")
                quit()

            self.credentials = self.get_credentials()

            pool = Pool(20)
            pool.map(self.ocr_file_with_retry, files)
            pool.close()
            pool.join()

            logger.info("\nFolder contains: " + str(len(files)) + " files\nProcessed: " + str(self.img_files_counter.value) + " images\nOCRed: " +
                        str(self.txt_files_counter.value) + " files\nConverted to Wylie:" +
                        str(self.wyl_files_counter.value) + " files")

            if flags.zip_results:
                logger.info('Archiving txt files...')
                self.zip_file_with_ext(self.foldername, '.txt')
                logger.info('Done.')

                logger.info('Archiving wyl files...')
                self.zip_file_with_ext(self.foldername, '.wyl')
                logger.info('Done.')

            # logger.debug('Errors: ')
            # for error in self.errors.value:
            #     if error:
            #         logger.error(str(error))


if __name__ == '__main__':
    try:
        import argparse
        parser = argparse.ArgumentParser(
            description="OCR folder using Google Drive",
            parents=[tools.argparser],
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="Installation instructions.\n\n"
            "1. Use the guidelines from here: "
            "https://developers.google.com/drive/v3/web/quickstart/python\n"
            "The script works with Python 3.* only (due to Wylie conversion part)\n"
            "2. pip3 install --upgrade google-api-python-client\n"
            "3. chmod +x ocr.py\n\n"
            "Note.\n\n"
            "If you see errors, just restart the script in the end,\n"
            "it will skip processing for already existent files\n"
        )

        parser.add_argument('folder', nargs='+', help='Folder/zip file to OCR')
        parser.add_argument('-w', '--warnings', action='store_true',
                            help='Add warnings to wylie files')
        parser.add_argument('-s', '--skip_nontibetan', action='store_true',
                            help='Skip non-tibetan symbols in wylie conversion')
        parser.add_argument('-z', '--zip_results', action='store_true',
                            help='Zip resulting files')
        flags = parser.parse_args()
    except ImportError:
        flags = None

    OCR().start(flags)
