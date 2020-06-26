# General Imports
import asyncio
import uuid
import os
import cv2
import sys
import urllib
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def check_for_completion(lis):
    """
    Check for errored image downloads and returns the error count

    Parameters
    ----------
    lis : list
        list of dict of input data

    Returns:
    --------
    error : int
        the error count
    """

    error = 0
    for n in lis:
        if not os.path.isfile(n['img_path']):
            error += 1
    return error


def download_one_image(url, abs_path):
    """Downloads one image

    Parameters
    ----------
    url: str

    abs_path: str

    Returns
    -------
    None
    """
    try:
        # Check path
        if not os.path.isdir(os.path.dirname(abs_path)):
            os.makedirs(os.path.dirname(abs_path))

        # Download_file
        r = urllib.request.urlopen(url, timeout=20)
        with open(abs_path, 'wb') as f:
            f.write(r.read())
            f.close()

        # Check for corrupted image
        temp = cv2.imread(abs_path)
        if temp is not None:
            cv2.imwrite(abs_path, temp)
        else:
            os.remove(abs_path)
        return True
    except Exception as e:
        pass
        return False


async def async_image_downloader(lis, no_process):
    """Takes in a list of image paths and downloads images asynchronously.

    Parameters
    ----------
    lis : list of dict
        Example:
            [{'image_url': <>, 'img_path': <>},
             {'image_url': <>, 'img_path': <>}]
    no_process : int
        Number of processes.
    """

    # Filter only not downloaded
    lis = [n for n in lis if not os.path.isfile(n['img_path'])]

    # Starting down
    with ThreadPoolExecutor(max_workers=no_process) as executor:
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                executor,
                download_one_image,
                n['image_url'],
                n['img_path']
            )
            for n in lis
        ]
        responses = []
        for f in tqdm(asyncio.as_completed(futures), total=len(futures)):
            responses.append(await f)

        for response in await asyncio.gather(*futures):
            pass

    # Check for completion
    error = check_for_completion(lis)
    print("Error Count : ", error)


def download_df_lis(df_lis, folder, column_name="image_url", no_process=50):

    print("Starting to download images...")

    # Generate the image path to the files that will be downloaded.
    new_column = "img_path"
    for i, n in enumerate(df_lis):
        image_name = str(uuid.uuid5(
            uuid.NAMESPACE_URL, n[column_name])) + '.jpg'
        df_lis[i][new_column] = os.path.join(folder, image_name)

    # Generate list of dictionaries each with the image_urls and the target image paths
    lis = [{'image_url': n[column_name], 'img_path': n[new_column]} for n in df_lis]
    image_downloader(lis, no_process)
    return df_lis


def image_downloader(lis, no_process):
    """Function that starts an event loop and calls the async_image_downloader.

    Parameters
    ----------
    lis: list

    no_process: int
        Number of processes.

    Returns
    -------
    None
    """
    # async download
    try:
        # Create a new loop, run the task and close the loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        future = asyncio.ensure_future(async_image_downloader(lis, no_process))
        loop.run_until_complete(future)
    except Exception as e:
        # If an asyncio loop already running, get it and create a new task for it
        loop = asyncio.get_event_loop()
        loop.create_task(async_image_downloader(lis, no_process))
        asyncio.run_coroutine_threadsafe(async_image_downloader(lis, no_process), loop)

    print("Image Download Done")


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1])
    df = download_df_lis(df.to_dict(orient='records'), sys.argv[2])
    df = pd.DataFrame().from_dict(df)
    df.to_csv(sys.argv[1], index=False)
