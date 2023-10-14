import asyncio
import aiohttp
from aiohttp import ClientSession
import cv2
import numpy as np
from bs4 import BeautifulSoup
import sys
sys.path.append('C:\Users\xDis4ster\Documents\GitHub\Wallpaper-Engine-Tagger\directml')

import ctypes
directml_dll = ctypes.CDLL('directml/directml.dll')
# Create a DirectML device
device = directml_dll.DMLCreateDevice()

# Use the device for your computations

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import logging
import json
import time
from typing import List, Tuple

logging.basicConfig(filename='wallpaper_scanner.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Define browsersort types and pages 
BROWSERSORT_TYPES = ['trend', 'mostrecent', 'lastupdated', 'totaluniquesubscribers'] 
NUM_PAGES = 1000

# Async helper to fetch page data
async def fetch(url: str, session: ClientSession) -> str:
    try:
        async with session.get(url) as response:
            return await response.text()
    except Exception as e:
        logging.error(json.dumps({"event": "fetch_error", "url": url, "error": str(e)}))
        return ''

# Async helper to get all page URLs 
async def get_page_urls(browsersort: str) -> List[str]:
    page_urls = []
    async with ClientSession() as session:
        for i in range(1, NUM_PAGES+1):
            url = f"https://steamcommunity.com/workshop/browse/?appid=431960&browsesort={browsersort}&actualsort={browsersort}&p={i}"
            page_data = await fetch(url, session)
            page_urls.append(url)
    return page_urls

try:
    # Convert to DirectML model
    model = dmtf.convert_tf_model(model, device)
except Exception as e:
    # Load ResNet50 model
    model = ResNet50(weights='imagenet')


x = preprocess_input(x)



# Async helper to classify image
async def classify_image(url: str) -> Tuple[str, str]:
    try:
        async with aiohttp.ClientSession() as session:
            resp = await session.get(url)
            image_data = await resp.read()
        img = cv2.imdecode(np.asarray(bytearray(image_data), dtype=np.uint8), cv2.IMREAD_COLOR) 
        x = cv2.resize(img, (224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        with device:
            preds = model.predict(x)
        # Check if top prediction is female
        if decode_predictions(preds, top=1)[0][0] == 'woman':
            return (url, 'girl')
        else:
            return (url, '')
    except Exception as e:
        logging.error(json.dumps({"event": "classification_error", "url": url, "error": str(e)}))
        return (url, '')

# Classify images concurrently 
async def classify_images(preview_urls: List[str]) -> List[Tuple[str, str]]:
    tasks = [classify_image(u) for u in preview_urls]
    return await asyncio.gather(*tasks)

# Get all page URLs for each browsersort type
async def main():
    page_urls = await asyncio.gather(*[get_page_urls(b) for b in BROWSERSORT_TYPES])

    # Extract preview URLs from page data
    preview_urls = []
    for url in page_urls:
        page_data = await fetch(url, session)
        # Use BeautifulSoup to parse HTML and extract preview URLs
        soup = BeautifulSoup(page_data, 'html.parser')
        previews = soup.find_all('div', {'class': 'workshopItemPreviewHolder'})
        for preview in previews:
            preview_url = preview.find('img')['src']
            preview_urls.append(preview_url)

    # Classify images and write results to file
    results = await classify_images(preview_urls)
    with open('wallpapers.txt', 'w') as f:
        for url, tag in results:
            f.write(f"{url}, {tag}\n")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    logging.info(json.dumps({"event": "completed", "elapsed_time": time.time() - start_time}))