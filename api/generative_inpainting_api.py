import requests
import json

import numpy as np


class GenerativeInpaintingApi:
    def __init__(self, root_path):
        self._url = 'http://localhost:9000/inpaint'

    def inpaint(self, image, mask):
        data = {'image': image.tolist(), 'mask': mask.tolist()}
        response = requests.post(self._url, json=data, timeout=300)
        response = json.loads(response.content.decode('utf-8'))
        return np.array(response['result'], dtype=np.uint8)









