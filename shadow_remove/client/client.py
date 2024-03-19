import asyncio
import aiohttp
from PIL import Image
import io
import requests


# async def main():
#     url = 'http://127.0.0.1:8080/enhanced/fbs'
#     with open('2015_02927.jpg', 'rb') as f:
#         async with aiohttp.ClientSession() as session:
#             async with session.post(url, data=f.read()) as response:
#                 return await response.read()
#
#
# if __name__ in "__main__":
#     image_bytes = asyncio.run(main())  # Assuming you're using python 3.7+
#     image = Image.open(io.BytesIO(image_bytes))
#     image.show()

if __name__ in '__main__':
    url = 'http://127.0.0.1:8080/enhanced/fbs'
    with open('../images/cat.jpg', 'rb') as f:
        res = requests.post(url=url, data=f.read())
        image = Image.open(io.BytesIO(res.content))
        image.show()
