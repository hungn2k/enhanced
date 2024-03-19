from aiohttp import web
from PIL import Image
import io
from runner.runner_shadow_remove import enhanced
import numpy as np
from settings import SERVICE

PORT = SERVICE["SERVER_PORT"]
HOST = SERVICE["SERVER_HOST"]


async def image_enhanced_fbs(request):
    data = await request.read()

    if data:
        origin_image = Image.open(io.BytesIO(data))
        enhanced_image = enhanced(np.array(origin_image), use_filter=True, filter_name="fbs")

        result = Image.fromarray(enhanced_image)
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return web.Response(body=img_byte_arr, headers={"Access-Control-Allow-Origin": "*"})

    msg = {"message": "data field is not file."}
    return web.json_response(msg, status=400)


async def image_enhanced_bc(request):
    data = await request.read()
    if data:
        origin_image = Image.open(io.BytesIO(data))
        enhanced_image = enhanced(np.array(origin_image), use_filter=True, filter_name="cr")

        result = Image.fromarray(enhanced_image)
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return web.Response(body=img_byte_arr, headers={"Access-Control-Allow-Origin": "*"})

    msg = {"message": "data field is not file."}
    return web.json_response(msg, status=400)


async def image_enhanced_guided(request):
    data = await request.read()

    if data:
        origin_image = Image.open(io.BytesIO(data))
        enhanced_image = enhanced(np.array(origin_image), use_filter=True, filter_name="guided_filter")

        result = Image.fromarray(enhanced_image)
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return web.Response(body=img_byte_arr, headers={"Access-Control-Allow-Origin": "*"})

    msg = {"message": "data field is not file."}
    return web.json_response(msg, status=400)


async def image_enhanced_no_filter(request):
    data = await request.read()

    if data:
        origin_image = Image.open(io.BytesIO(data))
        enhanced_image = enhanced(np.array(origin_image), use_filter=False)

        result = Image.fromarray(enhanced_image)
        img_byte_arr = io.BytesIO()
        result.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return web.Response(body=img_byte_arr, headers={"Access-Control-Allow-Origin": "*"})

    msg = {"message": "data field is not file."}
    return web.json_response(msg, status=400)


async def app_factory():
    app = web.Application()
    app.add_routes([web.post('/enhanced/fbs', image_enhanced_fbs),
                    web.post('/enhanced/cr', image_enhanced_bc),
                    web.post('/enhanced/guided', image_enhanced_guided),
                    web.post('/enhanced/no_filter', image_enhanced_no_filter)])
    return app


if __name__ in "__main__":
    web.run_app(app_factory(), host=HOST, port=PORT)
