"""Conservative image preparation with an inverse coordinate transform."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any


def prepare_image(raw: bytes, output: Path) -> dict[str, Any]:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(io.BytesIO(raw)) as original:
            orientation = original.getexif().get(274, 1)
            image = ImageOps.exif_transpose(original).convert('RGB')
            width, height = image.size
            if width * height > 32_000_000:
                return {'status':'skipped', 'reason':'large_image'}
            array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    except (ImportError, OSError, ValueError):
        return {'status':'unavailable', 'reason':'unreadable_image'}
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    flags = []
    if min(width, height) < 160: flags.append('low_resolution')
    if sharpness < 35: flags.append('blurred')
    if contrast < 15: flags.append('low_contrast')
    transform = np.eye(3)
    # Crop only surrounding blank margins; keep a margin around all dark pixels.
    ink = cv2.findNonZero((gray < 220).astype('uint8'))
    crop = [0, 0, width, height]
    if ink is not None:
        x, y, w, h = cv2.boundingRect(ink)
        pad = max(12, round(min(width, height) * .015))
        left, top, right, bottom = max(0, x-pad), max(0, y-pad), min(width, x+w+pad), min(height, y+h+pad)
        if (right-left)*(bottom-top) >= width*height*.25:
            array = array[top:bottom, left:right]
            crop = [left, top, right, bottom]
            transform[:2, 2] = [-left, -top]
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=max(40, array.shape[1]//4), maxLineGap=15)
    angles = [] if lines is None else [float(np.degrees(np.arctan2(y2-y1, x2-x1))) for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4) if x2 != x1]
    angles = [angle for angle in angles if abs(angle) <= 7]
    angle = float(np.median(angles)) if len(angles) >= 3 else 0.0
    if .25 <= abs(angle) <= 7:
        h, w = array.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        cosine, sine = abs(matrix[0,0]), abs(matrix[0,1])
        new_w, new_h = int(h*sine+w*cosine)+2, int(h*cosine+w*sine)+2
        matrix[0,2] += (new_w-w)/2; matrix[1,2] += (new_h-h)/2
        array = cv2.warpAffine(array, matrix, (new_w,new_h), borderValue=(255,255,255))
        affine = np.eye(3); affine[:2] = matrix; transform = affine @ transform
    array = cv2.bilateralFilter(array, 3, 20, 20)
    if not cv2.imwrite(str(output), array):
        raise OSError('Cannot save prepared OCR image.')
    return {'status':'prepared', 'exif_orientation':orientation, 'deskew_degrees':round(angle,3), 'crop':crop,
            'page_width':width, 'page_height':height, 'inverse_transform':np.linalg.inv(transform).tolist(),
            'quality':{'sharpness':round(sharpness,3), 'contrast':round(contrast,3), 'flags':flags},
            'operations':['exif_rotation', 'blank_margin_crop', 'bounded_deskew', 'edge_preserving_denoise']}


def restore_blocks(blocks: list[dict[str, Any]], preparation: dict[str, Any]) -> None:
    if preparation.get('status') != 'prepared':
        return
    import numpy as np
    inverse = np.asarray(preparation['inverse_transform'])
    for block in blocks:
        points = np.asarray([[*point, 1] for point in block['polygon']]) @ inverse.T
        block['polygon'] = [[round(float(x),3), round(float(y),3)] for x,y,_ in points]
        block.update(coordinate_space='display_pixels', page_width=preparation['page_width'], page_height=preparation['page_height'])
