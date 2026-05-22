import requests
import base64
import json
import io
import os
import numpy as np
from PIL import Image
import tempfile
from typing import List
from gradio_client import Client, handle_file



try:
    from config import SOM_ADDRESS, GROUNDING_DINO_ADDRESS, DEPTH_ANYTHING_ADDRESS
except ImportError:
    SOM_ADDRESS = "http://localhost:8080/"
    GROUNDING_DINO_ADDRESS = "http://localhost:8081/"
    DEPTH_ANYTHING_ADDRESS = "http://localhost:8082/"

# som_client = "loaded"
som_client = Client(SOM_ADDRESS)
gd_client = "loaded"
da_client = "loaded"

class AnnotatedImage:
    def __init__(self, annotated_image: Image.Image, original_image: Image.Image = None):
        self.annotated_image = annotated_image
        self.original_image = original_image

def _pil_to_b64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


# def segment_and_mark(image: Image.Image, granularity: float = 1.8, alpha: float = 0.1, anno_mode: list = ['Mask', 'Mark']):
#     img_b64 = _pil_to_b64(image)
#     payload = {"data": [img_b64, granularity, alpha, "Number", anno_mode], "fn_index": 0}
#     try:
#         url = f"{SOM_ADDRESS.rstrip('/')}/run/predict"
#         res = requests.post(url, json=payload, timeout=60).json()
#         if "data" not in res: raise Exception(f"SOM 서버 응답 에러: {res}")
#         output_image = Image.open(io.BytesIO(base64.b64decode(res['data'][0].split(",")[1])))
#         w, h = output_image.size
#         bboxes = [[m['bbox'][0]/w, m['bbox'][1]/h, m['bbox'][2]/w, m['bbox'][3]/h] for m in res['data'][1]]
#         return AnnotatedImage(output_image, image), bboxes
#     except Exception as e:
#         raise Exception(f" [SOM ERROR] {str(e)}")

# def segment_and_mark(image, granularity:float = 1.8, alpha:float = 0.1, anno_mode:list = ['Mask', 'Mark']):
    
#     with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
#         image.save(tmp_file.name, 'JPEG')
#         image = tmp_file.name

#         outputs = som_client.predict(handle_file(image), granularity, alpha, "Number", anno_mode)

#         original_image = Image.open(image)
#         output_image = Image.open(outputs[0])
        
#         output_image = AnnotatedImage(output_image, original_image)
        
#         w,h = output_image.annotated_image.size
                
#         masks = outputs[1]
        
#         bboxes = []
        
#         for mask in masks:
#             bbox = mask['bbox']
#             bboxes.append((bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h))
        
#     return output_image, bboxes

import os
import tempfile
from PIL import Image
from gradio_client import handle_file

def segment_and_mark(image, granularity:float = 1.8, alpha:float = 0.1, anno_mode:list = ['Mask', 'Mark']):
    # 1. 파일이 전송되기 전에 지워지는 것을 방지하기 위해 delete=False 설정
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    
    try:
        # 2. 이미지 저장 후 핸들을 닫아야 Gradio Client가 파일을 읽을 수 있음
        image.save(tmp_path, 'JPEG')
        tmp.close() 

        # 3. fn_index=0을 사용하여 첫 번째 메인 함수를 호출함
        # 스크린샷에 보이는 입력 순서대로 데이터를 전달합니다.
        outputs = som_client.predict(
            handle_file(tmp_path), # 1. 이미지 (드롭박스)
            granularity,           # 2. granularity (넘버)
            alpha,                 # 3. alpha (넘버)
            "Number",              # 4. label_mode (라디오)
            anno_mode,             # 5. anno_mode (체크박스)
            fn_index=0             # 핵심: 에러를 방지하기 위해 0번 인덱스 명시
        )

        # 4. 결과 검증 (에러 발생 시 여기서 멈춤)
        if not outputs or len(outputs) < 2:
            raise ValueError(f"서버가 예상치 못한 응답을 줬습니다: {outputs}")

        output_image_path = outputs[0]
        masks = outputs[1]

        original_image = image.copy()
        output_image = Image.open(output_image_path).convert("RGB")
        
        # 사용자님의 AnnotatedImage 클래스로 래핑
        output_obj = AnnotatedImage(output_image, original_image)
        
        w, h = output_obj.annotated_image.size
        bboxes = []
        for mask in masks:
            bbox = mask.get('bbox', [0, 0, 0, 0])
            # 0~1 사이의 정규화된 좌표로 변환
            bboxes.append((bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h))
            
        return output_obj, bboxes

    except Exception as e:
        # 에러 발생 시 상세 내용을 출력하고 에이전트가 알 수 있게 raise함
        print(f"SOM 실행 중 에러 발생: {e}")
        raise e
    
    finally:
        # 5. 작업이 끝나면 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def detection(image: Image.Image, objects: List[str], box_threshold: float = 0.35, text_threshold: float = 0.25):
    img_b64 = _pil_to_b64(image)
    payload = {"data": [img_b64, ', '.join(objects), box_threshold, text_threshold], "fn_index": 0}
    try:
        url = f"{GROUNDING_DINO_ADDRESS.rstrip('/')}/api/predict/"
        res = requests.post(url, json=payload, timeout=60).json()
        if "data" not in res: raise Exception(f"DINO 서버 응답 에러: {res}")
        det_img_data = base64.b64decode(res['data'][0].split(",")[1])
        output_image = Image.open(io.BytesIO(det_img_data))
        processed_boxes = [[b[0]-b[2]/2, b[1]-b[3]/2, b[2], b[3]] for b in res['data'][1]['boxes']]
        return AnnotatedImage(output_image, image), processed_boxes
    except Exception as e:
        raise Exception(f" [DINO ERROR] {str(e)}")

def depth(image: Image.Image):
    img_b64 = _pil_to_b64(image)
    payload = {"data": [img_b64], "fn_index": 0}
    try:
        url = f"{DEPTH_ANYTHING_ADDRESS.rstrip('/')}/api/predict/"
        res = requests.post(url, json=payload, timeout=60).json()
        if "data" not in res: return image
        return Image.open(io.BytesIO(base64.b64decode(res['data'][0].split(",")[1])))
    except Exception as e:
        print(f"[Depth error] {e}")
        return image


def crop_image(image, x:float, y:float, width:float, height:float):
    w, h = image.size
    return image.crop((max(0, x)*w, max(0, y)*h, min(1, x+width)*w, min(1, y+height)*h))

def zoom_in_image_by_bbox(image, box, padding=0.05):
    x, y, w, h = box
    return crop_image(image, x-padding, y-padding, w+2*padding, h+2*padding)

def sliding_window_detection(image: Image.Image, objects):
    box_width, box_height = 1/3, 1/3
    possible_patches, possible_boxes = [], []
    for x in np.arange(0, 7/9, 2/9):
        for y in np.arange(0, 7/9, 2/9):
            cropped_img = crop_image(image, x, y, box_width, box_height)
            ann_img, det_boxes = detection(cropped_img, objects)
            if det_boxes:
                possible_patches.append(ann_img)
                possible_boxes.append(det_boxes)
    return possible_patches, possible_boxes

def overlay_images(background_img, overlay_img, alpha=0.3, bounding_box=[0, 0, 1, 1]):
    bg_w, bg_h = background_img.size
    x, y, w, h = [int(v * s) for v, s in zip(bounding_box, [bg_w, bg_h, bg_w, bg_h])]
    overlay_resized = overlay_img.resize((w, h), Image.Resampling.LANCZOS).convert('RGBA')
    overlay_resized.putalpha(int(255 * alpha))
    new_img = Image.new('RGBA', background_img.size, (255, 255, 255, 255))
    new_img.paste(background_img, (0,0))
    new_img.paste(overlay_resized, (x, y, x + w, y + h), overlay_resized)
    return new_img.convert('RGB')