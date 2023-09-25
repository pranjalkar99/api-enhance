import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image as PILImage
import os
import boto3
import requests
from io import BytesIO
import base64

import model
from fastapi import FastAPI, UploadFile, Form, File
import rawpy

app = FastAPI()  # FastAPI App

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_filepath = "./pretrained_models/adobe_dpe/curl_validpsnr_23.073045286204017_validloss_0.0701291635632515_testpsnr_23.584083321292365_testloss_0.061363041400909424_epoch_510_model.pt"
output_folder = "test_output"


s3 = boto3.client('s3')
s3_resource = boto3.resource('s3')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if not os.path.exists(checkpoint_filepath):
    raise Exception(f"Model checkpoint file '{checkpoint_filepath}' not found.")

net = model.CURLNet()
checkpoint = torch.load(checkpoint_filepath, map_location=DEVICE)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()
net.to(DEVICE)


def load_image(image_data):
    '''
    Load image from the given data
    '''
    try:
        img = PILImage.open(BytesIO(image_data))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        input_img = np.array(img).astype(np.uint8)
        return TF.to_tensor(input_img).to(DEVICE)
    except Exception as e:
        raise Exception(f"Error loading image: {e}")

def remove_junk():
    output_folder='./test_output'
    try:
        if os.path.exists(output_folder):
            for filename in os.listdir(output_folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    os.remove(os.path.join(output_folder, filename))
                else:
                    print("clean up not required")
    except Exception as e:
        print(f"Error removing junk: {e}")

@app.post('/enhance-single')
async def enhance(img: UploadFile = File(...), name: str = Form(...)):
    """
    Infer and return the enhanced image
    """
    try:
        remove_junk()
        ext = name.split('.')[-1].lower()
        if ext not in ['jpg', 'jpeg', 'png', 'dng']:
            return {"error": f"Unsupported file format: {ext}"}, 400

        img_data = await img.read()
        if ext == 'dng':
            with rawpy.imread(BytesIO(img_data)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
                img_data = PILImage.fromarray(rgb).tobytes()

        img_tensor = load_image(img_data)
    except Exception as e:
        return {"error": f"Image Reading error: {e}"}, 400

    try:
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = torch.clamp(img_tensor, 0, 1)

            net_output, _ = net(img_tensor)
            output_img = (net_output.squeeze(0).data.cpu().numpy() * 255).astype('uint8')
            output_img = PILImage.fromarray(output_img.transpose(1, 2, 0))

            output_filename = f"{name.replace('.', '_enhanced.')}"
            output_path = os.path.join(output_folder, output_filename)

            img_format = "JPEG" if ext == "jpg" else ext.upper()
            output_img.save(output_path, format=img_format)

            with BytesIO() as buffer:
                output_img.save(buffer, format=img_format)
                buffer.seek(0)
                data = base64.b64encode(buffer.read()).decode()

            return {'img': data, "name": name.replace('.', '_enhanced.')}
    except Exception as e:
        error_msg = f"Inference error: {str(e)}"
        return {"error": error_msg}, 400

@app.post('/enhance-bucket')
async def enhance_bucket(bucket_link: str):
    """
    Download images from the given bucket link, apply enhancement, store in another bucket, and delete the files
    """
    try:
        # Download images from the given bucket link
        bucket_name = bucket_link.split('/')[2]
        prefix = '/'.join(bucket_link.split('/')[3:])
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)['Contents']
        for obj in objects:
            key = obj['Key']
            ext = key.split('.')[-1].lower()
            if ext not in ['jpg', 'jpeg', 'png', 'dng']:
                continue
            response = s3.get_object(Bucket=bucket_name, Key=key)
            img_data = response['Body'].read()

            # Apply enhancement
            # ... your enhancement code here ...

            # Store the enhanced image in another bucket
            output_bucket_name = 'enhanced-images'
            output_key = f"{os.path.basename(key).replace('.', '_enhanced.')}"
            s3_resource.Bucket(output_bucket_name).put_object(Key=output_key, Body=img_data)

            # Delete the original image
            s3.delete_object(Bucket=bucket_name, Key=key)

        return {"message": "Images enhanced and stored successfully."}
    except Exception as e:
        return {"error": f"Error enhancing images: {e}"}, 400

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
