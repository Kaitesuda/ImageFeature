import base64
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
app = FastAPI()

@app.get('/')
def root():
    return{"hello"}

@app.get('/api/genhog')
async def genhog(image64:Request):
        try:
            json = await image64.json()
            dataImage = json["dataImage"]
            splitImage = dataImage.split(',',1)[1]
            
            decodeSplitImage = base64.b64decode(splitImage)
            npImage = np.frombuffer(decodeSplitImage,np.uint8)
            imageDecode = cv2.imdecode((npImage),cv2.IMREAD_GRAYSCALE)

            resized = cv2.resize(imageDecode,(128,128),cv2.INTER_AREA)
            height,width = resized.shape
            win_size = (height,width)
            cell_size = (8, 8)
            block_size = (16, 16)
            block_stride = (8, 8)
            num_bins = 9
            # Set the parameters of the HOG descriptor using the variablesdefined above
            hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
            cell_size, num_bins)
            # Compute the HOG Descriptor for the gray scale image
            hog_descriptor = hog.compute(resized)
            hogvec = hog_descriptor.tolist()
            return {'HOG': hogvec}
        except Exception as ee:
             print("error ",str(ee))
             raise HTTPException(status_code=500, detail="error")