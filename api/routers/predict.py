from typing import List

from fastapi import APIRouter, Path, Query, HTTPException
from pydantic import BaseModel

from ..dependancies import blob, model

router = APIRouter()


class Prediction(BaseModel):
    class_name: str = "Boeing 757"
    class_prob: float = 0.92


class PredictionSet(BaseModel):
    predictions: List[Prediction]
    topk: int = 1
    predictor: str = "whatplane"


def prepare_response(probs, class_names, predictor):
    predictions = [
        Prediction(class_name=class_name, class_prob=round(probs[i], 3))
        for i, class_name in enumerate(class_names)
    ]
    return PredictionSet(predictions=predictions, topk=len(predictions), predictor=predictor)


@router.get(
    "/predict/{uuid}",
    response_model=PredictionSet,
    status_code=200,
    tags=["predict"],
    responses={404: {"description": "File not found or is the wrong type"}},
)
async def image_prediction_api(
    uuid: str = Path(
        ..., title="The UUID of the image uploaded by the frontend application", min_length=36
    ),
    topk: int = Query(1, title="The number of classes returned ordered by probability", ge=1, le=5),
):

    try:
        blob_client = blob.ImageBlobClient(uuid)
        image = blob_client.get_uploaded_image()

        imagenet_probs, imagenet_classes, imagenet_likely_class = model.predict_imagenet(image, topk=5)

        # If image is an airliner, predict with airliners model, if not return imagenet
        if "Airliner" not in imagenet_likely_class:
            response = prepare_response(imagenet_probs[:topk], imagenet_classes[:topk], "imagenet")
        else:
            whatplane_probs, whatplane_classes = model.predict_whatplane(image, topk=topk)
            response = prepare_response(whatplane_probs, whatplane_classes, "whatplane")

            # Transfer blob to classified image container
            blob_client.copy_classified_blob(whatplane_classes[0])
    except:
        return HTTPException(status_code=500, detail="Internal Server Error")

    return response

