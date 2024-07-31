from .nodes.face_parse import FaceParse,FaceParsingResults
from .nodes.face_mesh import FaceMeshParse,FaceMeshResults
from .nodes.face_restore import FaceRestoreCFWithModel,CropFace,FaceRestoreModelLoader,PasteFacesTo
from .nodes.face_mask import TransformTemplateOntoFaceMask


# web ui的节点功能
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "FaceParse_":FaceParse,
    "FaceParseMask_":FaceParsingResults,
    "FaceMeshParse_":FaceMeshParse,
    "FaceMeshMask_":FaceMeshResults,
    "FaceRestoreModelLoader_": FaceRestoreModelLoader,
    "FaceRestore_": FaceRestoreCFWithModel,
    "CropFace_": CropFace, 
    "PasteFacesTo_":PasteFacesTo,
    "TransformTemplateOntoFaceMask_":TransformTemplateOntoFaceMask
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceParse_":"Face Parse",
    "FaceParseMask_":"Face Parse Mask",
    "FaceMeshParse_":"Face Mesh",
    "FaceMeshMask_":"Face Mesh Mask",
    "FaceRestoreModelLoader_": "FaceRestore Model Loader",
    "FaceRestore_": 'Face Restore',
    "CropFace_": 'Crop Face',
    "PasteFacesTo_":'Paste Faces To',
    'TransformTemplateOntoFaceMask_':'TransformTemplateOntoFaceMask'
}