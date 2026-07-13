import cv2
from tryon.idm_vton_remote import run_tryon

person = cv2.imread("persons/person7.jpg")
result = run_tryon(
    person_image=person,
    garment_image_path="garments/763875967bc787c5b8c8162eb7af0959.jpg",
    garment_description="a pink knit sweater and wide-leg pink pants",
    num_inference_steps=20,  # كان 30
)
cv2.imwrite("result.jpg", result)
print("خلص -- شوفي result.jpg")