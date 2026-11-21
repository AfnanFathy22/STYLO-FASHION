import cv2
from tryon.idm_vton_remote import run_tryon

person = cv2.imread("photo.jpeg")          # صورتك
result = run_tryon(
    person_image=person,
    garment_image_path="garment.jpg",       # صورة الفستان/القميص
    garment_description="a black dress",
)
cv2.imwrite("result.jpg", result)
print("خلص -- شوفي result.jpg")