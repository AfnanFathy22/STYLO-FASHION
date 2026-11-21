import os
import cv2
import numpy as np
from tryon.idm_vton_remote import run_tryon

PERSONS_DIR = "images/persons"
GARMENTS_DIR = "images/garments"
RESULTS_DIR = "results"
CROPS_DIR = "garment_crops"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)


def split_garment_top_bottom(garment_path, top_ratio=0.55):
    """يقص صورة الطقم لجزء فوقاني وتحتاني تقريبيًا حسب top_ratio."""
    img = cv2.imread(garment_path)
    if img is None:
        raise ValueError(f"فشل تحميل: {garment_path}")
    h, w = img.shape[:2]
    split_y = int(h * top_ratio)
    top_crop = img[0:split_y, 0:w]
    bottom_crop = img[split_y:h, 0:w]
    base = os.path.splitext(os.path.basename(garment_path))[0]
    top_path = os.path.join(CROPS_DIR, f"{base}_top.jpg")
    bottom_path = os.path.join(CROPS_DIR, f"{base}_bottom.jpg")
    cv2.imwrite(top_path, top_crop)
    cv2.imwrite(bottom_path, bottom_crop)
    return top_path, bottom_path


def run_single_garment(person_img, garment_path, desc, steps=20):
    """قطعة واحدة بس (فستان أو توب لوحده) — طلب واحد."""
    return run_tryon(
        person_image=person_img,
        garment_image_path=garment_path,
        garment_description=desc,
        num_inference_steps=steps,
    )


def run_full_outfit(person_img, garment_path, top_desc, bottom_desc, top_ratio=0.55, steps=20):
    """فوق وتحت — طلبين متسلسلين، التاني بياخد نتيجة الأول."""
    top_path, bottom_path = split_garment_top_bottom(garment_path, top_ratio)
    step1 = run_tryon(
        person_image=person_img,
        garment_image_path=top_path,
        garment_description=top_desc,
        num_inference_steps=steps,
    )
    step2 = run_tryon(
        person_image=step1,
        garment_image_path=bottom_path,
        garment_description=bottom_desc,
        num_inference_steps=steps,
    )
    return step2


# --------------------------------------------------------------------
# الماتشينج الكامل: mode="single" لقطعة واحدة (فستان/توب لوحده)
#                    mode="split" لطقم فوق+تحت
# top_ratio اتحطت تقريبية، اتأكدي منها بعد أول تشغيل وعدّليها لو محتاج
# --------------------------------------------------------------------
pairs = [
    # ---------------- بنات ----------------
    {"person": "person1.jpg",  "garment": "photo.jpg",   "mode": "single",
     "desc": "black long dress"},
    {"person": "person3.jpg",  "garment": "photo3.jpg",  "mode": "split",
     "top_desc": "cream top", "bottom_desc": "brown pleated skirt", "top_ratio": 0.5},
    {"person": "person5.jpg",  "garment": "photo4.jpg",  "mode": "split",
     "top_desc": "cami top", "bottom_desc": "dark skirt", "top_ratio": 0.5},
    {"person": "person6.jpg",  "garment": "photo5.jpg",  "mode": "single",
     "desc": "dark brown long dress"},
    {"person": "person7.jpg",  "garment": "photo6.jpg",  "mode": "split",
     "top_desc": "black top", "bottom_desc": "black skirt", "top_ratio": 0.5},
    {"person": "person8.jpg",  "garment": "photo7.jpg",  "mode": "split",
     "top_desc": "plaid top", "bottom_desc": "plaid skirt", "top_ratio": 0.5},
    {"person": "person9.jpg",  "garment": "photo8.jpg",  "mode": "single",
     "desc": "brown blazer dress"},
    {"person": "person12.jpg", "garment": "photo9.jpg",  "mode": "split",
     "top_desc": "green cardigan", "bottom_desc": "green wide-leg pants", "top_ratio": 0.5},
    {"person": "person13.jpg", "garment": "photo10.jpg", "mode": "single",
     "desc": "cream cardigan"},
    {"person": "person16.jpg", "garment": "photo11.jpg", "mode": "split",
     "top_desc": "brown coat", "bottom_desc": "brown skirt", "top_ratio": 0.55},
    {"person": "person17.jpg", "garment": "photo13.jpg", "mode": "single",
     "desc": "blush pink robe-style top"},
    {"person": "person18.jpg", "garment": "photo14.jpg", "mode": "single",
     "desc": "cream dress with pink sash"},
    {"person": "person20.jpg", "garment": "photo15.jpg", "mode": "single",
     "desc": "light blue long dress"},

    # ---------------- ولاد ----------------
    {"person": "person11.jpg", "garment": "photo1.jpg",  "mode": "split",
     "top_desc": "black jacket", "bottom_desc": "black pants", "top_ratio": 0.45},
    {"person": "person14.jpg", "garment": "photo2.jpg",  "mode": "single",
     "desc": "long light-colored outfit"},
    {"person": "person15.jpg", "garment": "photo12.jpg", "mode": "split",
     "top_desc": "beige sweater", "bottom_desc": "beige pants", "top_ratio": 0.5},
]

for pair in pairs:
    person_path = os.path.join(PERSONS_DIR, pair["person"])
    garment_path = os.path.join(GARMENTS_DIR, pair["garment"])

    if not os.path.exists(person_path):
        print(f"⚠️ الشخص مش موجود: {person_path}")
        continue
    if not os.path.exists(garment_path):
        print(f"⚠️ القطعة مش موجودة: {garment_path}")
        continue

    person_img = cv2.imread(person_path)
    if person_img is None:
        print(f"⚠️ فشل تحميل: {pair['person']}")
        continue

    print(f"جاري: {pair['person']} + {pair['garment']} ({pair['mode']}) ...")
    try:
        if pair["mode"] == "single":
            result = run_single_garment(person_img, garment_path, pair["desc"])
        else:
            result = run_full_outfit(
                person_img, garment_path,
                pair["top_desc"], pair["bottom_desc"],
                top_ratio=pair.get("top_ratio", 0.55),
            )
    except Exception as e:
        print(f"❌ فشل {pair['person']}: {e}")
        continue

    if result is None:
        print(f"⚠️ result فاضي لـ {pair['person']}")
        continue

    out_path = os.path.join(RESULTS_DIR, f"result_{os.path.splitext(pair['person'])[0]}.jpg")
    cv2.imwrite(out_path, result)
    print(f"✅ اتحفظت: {out_path}")

print("خلصت كل الأشخاص — شوفي فولدر results")