import os
import cv2
import numpy as np
import streamlit as st

from core.card_detector import CardDetector
from core.face_detector import FaceDetector
from core.verifier import FaceVerifier
from core.ocr_extractor import OcrExtractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTO_PATH = os.path.join(BASE_DIR, "models", "deploy.prototxt")
MODEL_PATH = os.path.join(BASE_DIR, "models", "res10_300x300_ssd_iter_140000.caffemodel")


@st.cache_resource
def load_pipeline():
    return (
        CardDetector(),
        FaceDetector(PROTO_PATH, MODEL_PATH),
        FaceVerifier(),
        OcrExtractor(),
    )


def to_cv2_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


st.set_page_config(page_title="Sign In", page_icon="🔐", layout="centered")
st.title("🔐 Sign In")
st.caption("Verify your identity with your ID card to sign in")

card_detector, face_detector, verifier, ocr = load_pipeline()

if "fields" not in st.session_state:
    st.session_state.fields = {"name": "", "place": "", "center": ""}
if "verified" not in st.session_state:
    st.session_state.verified = None
if "raw_ocr_lines" not in st.session_state:
    st.session_state.raw_ocr_lines = []
if "signed_in" not in st.session_state:
    st.session_state.signed_in = False

st.subheader("Identity Verification")

col1, col2 = st.columns(2)
with col1:
    card_file = st.file_uploader("ID card image", type=["jpg", "jpeg", "png"])
with col2:
    camera_frame = st.camera_input("Take your photo now")

compare_disabled = not (card_file and camera_frame)
if st.button("Compare now", disabled=compare_disabled, type="primary"):
    with st.status("Processing...", expanded=True) as status:
        card_img = to_cv2_image(card_file)
        live_img = to_cv2_image(camera_frame)

        status.write("Detecting card boundaries...")
        id_card = card_detector.detect(card_img)
        if id_card is None:
            status.update(label="Couldn't detect the card in the image", state="error")
            st.stop()

        status.write("Detecting the face on the card...")
        card_face = face_detector.detect(id_card)
        if card_face is None:
            status.update(label="Couldn't find a face on the card", state="error")
            st.stop()

        status.write("Detecting your face in the camera...")
        live_face = face_detector.detect(live_img)
        if live_face is None:
            status.update(label="Couldn't find your face in the camera", state="error")
            st.stop()

        status.write("Reading card data...")
        fields, raw_lines = ocr.extract_fields(id_card)
        st.session_state.raw_ocr_lines = raw_lines

        status.write("Comparing faces...")
        verified, distance = verifier.verify(live_face, card_face)

        st.session_state.verified = verified
        st.session_state.fields = fields if verified else {"name": "", "place": "", "center": ""}
        st.session_state.signed_in = False

        if verified:
            status.update(label=f"Verified successfully (distance={distance:.3f})", state="complete")
        else:
            status.update(label=f"Faces do not match (distance={distance:.3f})", state="error")

st.divider()
st.subheader("Your Information")

if st.session_state.verified is False:
    st.error("Verification failed, fields will stay empty")
elif st.session_state.verified is None:
    st.info("No comparison run yet")

name = st.text_input("Name", value=st.session_state.fields["name"], disabled=not st.session_state.verified)
place = st.text_input("Place", value=st.session_state.fields["place"], disabled=not st.session_state.verified)
center = st.text_input("Center", value=st.session_state.fields["center"], disabled=not st.session_state.verified)

st.divider()

if st.session_state.signed_in:
    st.success(f"Signed in successfully as {name}")
else:
    if st.button("Sign In", disabled=not st.session_state.verified, type="primary"):
        st.session_state.signed_in = True
        st.rerun()

if st.session_state.raw_ocr_lines:
    with st.expander("🔍 Raw OCR text (for debugging)"):
        for line in st.session_state.raw_ocr_lines:
            st.text(line)