from face_realtime import FaceRealtime

label_map = {
    'ADEL': 0,
    'HAMDAN': 1,
    'KURNIAWAN': 2,
    'NINO': 3,
    'SANDI': 4
}

app = FaceRealtime(
    model_path="face_model.keras",
    label_map=label_map,
    detect_interval=5 
)

app.run()