import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import os


# ============================================================
# UTILITY: CARI LAYER CONV TERAKHIR
# ============================================================

def get_last_conv_layer_name(model):
    """Mencari nama layer Conv2D terakhir dalam model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Tidak ada layer Conv2D dalam model.")


# ============================================================
# CORE: HITUNG GRAD-CAM HEATMAP
# ============================================================

def make_gradcam_heatmap(img_array, model,
                          last_conv_layer_name,
                          pred_index=None):
    """
    Hitung Grad-CAM heatmap.

    Returns:
        heatmap  : numpy array (H', W'), nilai 0.0–1.0
        pred_idx : kelas yang divisualisasikan
        confidence: confidence score kelas tersebut
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Model bagian 1: input → last conv layer
    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=last_conv_layer.output
    )

    # Model bagian 2: last conv layer → output
    classifier_input = tf.keras.Input(
        shape=last_conv_layer.output.shape[1:]
    )
    x = classifier_input
    found = False
    for layer in model.layers:
        if found:
            x = layer(x)
        if layer.name == last_conv_layer_name:
            found = True
    classifier_model = tf.keras.Model(classifier_input, x)

    # Hitung gradien
    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        preds = classifier_model(conv_output)

        # Ambil kelas prediksi teratas jika tidak ditentukan
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))

        class_score = preds[:, pred_index]

    grads = tape.gradient(class_score, conv_output)

    # Cek gradien valid
    if grads is None:
        raise ValueError(
            "Gradien None — pastikan model tidak pakai "
            "argmax/threshold di output layer."
        )

    # Global average pooling gradien → bobot per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum: bobot × feature map
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: buang negatif
    heatmap = tf.maximum(heatmap, 0)

    # Normalisasi — handle jika semua nilai 0
    heatmap_max = tf.math.reduce_max(heatmap)
    if heatmap_max == 0:
        heatmap = tf.zeros_like(heatmap)
    else:
        heatmap = heatmap / heatmap_max

    # Ambil confidence
    confidence = float(preds[0][pred_index])

    return heatmap.numpy(), pred_index, confidence


# ============================================================
# VISUALISASI: SATU GAMBAR SATU KELAS
# ============================================================

def display_gradcam(img_path, heatmap,
                     pred_index=None,
                     confidence=None,
                     class_names=None,
                     alpha=0.45,
                     save_path=None):
    """
    Tampilkan tiga panel:
    kiri = gambar asli, tengah = heatmap, kanan = overlay.
    """
    # Load gambar asli
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize heatmap ke ukuran gambar asli
    h, w = img_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Terapkan colormap JET
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(
        heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(
        heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay: blend gambar + heatmap
    overlay = cv2.addWeighted(
        img_rgb, 1 - alpha,
        heatmap_colored, alpha, 0
    )

    # Susun judul
    if pred_index is not None and class_names:
        label = class_names[pred_index]
    elif pred_index is not None:
        label = f"Kelas {pred_index}"
    else:
        label = "—"

    conf_str = f"  ({confidence*100:.1f}%)" \
               if confidence is not None else ""

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')

    titles = ['Gambar Asli',
              'Grad-CAM Heatmap',
              f'Overlay — {label}{conf_str}']
    images = [img_rgb, heatmap_resized, overlay]
    cmaps  = [None, 'jet', None]

    for ax, title, im, cmap in zip(axes, titles, images, cmaps):
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, color='white',
                     fontsize=11, pad=8)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    # Colorbar untuk heatmap
    sm = plt.cm.ScalarMappable(
        cmap='jet',
        norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1], shrink=0.8)
    cbar.set_label('Kepentingan',
                    color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.suptitle('Grad-CAM Visualization',
                  color='white', fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Disimpan: {save_path}")

    plt.show()
    plt.close()


# ============================================================
# VISUALISASI: SEMUA KELAS SEKALIGUS
# ============================================================

def display_gradcam_all_classes(img_path, img_array,
                                  model,
                                  last_conv_layer_name,
                                  class_names,
                                  alpha=0.45,
                                  save_path=None):
    """
    Tampilkan Grad-CAM untuk SETIAP kelas sekaligus.
    Berguna untuk face recognition: lihat region mana
    yang membedakan setiap identitas.
    """
    n_classes = len(class_names)

    # Prediksi awal — tampilkan confidence semua kelas
    preds_all = model.predict(img_array, verbose=0)[0]
    pred_top  = int(np.argmax(preds_all))

    # Load gambar
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    # Grid: 2 baris × n_classes kolom
    # Baris 1: heatmap tiap kelas
    # Baris 2: overlay tiap kelas
    fig, axes = plt.subplots(
        2, n_classes,
        figsize=(4 * n_classes, 9)
    )
    fig.patch.set_facecolor('#0f0f23')

    if n_classes == 1:
        axes = axes.reshape(2, 1)

    for cls_idx, cls_name in enumerate(class_names):
        # Hitung Grad-CAM untuk kelas ini
        heatmap, _, conf = make_gradcam_heatmap(
            img_array, model,
            last_conv_layer_name,
            pred_index=cls_idx
        )

        # Resize heatmap
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(
            heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(
            heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(
            img_rgb, 1 - alpha,
            heatmap_colored, alpha, 0
        )

        # Warna judul: hijau jika prediksi teratas, putih lainnya
        title_color = '#00ff88' \
            if cls_idx == pred_top else 'white'
        star = ' ★' if cls_idx == pred_top else ''

        # Baris 1: heatmap
        axes[0, cls_idx].imshow(heatmap_resized, cmap='jet')
        axes[0, cls_idx].set_title(
            f'{cls_name}{star}\n{conf*100:.1f}%',
            color=title_color, fontsize=10
        )
        axes[0, cls_idx].axis('off')
        axes[0, cls_idx].set_facecolor('#0f0f23')

        # Baris 2: overlay
        axes[1, cls_idx].imshow(overlay)
        axes[1, cls_idx].axis('off')
        axes[1, cls_idx].set_facecolor('#0f0f23')

    # Label baris
    axes[0, 0].set_ylabel('Heatmap',
                            color='white', fontsize=11)
    axes[1, 0].set_ylabel('Overlay',
                            color='white', fontsize=11)

    # Legend warna
    patch = mpatches.Patch(
        color='#00ff88',
        label='★ Prediksi teratas'
    )
    fig.legend(handles=[patch],
               loc='lower center',
               ncol=1,
               facecolor='#0f0f23',
               labelcolor='white',
               fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle(
        'Grad-CAM — Semua Kelas\n'
        '(region mana yang diperhatikan untuk tiap identitas?)',
        color='white', fontsize=13
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Disimpan: {save_path}")

    plt.show()
    plt.close()


# ============================================================
# VISUALISASI: MULTI GAMBAR SATU KELAS
# ============================================================

def display_gradcam_multi_images(img_paths, img_arrays,
                                   model,
                                   last_conv_layer_name,
                                   class_names,
                                   alpha=0.45,
                                   save_path=None):
    """
    Jalankan Grad-CAM pada beberapa gambar sekaligus.
    Setiap baris = satu gambar.
    Kolom: asli | heatmap | overlay | label prediksi
    """
    n = len(img_paths)
    fig, axes = plt.subplots(n, 3,
                              figsize=(12, 4.5 * n))
    fig.patch.set_facecolor('#1a1a2e')

    if n == 1:
        axes = axes.reshape(1, 3)

    for row, (img_path, img_array) in enumerate(
            zip(img_paths, img_arrays)):

        # Prediksi
        preds = model.predict(img_array, verbose=0)[0]
        pred_idx  = int(np.argmax(preds))
        pred_conf = preds[pred_idx]
        pred_name = class_names[pred_idx]

        # Grad-CAM untuk kelas prediksi
        heatmap, _, _ = make_gradcam_heatmap(
            img_array, model,
            last_conv_layer_name,
            pred_index=pred_idx
        )

        # Load dan proses gambar
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(
            heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(
            heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(
            img_rgb, 1 - alpha,
            heatmap_colored, alpha, 0
        )

        # Plot
        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(
            os.path.basename(img_path),
            color='white', fontsize=9
        )
        axes[row, 0].axis('off')

        axes[row, 1].imshow(heatmap_resized, cmap='jet')
        axes[row, 1].set_title(
            'Heatmap', color='white', fontsize=9)
        axes[row, 1].axis('off')

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(
            f'Prediksi: {pred_name} '
            f'({pred_conf*100:.1f}%)',
            color='#00ff88', fontsize=9
        )
        axes[row, 2].axis('off')

        # Bar confidence semua kelas (di dalam panel overlay)
        # sebagai teks kecil
        top3_idx  = np.argsort(preds)[::-1][:3]
        info_text = '\n'.join([
            f"{class_names[i]}: {preds[i]*100:.1f}%"
            for i in top3_idx
        ])
        axes[row, 2].text(
            0.02, 0.02, info_text,
            transform=axes[row, 2].transAxes,
            color='white', fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='black', alpha=0.6)
        )

    plt.suptitle('Grad-CAM — Multi Image',
                  color='white', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150,
                    bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"Disimpan: {save_path}")

    plt.show()
    plt.close()


# ============================================================
# MAIN — JALANKAN SEMUA
# ============================================================

if __name__ == "__main__":
    from cnn_face import CNNface

    # --- Setup model ---
    print("Loading model...")
    cnn = CNNface(input_shape=(112, 112, 3), num_classes=5)

    model_path = "face_model.keras"
    if os.path.exists(model_path):
        cnn.load(model_path)
        trained_model = cnn.model
        print(f"Model loaded dari {model_path}")
    else:
        print("Model belum ada, pakai arsitektur kosong...")
        trained_model = cnn.build_lenet5()

    # --- Cari layer conv terakhir ---
    last_conv = get_last_conv_layer_name(trained_model)
    print(f"Layer conv terakhir: {last_conv}")

    # --- Nama kelas ---
    # Sesuaikan dengan urutan kelas di datasetmu
    class_names = [
        'KURNIAWAN',
        'PERSON_2',
        'PERSON_3',
        'PERSON_4',
        'PERSON_5',
    ]

    # --- Path gambar test ---
    img_path = (
        'DATASET_FACES/TESTING/KURNIAWAN/'
        'Screenshot 2025-03-18 at 12.12.05_face.jpg'
    )

    if not os.path.exists(img_path):
        print(f"Gambar tidak ditemukan: {img_path}")
        exit()

    # --- Preprocess gambar ---
    img_raw  = tf.keras.utils.load_img(
        img_path, target_size=(112, 112))
    img_arr  = tf.keras.utils.img_to_array(img_raw)
    img_arr  = np.expand_dims(img_arr, axis=0)
    img_arr  = img_arr / 255.0

    # --- Prediksi awal ---
    preds     = trained_model.predict(img_arr, verbose=0)[0]
    pred_idx  = int(np.argmax(preds))
    pred_conf = preds[pred_idx]
    print(f"\nPrediksi: {class_names[pred_idx]} "
          f"({pred_conf*100:.2f}%)")
    print("Confidence semua kelas:")
    for i, (name, score) in enumerate(
            zip(class_names, preds)):
        marker = ' ← prediksi teratas' \
                 if i == pred_idx else ''
        print(f"  {name:20s}: {score*100:.2f}%{marker}")

    # --- Grad-CAM untuk kelas prediksi teratas ---
    print("\n[1] Grad-CAM kelas prediksi teratas...")
    heatmap, _, conf = make_gradcam_heatmap(
        img_arr, trained_model, last_conv,
        pred_index=pred_idx
    )
    display_gradcam(
        img_path, heatmap,
        pred_index=pred_idx,
        confidence=conf,
        class_names=class_names,
        alpha=0.45,
        save_path='gradcam_top1.png'
    )

    # --- Grad-CAM untuk semua kelas ---
    print("\n[2] Grad-CAM semua kelas...")
    display_gradcam_all_classes(
        img_path, img_arr,
        trained_model,
        last_conv,
        class_names,
        alpha=0.45,
        save_path='gradcam_all_classes.png'
    )

    # --- Multi image (jika ada beberapa gambar test) ---
    test_folder = f'DATASET_FACES/TESTING/KURNIAWAN/'
    if os.path.exists(test_folder):
        test_files = [
            os.path.join(test_folder, f)
            for f in os.listdir(test_folder)
            if f.lower().endswith(
                ('.jpg', '.jpeg', '.png'))
        ][:4]   # ambil maksimal 4 gambar

        if len(test_files) > 1:
            print(f"\n[3] Grad-CAM multi image "
                  f"({len(test_files)} gambar)...")

            img_arrays = []
            for path in test_files:
                im = tf.keras.utils.load_img(
                    path, target_size=(112, 112))
                ia = tf.keras.utils.img_to_array(im)
                ia = np.expand_dims(ia, axis=0) / 255.0
                img_arrays.append(ia)

            display_gradcam_multi_images(
                test_files, img_arrays,
                trained_model,
                last_conv,
                class_names,
                alpha=0.45,
                save_path='gradcam_multi.png'
            )