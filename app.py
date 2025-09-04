import streamlit as st
import requests
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

# URL de ton API FastAPI déployée sur Cloud Run



st.set_page_config(page_title="Disfluences Demo", layout="centered")

st.title("🎙️ Détection de disfluences - Demo Streamlit")

# Upload de fichier WAV
uploaded_file = st.file_uploader("Choisis un fichier audio (.wav)", type=["wav"])

if uploaded_file is not None:
    # Charger le fichier audio
    y, sr = librosa.load(uploaded_file, sr=12000, mono=True)

    # Playback
    st.audio(uploaded_file, format="audio/wav")

    # 🔹 Tabs pour Waveform et Spectrogramme
    tab1, tab2 = st.tabs(["📉 Waveform", "🎛️ Spectrogramme"])

    with tab1:
        st.subheader("Waveform")
        fig, ax = plt.subplots(figsize=(10, 3))
        times = np.linspace(0, len(y)/sr, num=len(y))
        ax.plot(times, y, color="navy")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Représentation temporelle du signal")
        st.pyplot(fig)

    with tab2:
        st.subheader("Spectrogramme (Mel-scaled)")
        fig, ax = plt.subplots(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr//2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=sr//2, ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Spectrogramme log-Mel")
        st.pyplot(fig)


        # Bouton Predict
    if st.button("🚀 Prédire les disfluences"):
        # Prépare l’audio pour envoi
        files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
        with st.spinner("Envoi au modèle..."):
            response = requests.post(f"{st.secrets['api_url']}/predict_audio", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prédiction réussie !")

            # Tabs pour l'affichage
            tab3 = st.tabs(["📊 Prédiction graphique", "📈 Statistiques globales"])

            # --- Graphique temporel ---
            with tab3[0]:
                st.subheader("Prédiction graphique")

                # Récupérer les prédictions (liste de classes, ex: ["fluent", "EP", "FP/Laugh"])
                preds = result["frame_labels"]

                # Axe temporel (chaque point = 0.1s)
                n = len(preds)
                t = np.arange(0, n * 0.1, 0.1)

                # Ordre manuel des classes (fluent au milieu)
                ordered_classes = ["EP", "fluent", "FP/Laugh"]
                class_to_int = {cls: i for i, cls in enumerate(ordered_classes)}

                # Conversion des prédictions
                y = [class_to_int.get(c, -1) for c in preds]  # -1 si une classe imprévue

                # Plot
                fig, ax = plt.subplots(figsize=(18, 6))
                ax.step(t, y, where="mid", color="navy", linewidth=1.5)

                ax.set_xlabel("Temps (s)", fontsize=12)
                ax.set_ylabel("Classe", fontsize=12)
                ax.set_yticks(list(class_to_int.values()))
                ax.set_yticklabels(list(class_to_int.keys()), fontsize=11)
                ax.set_title("Séquence des prédictions (chaque 0.1s)", fontsize=14, weight="bold")
                ax.grid(True, linestyle="--", alpha=0.6)

                st.pyplot(fig)


            # --- Statistiques globales ---
            with tab3[1]:
                st.subheader("📈 Statistiques globales")

                from collections import Counter
                counts = Counter(preds)
                total = len(preds)
                proportions = {cls: count/total*100 for cls, count in counts.items()}

                # Métriques principales
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total frames (0.1s)", total)
                with col2:
                    st.metric("Nombre de classes détectées", len(counts))

                # Métriques par classe
                for cls, count in counts.items():
                    st.metric(f"{cls}", f"{count} ({proportions[cls]:.1f}%)")

                # Histogramme (bar chart)
                labels = list(counts.keys())
                values = list(counts.values())

                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.bar(labels, values, color="steelblue")
                ax.set_xlabel("Classe")
                ax.set_ylabel("Nombre de frames (0.1s)")
                ax.set_title("Répartition des classes")

                # Afficher les valeurs au-dessus des barres
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(val),
                            ha='center', va='bottom', fontsize=9)

                st.pyplot(fig)

        else:
            st.error(f"Erreur API ({response.status_code})")
            st.text(response.text)









    # Bouton Predict
    #if st.button("🚀 Prédire les disfluences"):
        # Prépare l’audio pour envoi
        #files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
        #with st.spinner("Envoi au modèle..."):
            #response = requests.post(f"{st.secrets["api_url"]}/predict_audio", files=files)

        #if response.status_code == 200:
            #result = response.json()
            #st.success("✅ Prédiction réussie !")
            #st.json(result)

            #tab3 = st.tabs(["📊 Prédiction graphique"])

            #with tab3[0]:
                #st.subheader("Prédiction graphique")

                # Récupérer les prédictions (300 valeurs)
                #data = result["decisecond_preds"]

                # Créer un axe temporel (chaque point = 0.1s)
                #n = len(data)
                #t = np.arange(0, n * 0.1, 0.1)

                # Tracé
                #fig, ax = plt.subplots(figsize=(12, 3))
                #ax.step(t, data, where="mid", color="navy")  # step plot = mieux pour labels discrets
                #ax.set_xlabel("Temps (s)")
                #ax.set_ylabel("Classe prédite")
                #ax.set_title("Prédiction des catégories de l'enregistrement (chaque 0.1s)")
                #st.pyplot(fig)


        #else:
            #st.error(f"Erreur API ({response.status_code})")
            #st.text(response.text)
