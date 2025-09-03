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
            response = requests.post(f"{st.secrets["api_url"]}/predict_audio", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prédiction réussie !")
            st.json(result)

            tab3 = st.tabs(["📊 Prédiction graphique"])

            with tab3[0]:
                st.subheader("Prédiction graphique")

                # Récupérer les prédictions (300 valeurs)
                data = result["decisecond_preds"]

                # Créer un axe temporel (chaque point = 0.1s)
                n = len(data)
                t = np.arange(0, n * 0.1, 0.1)

                # Tracé
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.step(t, data, where="mid", color="navy")  # step plot = mieux pour labels discrets
                ax.set_xlabel("Temps (s)")
                ax.set_ylabel("Classe prédite")
                ax.set_title("Prédiction des catégories de l'enregistrement (chaque 0.1s)")
                st.pyplot(fig)


        else:
            st.error(f"Erreur API ({response.status_code})")
            st.text(response.text)





        #if response.status_code == 200:
            #result = response.json()

            #st.success("✅ Prédiction réussie !")
            #st.json(result)

            #tab3 = st.tabs(["📉 Prédiction graphique"])

            #st.subheader("Prédiction graphique")
            #data = result['decisecond_preds']
            #n = len(data)
            #t = np.arange(0,n*0.1,0.1)

            #fig, ax = plt.subplots(figsize=(12, 2.2))
            #ax.plot(t, data, color='navy')
            #ax.set_xlabel("Temps (s)")
            #ax.set_ylabel("Prédiction (label)")
            #ax.set_title("Prédiction des catégories de l'enregistrement")
            #st.pyplot(fig)
