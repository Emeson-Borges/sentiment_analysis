import streamlit as st
import cv2
import cvlib as cv
from PIL import Image
import numpy as np

def main():
    st.title("Análise de Sentimentos em Vídeos")

    # Formulário para upload de vídeo
    uploaded_file = st.file_uploader("Faça upload de um vídeo", type=["mp4"])

    if uploaded_file is not None:
        # Leitura do vídeo
        video_bytes = uploaded_file.read()
        st.video(video_bytes)

        # Salvar o vídeo temporariamente em um arquivo .mp4
        with open("temp_video.mp4", "wb") as temp_video:
            temp_video.write(video_bytes)

        # Inicia a captura do vídeo
        cap = cv2.VideoCapture("temp_video.mp4")

        # Captura fotos de diferentes expressões faciais
        expressions = ["Alegre", "Feliz", "Triste"]
        expression_count = {expr: 0 for expr in expressions}

        images = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertendo a imagem para o formato RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detectando rostos na imagem
            faces, confidences = cv.detect_face(rgb_frame)

            for idx, face in enumerate(faces):
                (startX, startY, endX, endY) = face

                # Extrai o rosto
                face_img = frame[startY:endY, startX:endX]

                # Redimensiona o rosto para 300x300 (tamanho requerido pelo modelo)
                face_img = cv2.resize(face_img, (300, 300))

                # Converte a imagem para o formato RGB
                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                # Se a confiança da detecção for alta
                if confidences[idx] > 0.5:
                    if len(expressions) > idx:  # Se houver expressões detectadas
                        expression = expressions[idx].capitalize()

                        if expression_count[expression] == 0:
                            expression_count[expression] += 1
                            images.append(face_img_rgb)

        cap.release()

        # Organiza as imagens em uma linha
        st.image(images, caption=expressions, use_column_width=True)

if __name__ == "__main__":
    main()
