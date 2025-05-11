import cv2 
import os
import time
from datetime import datetime

# Crear carpeta para guardar las imágenes
if not os.path.exists('personas_detectadas'):
    os.makedirs('personas_detectadas')

print("Iniciando programa de detección de personas...")

# Abrir la cámara (DirectShow generalmente funciona mejor en Windows)
print("Abriendo cámara...")
camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Verificar si la cámara se abrió correctamente
if not camara.isOpened():
    print("ERROR: No se pudo abrir la cámara. Verifica que esté conectada y no esté siendo usada por otra aplicación.")
    print("Presiona Enter para salir...")
    input()
    exit()

print("¡Cámara abierta correctamente!")

# Cargar el detector de caras
print("Cargando detector de caras...")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if detector.empty():
    print("ERROR: No se pudo cargar el detector de caras.")
    camara.release()
    print("Presiona Enter para salir...")
    input()
    exit()

print("¡Detector de caras cargado correctamente!")

# Variables para control de capturas
ultima_captura = time.time()
intervalo_entre_capturas = 2  # segundos

print("\n=== PROGRAMA INICIADO ===")
print("* La ventana de la cámara debería abrirse")
print("* Se detectarán caras y se marcarán con un rectángulo verde")
print("* Las imágenes se guardarán en la carpeta 'personas_detectadas'")
print("* Presiona 'q' para salir del programa")

try:
    while True:
        # Capturar imagen de la cámara
        ret, imagen = camara.read()
        
        if not ret:
            print("Error al leer de la cámara. Intentando reconectar...")
            time.sleep(1)
            camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            continue
        
        # Convertir a escala de grises para la detección
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Detectar caras
        caras = detector.detectMultiScale(
            gris,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Dibujar un rectángulo alrededor de cada cara y guardar la imagen
        for (x, y, ancho, alto) in caras:
            # Dibujar el rectángulo
            cv2.rectangle(imagen, (x, y), (x+ancho, y+alto), (0, 255, 0), 2)
            
            # Verificar si ha pasado suficiente tiempo desde la última captura
            tiempo_actual = time.time()
            if tiempo_actual - ultima_captura >= intervalo_entre_capturas:
                # Crear nombre de archivo con fecha y hora
                marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S")
                nombre_archivo = f"personas_detectadas/persona_{marca_tiempo}.jpg"
                
                # Guardar la imagen
                cv2.imwrite(nombre_archivo, imagen)
                print(f"Persona detectada - Imagen guardada como: {nombre_archivo}")
                
                # Actualizar el tiempo de la última captura
                ultima_captura = tiempo_actual
        
        # Mostrar la imagen con las detecciones
        cv2.imshow('Detector de Personas', imagen)
        
        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error inesperado: {e}")

finally:
    # Liberar recursos
    print("Cerrando el programa...")
    camara.release()
    cv2.destroyAllWindows()
    print("Programa terminado.")
    print("Presiona Enter para salir...")
    input()