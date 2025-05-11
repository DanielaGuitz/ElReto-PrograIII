# 🎯 EL RETO - Sistema de Detección de Personas

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema de detección de personas utilizando visión por computadora. El sistema captura imágenes a través de la webcam, identifica personas en tiempo real, y guarda automáticamente las capturas en una carpeta dedicada llamada "personas_detectadas".

## 🛠 Tecnologías Utilizadas

- *Python* - Lenguaje de programación principal
- *OpenCV* - Para captura de imágenes a través de la webcam
- *Hugging Face* - API para procesamiento de imágenes
- *Modelo DETR* - Para la detección de objetos/personas

## 👨‍💻 Equipo de Desarrollo

### 👩‍💻 Daniela Maricler Guitz Palma (0905-23-15374)

*Contribuciones:*
- Investigación exhaustiva de APIs disponibles (Clarifai, Hugging Face, Roboflow, Azure Computer Vision)
- Análisis comparativo de planes gratuitos, facilidad de uso y compatibilidad con Python
- Selección y recomendación de Hugging Face como la API más adecuada para el proyecto
- Investigación de métodos para captura de imágenes con webcam en Python
- Documentación del proceso de investigación y decisiones tomadas

### 👨‍💻 Leonel Andrés Guerra Godoy (0905-23-3929)


*Contribuciones:*
- Investigación de la estructura organizativa del proyecto
- Identificación de librerías necesarias para el correcto funcionamiento
- Investigación de dependencias requeridas
- Solución de problemas de compatibilidad con Python y pip en Windows
- Modificación de comandos para garantizar la correcta ejecución en diferentes entornos

### 👨‍💻 Benedicto de Jesús Martinez Martinez (0905-23-10445)


*Contribuciones:*
- Desarrollo completo del código del proyecto
- Creación del archivo principal "detector_personas.py"
- Implementación de la integración con APIs y dependencias
- Desarrollo del sistema de detección con cámara
- Implementación del sistema de almacenamiento de capturas en la carpeta "personas_detectadas"

## 🚀 Instrucciones de Instalación

bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/el-reto-deteccion-personas.git
cd el-reto-deteccion-personas

# Instalar dependencias
pip install opencv-python
pip install transformers
pip install torch
pip install pillow


## 📝 Uso del Sistema

bash
# Ejecutar el detector de personas
python detector_personas.py


El sistema activará la cámara web, comenzará a detectar personas en tiempo real y guardará automáticamente las capturas cuando detecte personas en el campo visual.

## 👨‍🏫 Información del Curso

- *Universidad:* Universidad Mariano Gálvez de Guatemala, Campus Jutiapa
- *Facultad:* Ingeniería en Sistemas de Información
- *Curso:* Programación III
- *Docente:* Ing. Ruldyn Ayala

