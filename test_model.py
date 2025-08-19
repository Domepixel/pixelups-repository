import torch
from torchvision import transforms
from PIL import Image

# Cargar el modelo
modelo = torch.jit.load("modelo_quemaduras_resnet50.pt", map_location=torch.device('cpu'))
modelo.eval()

# Transformación igual que entrenamiento
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Etiquetas
etiquetas = ['Grado 1', 'Grado 2', 'Grado 3', 'No quemadura']

# Cargar imagen
img = Image.open("imagen_prueba.jpg").convert('RGB')  # ← Cambia "tu_imagen.jpg" por el archivo real
input_tensor = transformacion(img).unsqueeze(0)

# Predicción
with torch.no_grad():
    output = modelo(input_tensor)
    probs = torch.nn.functional.softmax(output[0], dim=0)
    pred = torch.argmax(probs).item()
    probabilidad = probs[pred].item()

print(f"Diagnóstico: {etiquetas[pred]} (confianza: {probabilidad:.2f})")

