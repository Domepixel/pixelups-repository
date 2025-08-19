from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
import os
import torch
from torchvision import transforms
from PIL import Image as PILImage, UnidentifiedImageError
import datetime
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename
from supabase_client import supabase
#aqui nuevo reportlab
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle


def crear_informe_pdf(nombre, edad, sexo, sintomas, resultados, ruta_guardado_pdf, carpeta_imagenes):
    doc = SimpleDocTemplate(ruta_guardado_pdf, pagesize=letter,
                            rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)

    elementos = []

    styles = getSampleStyleSheet()
    estilo_titulo = styles['Title']
    estilo_normal = styles['Normal']

    # Agregar logo
    try:
        logo_path = 'static/img/logo_Pixel.png'  # Ajusta la ruta si es necesario
        logo = Image(logo_path, width=3.5*inch, height=1*inch)
        logo.hAlign = 'CENTER'
        elementos.append(logo)
    except Exception as e:
        print("No se pudo cargar el logo:", e)

    elementos.append(Spacer(1, 12))

    # Título
    titulo = Paragraph(f"Informe del Paciente: {nombre}", estilo_titulo)
    elementos.append(titulo)
    elementos.append(Spacer(1, 12))

    # Datos del paciente
    datos = f"""
    <b>Edad:</b> {edad} <br/>
    <b>Sexo:</b> {sexo} <br/>
    <b>Síntomas:</b> {sintomas} <br/>
    <b>Fecha:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    elementos.append(Paragraph(datos, estilo_normal))
    elementos.append(Spacer(1, 24))

    # Tabla con resultados
    data = [['Archivo de Imagen', 'Diagnóstico']]
    for res in resultados:
        data.append([res['archivo'], res['diagnostico']])

    tabla = Table(data, colWidths=[3*inch, 3*inch])
    tabla.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0d6efd')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 12),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
    ]))

    elementos.append(tabla)
    elementos.append(Spacer(1, 20))

    # Mostrar imágenes subidas
    for res in resultados:
        imagen_path = os.path.join(carpeta_imagenes, res['archivo'])
        try:
            img_obj = Image(imagen_path, width=3*inch, height=3*inch)
            img_obj.hAlign = 'CENTER'
            elementos.append(img_obj)
            elementos.append(Spacer(1, 12))
        except Exception as e:
            print(f"No se pudo cargar la imagen {imagen_path}: {e}")

    # Pie de página
    creditos = """
    Proyecto desarrollado por estudiantes de Ingeniería Biomédica<br/>
    Universidad Politécnica Salesiana<br/>
    Tutor del proyecto: Ing. Roberto Bayas<br/>
    Estudiantes Investigadoras: Doménica Navarrete, Daniela Alarcón
    """
    elementos.append(Paragraph(creditos, estilo_normal))

    doc.build(elementos)

#hasta aquí lo nuevo

app = Flask(__name__)
app.secret_key = "secret_key_para_flash_messages"
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# Asegura la carpeta de subidas
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Modelo TorchScript
modelo = torch.jit.load("modelo_quemaduras_resnet50.pt", map_location=torch.device('cpu'))
modelo.eval()

# Transformación
transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Etiquetas según entrenamiento
etiquetas = ['Grado 1', 'Grado 2', 'Grado 3', 'No quemadura']

# Decorador para proteger rutas que requieren login
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("Debes iniciar sesión para acceder a esta página.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/")
def index():
    user = session.get("user")
    return render_template("index.html", user=user)

@app.route("/modelos")
@login_required
def modelos():
    return render_template("modelos.html", user=session.get('user'))

@app.route("/modelo/<nombre>")
@login_required
def modelo_detalle(nombre):
    return render_template("modelo_detalle.html", nombre_modelo=nombre, user=session.get('user'))

@app.route("/procesar", methods=["POST"])
@login_required
def procesar():
    nombre = request.form.get("nombre", "").strip()
    edad = request.form.get("edad", "").strip()
    sexo = request.form.get("sexo", "").strip()
    sintomas = request.form.get("sintomas", "").strip()
    archivos = request.files.getlist("imagenes")

    if not archivos or archivos[0].filename == '':
        flash("Debe subir al menos una imagen.")
        return redirect(request.referrer)

    resultados = []

    for archivo in archivos[:10]:
        filename = secure_filename(archivo.filename)
        ruta_guardada = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            archivo.save(ruta_guardada)
            img = PILImage.open(ruta_guardada).convert('RGB')
            input_tensor = transformacion(img).unsqueeze(0)

            with torch.no_grad():
                output = modelo(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                pred = torch.argmax(probs).item()
                probabilidad = probs[pred].item()

            diagnostico = f"{etiquetas[pred]} (confianza: {probabilidad:.2f})"

            resultados.append({
                "archivo": filename,
                "diagnostico": diagnostico
            })

            supabase.table("imagenes_diagnostico").insert({
                "nombre": nombre,
                "edad": edad,
                "sexo": sexo,
                "sintomas": sintomas,
                "imagen": filename,
                "diagnostico": etiquetas[pred],
                "confianza": round(probabilidad, 2),
                "fecha": datetime.datetime.now().isoformat(),
                "usuario_email": session.get('user')
            }).execute()

        except UnidentifiedImageError:
            resultados.append({
                "archivo": filename,
                "diagnostico": "Archivo no reconocido como imagen"
            })
        except Exception as e:
            resultados.append({
                "archivo": filename,
                "diagnostico": f"Error procesando la imagen: {str(e)}"
            })

    # Crear PDF profesional con reportlab.platypus
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    informe_path = os.path.join(app.config['UPLOAD_FOLDER'], f"informe_{nombre}_{timestamp}.pdf")

    crear_informe_pdf(nombre, edad, sexo, sintomas, resultados, informe_path, app.config['UPLOAD_FOLDER'])

    pdf_filename = f"{nombre}_{timestamp}.pdf"
    with open(informe_path, "rb") as f:
        supabase.storage.from_("informes-pdf").upload(pdf_filename, f)
        #supabase.storage.from_("informes-pdf").upload(pdf_filename, f, content_type="application/pdf")

    # Actualizar la tabla con el nombre del archivo PDF
    supabase.table("imagenes_diagnostico")\
        .update({"pdf_filename": pdf_filename})\
        .eq("usuario_email", session.get('user'))\
        .eq("nombre", nombre)\
        .execute()

    # Obtener URL pública (opcional para debug o usarla luego)
    url_publica = supabase.storage.from_("informes-pdf").get_public_url(pdf_filename)
    print("URL del informe:", url_publica)

    return send_file(informe_path, as_attachment=True)


    # Subir PDF
    pdf_filename = f"{nombre}_{timestamp}.pdf"
    
    with open(informe_path, "rb") as f:
     supabase.storage.from_("informes-pdf").upload(pdf_filename, f)

    url_publica = supabase.storage.from_("informes-pdf").get_public_url(pdf_filename)
    print("URL del informe:", url_publica)

    return send_file(informe_path, as_attachment=True)

@app.route("/historial")
@login_required
def historial():
    try:
        respuesta = supabase.table("imagenes_diagnostico").select("*").eq("usuario_email", session.get('user')).order("fecha", desc=True).limit(20).execute()
        historial = respuesta.data
    except Exception:
        historial = []

    return render_template("historial.html", historial=historial, user=session.get('user'))

# LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        result = supabase.auth.sign_in_with_password({"email": email, "password": password})

        if result.user:
            session["user"] = result.user.email
            flash("Inicio de sesión exitoso.")
            return redirect(url_for("index"))
        else:
            flash("Credenciales incorrectas o error al iniciar sesión.")
            return redirect(url_for("login"))
    
    return render_template("login.html")

# REGISTRO
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        result = supabase.auth.sign_up({"email": email, "password": password})

        if result.user:
            flash("Registro exitoso. Revisa tu correo para confirmar y luego inicia sesión.")
            return redirect(url_for("login"))
        else:
            flash("Error al registrarse.")
            return redirect(url_for("register"))
    
    return render_template("register.html")

# LOGOUT
@app.route("/logout")
@login_required
def logout():
    session.pop("user", None)
    flash("Has cerrado sesión correctamente.")
    return redirect(url_for("index"))

#if __name__ == "__main__":
    #app.run(debug=True)

    if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    # host 0.0.0.0 para que Render pueda ver tu servidor
    app.run(host="0.0.0.0", port=port)
