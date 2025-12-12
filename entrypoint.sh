# feedback-classifier/entrypoint.sh
#!/bin/bash
set -e

echo "🚀 Iniciando con Backblaze B2..."

# ✅ PASO 1: Instalar PyTorch SOLO desde índice CPU
echo "📦 Instalando PyTorch CPU..."
pip install --no-cache-dir torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# ✅ PASO 2: Instalar otras dependencias desde PyPI normal
echo "📦 Instalando transformers y dependencias..."
pip install --no-cache-dir transformers pandas scikit-learn

# Crear directorio para modelos
mkdir -p model_files

# Verificar si ya existe el modelo
if [ ! -f "model_files/config.json" ] && [ ! -d "model_files/category_model" ]; then
    echo "📥 Descargando modelo desde Backblaze B2..."
    BACKBLAZE_URL="https://f005.backblazeb2.com/file/Modelosml/modelo_categoria_final.zip"
    
    curl -L --fail --retry 3 --retry-delay 10 --max-time 600 \
        --progress-bar \
        -o /tmp/model.zip \
        "$BACKBLAZE_URL" || {
            echo "❌ Error descargando modelo"
            exit 1
        }
    
    echo "📂 Extrayendo modelo..."
    unzip -q /tmp/model.zip -d model_files/ && \
    rm /tmp/model.zip && \
    echo "✅ Modelo descargado y extraído correctamente"
else
    echo "✅ Modelo ya existe (usando caché)"
fi

# Verificar que la aplicación puede arrancar
echo "🔍 Verificando sistema..."
python -c "
import sys
sys.path.append('/app')
try:
    from app.main import app
    print('✅ Aplicación verificada')
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

echo "✅ Todos los sistemas listos"
exec "$@"