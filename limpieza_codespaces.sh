#feedback-classifier/limpieza_codespaces.sh
#!/bin/bash
echo "🚨 LIMPIEZA COMPLETA DE CODESPACES (<5% ESPACIO)"

echo "📊 ESPACIO ANTES:"
df -h | grep -E "(overlay|Use%)"

echo "🧹 1. Limpiando archivos según .gitignore..."
rm -rf model_files/ temp_model.zip logs/ 2>/dev/null
find . -name "*.zip" -o -name "*.pyc" -o -name "*.db" -o -name "*.log" -delete 2>/dev/null
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "🐳 2. Limpiando Docker completo..."
docker stop $(docker ps -aq) 2>/dev/null
docker system prune -a --volumes -f >/dev/null 2>&1

echo "🐍 3. Limpiando caché Python/ML..."
pip cache purge >/dev/null 2>&1
rm -rf ~/.cache/huggingface/ ~/.cache/torch/ ~/.cache/pip/ 2>/dev/null

echo "💻 4. Limpiando sistema..."
sudo rm -rf /tmp/* /var/tmp/* 2>/dev/null
sudo apt-get clean >/dev/null 2>&1
rm -rf ~/.cache/* 2>/dev/null

echo "🔧 5. Limpiando VS Code..."
rm -rf ~/.vscode-server/extensions/*/node_modules/ 2>/dev/null
rm -rf ~/.vscode-server/data/logs/* ~/.npm/ 2>/dev/null

echo "📊 ESPACIO DESPUÉS:"
df -h | grep -E "(overlay|Use%)"

USED_PERCENT=$(df | grep overlay | awk '{print $5}' | sed 's/%//')
if [ "$USED_PERCENT" -lt 80 ]; then
    echo "✅ ÉXITO: Espacio liberado correctamente ($USED_PERCENT% usado)"
else
    echo "⚠️ ADVERTENCIA: Aún alto uso de espacio ($USED_PERCENT% usado)"
fi
