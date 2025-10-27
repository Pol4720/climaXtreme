# Script para activar el nuevo dashboard
# Ejecutar desde PowerShell en el directorio Tools/src/climaxtreme/dashboard/

Write-Host "🔄 Migrando al nuevo dashboard de climaXtreme..." -ForegroundColor Cyan

# Verificar que estamos en el directorio correcto
$currentDir = Get-Location
if (-not (Test-Path "app.py") -or -not (Test-Path "app_new.py")) {
    Write-Host "❌ Error: Este script debe ejecutarse desde el directorio dashboard/" -ForegroundColor Red
    Write-Host "   Directorio actual: $currentDir" -ForegroundColor Yellow
    exit 1
}

# Hacer backup del dashboard antiguo
Write-Host "📦 Creando backup del dashboard antiguo..." -ForegroundColor Yellow
if (Test-Path "app_old_backup.py") {
    Write-Host "   ℹ️ Ya existe un backup previo (app_old_backup.py)" -ForegroundColor Gray
} else {
    Copy-Item "app.py" "app_old_backup.py"
    Write-Host "   ✅ Backup creado: app_old_backup.py" -ForegroundColor Green
}

# Reemplazar app.py con app_new.py
Write-Host "🔄 Activando nuevo dashboard..." -ForegroundColor Yellow
Copy-Item "app_new.py" "app.py" -Force
Write-Host "   ✅ Nuevo dashboard activado" -ForegroundColor Green

# Verificar que existen las páginas
Write-Host "🔍 Verificando estructura de páginas..." -ForegroundColor Yellow
if (Test-Path "pages") {
    $pageCount = (Get-ChildItem -Path "pages" -Filter "*.py").Count
    Write-Host "   ✅ Directorio pages/ encontrado con $pageCount páginas" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ Advertencia: No se encontró el directorio pages/" -ForegroundColor Red
    Write-Host "   El dashboard puede no funcionar correctamente" -ForegroundColor Red
}

# Verificar utils.py
if (Test-Path "utils.py") {
    Write-Host "   ✅ utils.py encontrado" -ForegroundColor Green
} else {
    Write-Host "   ⚠️ Advertencia: No se encontró utils.py" -ForegroundColor Red
}

Write-Host ""
Write-Host "✨ Migración completada!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Próximos pasos:" -ForegroundColor Cyan
Write-Host "   1. Ejecutar: streamlit run app.py" -ForegroundColor White
Write-Host "   2. Configurar conexión HDFS desde el sidebar" -ForegroundColor White
Write-Host "   3. Explorar las nuevas páginas de análisis" -ForegroundColor White
Write-Host ""
Write-Host "🔙 Para volver al dashboard antiguo:" -ForegroundColor Yellow
Write-Host "   Copy-Item app_old_backup.py app.py -Force" -ForegroundColor Gray
Write-Host ""
Write-Host "📚 Documentación: README_NEW_DASHBOARD.md" -ForegroundColor Cyan
