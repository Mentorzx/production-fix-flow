# 🔧 Configuração de RAM do WSL

## ⚠️ Problema Detectado

Seu WSL está limitado a **8GB RAM** (de 16GB físicos disponíveis).

**Confirmação:**
```bash
free -h
# Mem: 7.6Gi (deveria ser ~15Gi)
```

## ✅ Solução

### 1. Criar arquivo `.wslconfig` no Windows

**Localização:** `C:\Users\Alex\.wslconfig`

**Conteúdo:**
```ini
[wsl2]
memory=14GB       # 16GB total - 2GB para Windows
processors=12     # Usar todos os 12 threads
swap=4GB          # Swap adicional
localhostForwarding=true
```

### 2. Aplicar configuração

No **PowerShell do Windows** (como Administrador):
```powershell
wsl --shutdown
```

Aguarde ~10 segundos e abra o WSL novamente.

### 3. Verificar

No WSL:
```bash
free -h
# Agora deve mostrar ~14GB
```

## 📊 Impacto no PostgreSQL

**Antes (8GB WSL):**
- `shared_buffers = 2GB`
- `effective_cache_size = 6GB`
- `work_mem = 64MB`
- Classificação: `LOW_SPEC`

**Depois (14-16GB WSL):**
- `shared_buffers = 4GB` (25% de 16GB)
- `effective_cache_size = 12GB` (75% de 16GB)
- `work_mem = 128MB`
- Classificação: `MID_SPEC` ✅

## 🔄 Reconfigurar PostgreSQL

Após aumentar a RAM do WSL:

```bash
# 1. Remover configurações antigas
sudo cp /var/lib/pgsql/data/postgresql.conf.backup /var/lib/pgsql/data/postgresql.conf

# 2. Aplicar novas configurações (detectará 14-16GB)
source .venv/bin/activate
python3 -c "from pff.utils.hardware_detector import get_optimal_config, PostgreSQLConfigGenerator; profile, config = get_optimal_config(); print(PostgreSQLConfigGenerator.generate_postgresql_conf(config))" 2>/dev/null | sudo tee -a /var/lib/pgsql/data/postgresql.conf > /dev/null

# 3. Reiniciar PostgreSQL
sudo systemctl restart postgresql
```

## 📝 Notas

- **RAM total física:** 16GB
- **RAM para WSL:** 14GB (recomendado)
- **RAM para Windows:** 2GB (mínimo para sistema operacional)
- **Swap:** 4GB (emergência)

O detector automático de hardware (`pff/utils/hardware_detector.py`) ajustará automaticamente as configurações do PostgreSQL quando a RAM do WSL for aumentada.

---

**Arquivo gerado automaticamente pelo setup do PFF**
**Data:** 2025-10-18
