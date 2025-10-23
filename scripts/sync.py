import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def bootstrap_dependencies():
    """
    Checks and installs essential dependencies required for this script to run.
    """
    try:
        import tomlkit  # noqa: F401
        return
    except ModuleNotFoundError:
        print(
            "⚠️  Dependência 'tomlkit' não encontrada. Tentando instalar e carregar dinamicamente..."
        )

        base_python_executable = sys.executable

        try:
            print("   - Instalando 'tomlkit' com pip...")
            subprocess.run(
                [base_python_executable, "-m", "pip", "install", "tomlkit"],
                check=True,
                capture_output=True,
                text=True,
            )
            print("   - Instalação concluída.")
            print("   - Localizando o diretório do pacote...")
            result = subprocess.run(
                [
                    base_python_executable,
                    "-c",
                    "import site; print(site.getsitepackages()[0])",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            site_packages_path = result.stdout.strip()
            print(
                f"   - Adicionando '{site_packages_path}' ao caminho de busca do Python."
            )
            sys.path.append(site_packages_path)

            print("✅ 'tomlkit' instalado e carregado com sucesso para esta execução.")

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Falha crítica durante o bootstrap do 'tomlkit'.", file=sys.stderr)
            print(
                "   Por favor, instale manualmente no seu Python principal: pip install tomlkit",
                file=sys.stderr,
            )
            sys.exit(1)


bootstrap_dependencies()

import tomlkit  # noqa: E402


class PoetrySync:
    # Windows-specific libraries
    WINDOWS_PACKAGES = {
        "pywin32",
        "pywinauto",
        "win32com",
        "win32gui",
        "pypiwin32",
        "windows-curses",
        "pywinpty",
    }

    # Unix/Linux-specific libraries
    UNIX_PACKAGES = {
        "python-daemon",
        "pyinotify",
        "python-prctl",
        "systemd-python",
        "readline",
    }

    # macOS-specific libraries
    MACOS_PACKAGES = {"pyobjc", "pyobjc-core", "macfsevents"}

    # PyTorch core packages that need PyTorch source
    PYTORCH_CORE_PACKAGES = {
        "torch",
        "torchvision", 
        "torchaudio",
    }

    # PyTorch ecosystem packages that are on PyPI (not PyTorch source)
    PYTORCH_ECOSYSTEM_PACKAGES = {
        "torch-geometric",
    }

    # PyTorch extension packages that need special pip installation
    PYTORCH_EXTENSION_PACKAGES = {
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    }

    # Packages that should be installed via pip due to build issues
    PIP_ONLY_PACKAGES = {
        "pysimdjson",
        "torch-scatter",
        "torch-sparse",
        "torch-cluster",
        "torch-spline-conv",
    }

    def _robust_rmtree(self, path: Path, max_attempts: int = 3) -> bool:
        """Robustly remove a directory tree, handling Windows file locking issues"""
        if not path.exists():
            return True
            
        print(f"   🗑️  Removendo {path.name}...")
        
        for attempt in range(max_attempts):
            try:
                import shutil
                shutil.rmtree(path)
                print(f"   ✅ {path.name} removido com sucesso")
                return True
                
            except PermissionError as e:
                print(f"   ⚠️  Tentativa {attempt + 1}: {e}")
                
                if attempt < max_attempts - 1:
                    # Try different strategies based on platform
                    if self.platform == "windows":
                        # Use aggressive mode for force-clean (max_attempts > 3)
                        aggressive = max_attempts > 3 and attempt >= 1
                        self._try_windows_unlock(path, aggressive)
                    
                    # Wait a bit before retrying
                    wait_time = 2 ** attempt
                    print(f"   ⏳ Aguardando {wait_time} segundos...")
                    time.sleep(wait_time)
                else:
                    # Last resort: rename and let it be removed later
                    try:
                        backup_name = f"{path.name}_backup_{int(time.time())}"
                        backup_path = path.parent / backup_name
                        path.rename(backup_path)
                        print(f"   🔄 {path.name} renomeado para {backup_name}")
                        print("   💡 Será removido automaticamente em futuras execuções")
                        print("   🔍 Dica: Feche VS Code, PyCharm ou outros editores se estiverem abertos")
                        return True
                    except Exception as rename_error:
                        print(f"   ❌ Falha ao renomear: {rename_error}")
                        print("   💡 Tente:")
                        print("      • Fechar todos os editores (VS Code, PyCharm, etc.)")
                        print("      • Encerrar processos Python: taskkill /F /IM python.exe")
                        print("      • Executar como administrador")
                        print("      • Reiniciar o terminal")
                        return False
            except Exception as e:
                print(f"   ❌ Erro inesperado: {e}")
                return False
        
        return False

    def _try_windows_unlock(self, path: Path, aggressive: bool = False):
        """Try to unlock files on Windows"""
        print("   🔓 Tentando desbloquear arquivos no Windows...")
        
        # Kill any Python processes that might be using the venv
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.split('\n')[1:]:  # Skip header
                if line.strip() and str(path) in line:
                    # Extract PID
                    parts = line.split(',')
                    if len(parts) >= 2:
                        pid = parts[1].strip('"')
                        try:
                            subprocess.run(["taskkill", "/F", "/PID", pid], 
                                         capture_output=True, check=True)
                            print(f"   🔫 Processo Python {pid} encerrado")
                        except subprocess.CalledProcessError:
                            pass
                            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # tasklist/taskkill not available or failed
            pass
        
        # Try to make files writable
        try:
            if self.platform == "windows":
                subprocess.run(
                    ["attrib", "-R", f"{path}\\*.*", "/S"],
                    capture_output=True,
                    check=False  # Don't fail if this doesn't work
                )
                print("   📝 Atributos de arquivo atualizados")
        except (subprocess.CalledProcessError, FileNotFoundError):
            # attrib command not available or failed
            pass
    def _cleanup_old_backups(self):
        """Clean up old backup directories from previous failed removals"""
        print("   🧹 Limpando backups antigos...")
        
        backup_pattern = re.compile(r"\.venv_backup_\d+")
        cleaned = 0
        
        for item in self.project_root.iterdir():
            if item.is_dir() and backup_pattern.match(item.name):
                try:
                    import shutil
                    shutil.rmtree(item)
                    print(f"   🗑️  {item.name} removido")
                    cleaned += 1
                except Exception as e:
                    print(f"   ⚠️  Não foi possível remover {item.name}: {e}")
        
        if cleaned > 0:
            print(f"   ✅ {cleaned} backup(s) antigo(s) removido(s)")
        else:
            print("   ✅ Nenhum backup antigo encontrado")

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.pyproject_path = self.project_root / "pyproject.toml"
        self.requirements_path = self.project_root / "requirements.txt"
        self.custom_reqs_path = self.project_root / "requirements.custom.txt"
        self.poetry_lock_path = self.project_root / "poetry.lock"
        self.platform = self.detect_platform()

        # Detect if we are in a venv and get the base Python
        self.base_python = self.get_base_python()
        
        # Initialize storage for different dependency types
        self.pytorch_deps: Dict[str, str] = {}
        self.pip_deps: Dict[str, str] = {}
        self.git_deps: Dict[str, str] = {}
        self.special_deps: List[str] = []
        
        # Runtime flags
        self._force_emergency: bool = False
        self._poetry_failed: bool = False

    def get_base_python(self) -> str:
        """Get the base Python (outside the venv)"""
        if hasattr(sys, "real_prefix") or (
            hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
        ):
            # We are in a venv
            if self.platform == "windows":
                # On Windows, look for python.exe in PATH
                try:
                    result = subprocess.run(
                        ["where", "python"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        pythons = result.stdout.strip().split("\n")
                        # Filter Python that is not in the current venv
                        for python_path in pythons:
                            if "env" not in python_path and "venv" not in python_path:
                                return python_path
                except Exception:
                    pass
            # Fallback: use python3 or python
            return "python3" if self.platform != "windows" else "python"
        return sys.executable

    def detect_platform(self) -> str:
        """Detect the current platform"""
        system = platform.system().lower()
        if system == "windows":
            return "windows"
        elif system == "darwin":
            return "macos"
        else:
            return "linux"

    def is_platform_specific(self, package_name: str) -> bool:
        """Check if the package is platform-specific"""
        pkg_lower = package_name.lower()

        # Check common patterns
        if any(pattern in pkg_lower for pattern in ["win32", "windows", "pywin"]):
            return True
        if any(pattern in pkg_lower for pattern in ["linux", "unix", "posix"]):
            return True
        if any(pattern in pkg_lower for pattern in ["darwin", "macos", "objc"]):
            return True

        # Check specific lists
        return pkg_lower in (
            self.WINDOWS_PACKAGES | self.UNIX_PACKAGES | self.MACOS_PACKAGES
        )

    def should_include_package(self, package_name: str) -> bool:
        """Determine if the package should be included on the current platform"""
        pkg_lower = package_name.lower()

        # If not platform-specific, include
        if not self.is_platform_specific(pkg_lower):
            return True

        # Check compatibility with current platform
        if self.platform == "windows" and pkg_lower in self.WINDOWS_PACKAGES:
            return True
        elif self.platform == "linux" and pkg_lower in self.UNIX_PACKAGES:
            return True
        elif self.platform == "macos" and pkg_lower in self.MACOS_PACKAGES:
            return True

        return False

    def ensure_poetry(self):
        """Ensure Poetry is installed"""
        try:
            subprocess.run(["poetry", "--version"], check=True, capture_output=True)
            print("✅ Poetry encontrado")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("📦 Instalando Poetry...")
            if self.platform == "windows":
                subprocess.run(
                    [self.base_python, "-m", "pip", "install", "poetry"], check=True
                )
            else:
                # Use official installer
                subprocess.run(
                    "curl -sSL https://install.python-poetry.org | python3 -",
                    shell=True,
                    check=True,
                )

    def get_python_version(self) -> str:
        """Get current Python version"""
        version_info = sys.version_info
        return f">={version_info.major}.{version_info.minor},<4.0"

    def parse_requirement(self, line: str) -> Optional[Tuple[str, str, str]]:
        """Parse a requirements line"""
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        # Patterns for different formats
        patterns = [
            # package==version
            (r"^([a-zA-Z0-9\-_\.]+)\s*==\s*([^\s;#]+)", "normal"),
            # package>=version,<version
            (r"^([a-zA-Z0-9\-_\.]+)\s*([><=~]+[^\s;#]+)", "normal"),
            # git+https://...
            (r"^([a-zA-Z0-9\-_]+)\s*@\s*(git\+[^\s;#]+)", "git"),
            # -e git+https://...
            (r"^-e\s+(git\+[^\s;#]+)#egg=([a-zA-Z0-9\-_]+)", "git_editable"),
        ]

        for pattern, dep_type in patterns:
            match = re.match(pattern, line)
            if match:
                if dep_type == "git_editable":
                    url, name = match.groups()
                    return ("git", name, url)
                elif dep_type == "git":
                    name, url = match.groups()
                    return ("git", name, url)
                else:
                    name, version = match.groups()
                    return ("normal", name, version)

        return ("special", line, "")

    def configure_pytorch_source(self, pyproject_data: Any) -> str:
        """Configure PyTorch source in pyproject.toml and return source name"""
        print("🔧 Configurando fonte do PyTorch...")
        
        # Determine PyTorch source based on platform and CUDA availability
        if self.platform == "windows":
            # Check for CUDA on Windows
            try:
                subprocess.run(["nvidia-smi"], check=True, capture_output=True)
                pytorch_source = "pytorch-cu121"
                pytorch_url = "https://download.pytorch.org/whl/cu121"
                print("   🎮 CUDA detectado - usando versão GPU")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytorch_source = "pytorch-cpu"
                pytorch_url = "https://download.pytorch.org/whl/cpu"
                print("   💻 CUDA não encontrado - usando versão CPU")
        else:
            # For Linux/macOS, default to CPU (user can modify manually for CUDA)
            pytorch_source = "pytorch-cpu"
            pytorch_url = "https://download.pytorch.org/whl/cpu"
            print("   💻 Usando versão CPU do PyTorch")

        # Add PyTorch source to pyproject.toml
        if "tool" not in pyproject_data:
            pyproject_data["tool"] = tomlkit.table()
        if "poetry" not in pyproject_data["tool"]:
            pyproject_data["tool"]["poetry"] = tomlkit.table()

        poetry_section = pyproject_data["tool"]["poetry"]
        
        if "source" not in poetry_section:
            poetry_section["source"] = tomlkit.array()

        # Create source entry
        source_entry = tomlkit.inline_table()
        source_entry["name"] = pytorch_source
        source_entry["url"] = pytorch_url
        source_entry["priority"] = "explicit"
        
        # Check if source already exists
        sources = poetry_section.get("source", [])
        source_exists = any(
            source.get("name") == pytorch_source for source in sources
        )
        
        if not source_exists:
            poetry_section["source"].append(source_entry)
            print(f"   ✅ Fonte {pytorch_source} adicionada")
        
        return pytorch_source

    def categorize_dependencies(self) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]], List[Tuple[str, str]], List[str]]:
        """Categorize dependencies from requirements files"""
        normal_deps = {}
        platform_deps: Dict[str, Dict[str, str]] = {"windows": {}, "linux": {}, "macos": {}}
        excluded_deps: List[Tuple[str, str]] = []
        
        skip_packages = {
            "pff",
            "production-fix-flow",
        }

        # Process requirements.txt
        if self.requirements_path.exists():
            with open(self.requirements_path, "r", encoding="utf-8") as f:
                for line in f:
                    result = self.parse_requirement(line)
                    if result:
                        dep_type, name, version = result

                        # Skip self-dependency and variations
                        if name and name.lower() in skip_packages:
                            print(f"   ⏭️  Ignorando auto-dependência: {name}")
                            continue

                        if dep_type == "normal" and name:
                            # Categorize by dependency type
                            if name.lower() in self.PIP_ONLY_PACKAGES:
                                self.pip_deps[name] = version
                                print(f"   🔧 {name} será instalado via pip")
                                continue
                            elif name.lower() in self.PYTORCH_CORE_PACKAGES:
                                self.pytorch_deps[name] = version
                                print(f"   🔥 {name} identificado como PyTorch core")
                                continue
                            elif name.lower() in self.PYTORCH_ECOSYSTEM_PACKAGES:
                                # torch-geometric goes to normal deps (PyPI)
                                normal_deps[name] = version
                                print(f"   📦 {name} identificado como PyTorch ecosystem (PyPI)")
                                continue
                            elif name.lower() in self.PYTORCH_EXTENSION_PACKAGES:
                                self.pip_deps[name] = version
                                print(f"   🔧 {name} será instalado via pip (extensão PyTorch)")
                                continue

                            # Check if platform-specific
                            if self.is_platform_specific(name):
                                if self.should_include_package(name):
                                    # Detect for which platform
                                    if name.lower() in self.WINDOWS_PACKAGES:
                                        platform_deps["windows"][name] = version
                                    elif name.lower() in self.UNIX_PACKAGES:
                                        platform_deps["linux"][name] = version
                                    elif name.lower() in self.MACOS_PACKAGES:
                                        platform_deps["macos"][name] = version
                                else:
                                    excluded_deps.append((name, version))
                            else:
                                normal_deps[name] = version
                        elif dep_type == "git" and name:
                            self.git_deps[name] = version
                        elif dep_type == "special":
                            self.special_deps.append(name)

        # Process requirements.custom.txt
        if self.custom_reqs_path.exists():
            with open(self.custom_reqs_path, "r", encoding="utf-8") as f:
                for line in f:
                    result = self.parse_requirement(line)
                    if result:
                        dep_type, name, version = result
                        if dep_type == "git" and name:
                            self.git_deps[name] = version
                        else:
                            self.special_deps.append(line.strip())

        return normal_deps, platform_deps, excluded_deps, self.special_deps

    def sync_dependencies(self):
        """Sync all dependencies"""
        print("\n🔄 Sincronizando dependências...")
        print(f"   🖥️  Plataforma detectada: {self.platform}")

        # Load existing pyproject.toml
        if self.pyproject_path.exists():
            with open(self.pyproject_path, "r", encoding="utf-8") as f:
                content = f.read().lstrip("\ufeff")  # Remove BOM
                pyproject_data: Any = tomlkit.parse(content)
        else:
            # Create new document
            pyproject_data = tomlkit.document()

        # Ensure basic structure
        if "tool" not in pyproject_data:
            pyproject_data["tool"] = tomlkit.table()
        if "poetry" not in pyproject_data["tool"]:  # type: ignore
            pyproject_data["tool"]["poetry"] = tomlkit.table()  # type: ignore

        poetry_section = pyproject_data["tool"]["poetry"]  # type: ignore

        # Project metadata
        poetry_section["name"] = "production-fix-flow"  # type: ignore
        poetry_section["version"] = "4.0.0"  # type: ignore
        poetry_section["description"] = "Production Fix Flow - Orchestrator for API sequences"  # type: ignore
        poetry_section["authors"] = ["Alex Lira"]  # type: ignore
        poetry_section["readme"] = "README.md"  # type: ignore

        # Packages
        packages_array = tomlkit.array()
        package_item = tomlkit.inline_table()
        package_item["include"] = "pff"
        packages_array.append(package_item)
        poetry_section["packages"] = packages_array  # type: ignore

        # Configure PyTorch source
        pytorch_source = self.configure_pytorch_source(pyproject_data)

        # Dependencies
        if "dependencies" not in poetry_section:  # type: ignore
            poetry_section["dependencies"] = tomlkit.table()  # type: ignore
        deps = poetry_section["dependencies"]  # type: ignore

        # Ensure Python version
        deps["python"] = self.get_python_version()  # type: ignore

        # Categorize dependencies
        normal_deps, platform_deps, excluded_deps, special_deps = self.categorize_dependencies()

        # Add PyTorch core dependencies with source specification
        for name, version in self.pytorch_deps.items():
            # Create dependency with source - be more flexible with versions
            dep_spec = tomlkit.inline_table()
            
            # Convert exact versions to compatible versions for Poetry
            if version and "==" in version:
                clean_version = version.replace("==", "")
                # Use caret versioning for more flexibility
                dep_spec["version"] = f"^{clean_version}"
            else:
                dep_spec["version"] = version or "*"
                
            dep_spec["source"] = pytorch_source
            deps[name] = dep_spec  # type: ignore
            print(f"   🔥 {name} adicionado com fonte {pytorch_source}")

        # Add normal dependencies - also be more flexible
        for name, version in normal_deps.items():
            if version and "==" in version:
                clean_version = version.replace("==", "")
                # Use caret versioning for better compatibility
                deps[name] = f"^{clean_version}"  # type: ignore
            else:
                deps[name] = version or "*"  # type: ignore

        # Add development groups
        if "group" not in poetry_section:  # type: ignore
            poetry_section["group"] = tomlkit.table()  # type: ignore
        if "dev" not in poetry_section["group"]:  # type: ignore
            poetry_section["group"]["dev"] = tomlkit.table()  # type: ignore
        if "dependencies" not in poetry_section["group"]["dev"]:  # type: ignore
            poetry_section["group"]["dev"]["dependencies"] = tomlkit.table()  # type: ignore

        dev_deps = poetry_section["group"]["dev"]["dependencies"]  # type: ignore
        dev_deps["pytest"] = "^8.4.1"  # type: ignore
        dev_deps["black"] = "^24.0.0"  # type: ignore
        dev_deps["flake8"] = "^7.0.0"  # type: ignore
        dev_deps["mypy"] = "^1.0.0"  # type: ignore
        dev_deps["ipython"] = "^9.3.0"  # type: ignore

        # Scripts
        if "scripts" not in poetry_section:  # type: ignore
            poetry_section["scripts"] = tomlkit.table()  # type: ignore
        scripts = poetry_section["scripts"]  # type: ignore
        scripts["pff"] = "pff.cli:main"  # type: ignore

        # Extras for platforms
        if "extras" not in poetry_section:  # type: ignore
            poetry_section["extras"] = tomlkit.table()  # type: ignore
        extras = poetry_section["extras"]  # type: ignore

        # Process platform dependencies correctly
        if platform_deps["windows"]:
            extras["windows"] = list(platform_deps["windows"].keys())  # type: ignore
            for name, version in platform_deps["windows"].items():
                deps[name] = {"version": version, "optional": True}  # type: ignore

        if platform_deps["linux"]:
            extras["linux"] = list(platform_deps["linux"].keys())  # type: ignore
            for name, version in platform_deps["linux"].items():
                deps[name] = {"version": version, "optional": True}  # type: ignore

        if platform_deps["macos"]:
            extras["macos"] = list(platform_deps["macos"].keys())  # type: ignore
            for name, version in platform_deps["macos"].items():
                deps[name] = {"version": version, "optional": True}  # type: ignore

        # Build system
        if "build-system" not in pyproject_data:
            pyproject_data["build-system"] = tomlkit.table()
        build_system = pyproject_data["build-system"]  # type: ignore
        build_system["requires"] = ["poetry-core>=1.0.0"]  # type: ignore
        build_system["build-backend"] = "poetry.core.masonry.api"  # type: ignore

        # Save pyproject.toml
        with open(self.pyproject_path, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(pyproject_data))
        
        # Validate the generated pyproject.toml
        self._validate_pyproject()

        # Report
        print(f"\n✅ Sincronizadas {len(normal_deps)} dependências universais")
        print(f"✅ Sincronizadas {len(self.pytorch_deps)} dependências PyTorch core")
        print(f"✅ Sincronizadas {len(self.git_deps)} dependências Git")
        pip_deps_count = len([pkg for pkg in self.pip_deps.keys() 
                             if pkg.lower() in self.PIP_ONLY_PACKAGES])
        pytorch_ext_count = len([pkg for pkg in self.pip_deps.keys() 
                                if pkg.lower() in self.PYTORCH_EXTENSION_PACKAGES])
        print(f"✅ Sincronizadas {pip_deps_count} dependências pip-only")
        if pytorch_ext_count > 0:
            print(f"✅ Sincronizadas {pytorch_ext_count} extensões PyTorch (via pip)")

        if any(deps for deps in platform_deps.values()):
            print("\n📦 Dependências específicas de plataforma:")
            for plat, deps_dict in platform_deps.items():
                if deps_dict:
                    print(f"   {plat}: {len(deps_dict)} pacotes")

        if excluded_deps:
            print(f"\n⏭️  {len(excluded_deps)} dependências excluídas (outras plataformas)")

        if special_deps:
            print(f"\n⚠️  {len(special_deps)} dependências especiais encontradas")
            self._save_special_deps_script(special_deps)

    def _validate_pyproject(self):
        """Validate and potentially fix the generated pyproject.toml"""
        print("   🔍 Validando pyproject.toml...")
        
        try:
            # Try to parse the file we just wrote
            with open(self.pyproject_path, "r", encoding="utf-8") as f:
                content = f.read()
                parsed = tomlkit.parse(content)
            
            # Check for common issues
            issues_fixed = 0
            
            # Check if poetry section exists
            if "tool" in parsed and "poetry" in parsed["tool"]:  # type: ignore
                poetry_section = parsed["tool"]["poetry"]  # type: ignore
                
                # Ensure source priority is valid
                if "source" in poetry_section:  # type: ignore
                    sources = poetry_section.get("source", [])  # type: ignore
                    for source in sources:  # type: ignore
                        if hasattr(source, 'get') and hasattr(source, '__setitem__'):  # type: ignore
                            if source.get("priority") not in ["default", "primary", "secondary", "explicit"]:  # type: ignore
                                source["priority"] = "explicit"  # type: ignore
                                issues_fixed += 1
                
                # Check for version conflicts in dependencies
                if "dependencies" in poetry_section:  # type: ignore
                    deps = poetry_section["dependencies"]  # type: ignore
                    if hasattr(deps, 'items'):  # type: ignore
                        for name, spec in deps.items():  # type: ignore
                            if name == "python":
                                continue
                            
                            # Fix overly restrictive version specs
                            if hasattr(spec, 'get') and "version" in spec:  # type: ignore
                                version = spec.get("version", "")  # type: ignore
                                if isinstance(version, str) and version.count("^") > 1:
                                    # Fix double carets
                                    spec["version"] = version.replace("^^", "^")  # type: ignore
                                    issues_fixed += 1
                            elif isinstance(spec, str) and spec.count("^") > 1:
                                deps[name] = spec.replace("^^", "^")  # type: ignore
                                issues_fixed += 1
            
            if issues_fixed > 0:
                print(f"   🔧 Corrigidos {issues_fixed} problemas no pyproject.toml")
                with open(self.pyproject_path, "w", encoding="utf-8") as f:
                    f.write(tomlkit.dumps(parsed))
            else:
                print("   ✅ pyproject.toml válido")
                
        except Exception as e:
            print(f"   ⚠️  Erro ao validar pyproject.toml: {e}")
            print("   💡 Continuando mesmo assim...")

    def _save_special_deps_script(self, special_deps: List[str]):
        """Save script for installing special dependencies"""
        # Bash script
        bash_content = "#!/bin/bash\n# Instalação de dependências especiais\n\n"
        for dep in special_deps:
            bash_content += f"pip install '{dep}'\n"

        with open(self.project_root / "install_special.sh", "w", encoding="utf-8") as f:
            f.write(bash_content)

        # PowerShell script
        ps_content = "# Instalação de dependências especiais\n\n"
        for dep in special_deps:
            ps_content += f"pip install '{dep}'\n"

        with open(
            self.project_root / "install_special.ps1", "w", encoding="utf-8"
        ) as f:
            f.write(ps_content)

        if sys.platform != "win32":
            os.chmod(self.project_root / "install_special.sh", 0o755)

    def setup_environment(self):
        """Configure Poetry environment"""
        print("\n🔧 Configurando ambiente Poetry...")

        # Remove old lock if exists
        if self.poetry_lock_path.exists():
            os.remove(self.poetry_lock_path)
            print("   🗑️  poetry.lock removido")

        # Set Poetry to create venv in project
        try:
            subprocess.run(
                ["poetry", "config", "virtualenvs.in-project", "true"], check=True
            )
            print("   ✅ virtualenvs.in-project configurado")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Aviso ao configurar virtualenvs.in-project: {e}")

        # Use base Python (not venv's)
        print(f"   🐍 Usando Python: {self.base_python}")
        try:
            subprocess.run(["poetry", "env", "use", self.base_python], check=True)
            print("   ✅ Ambiente Python configurado")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  Erro ao configurar Python: {e}")

    def install_pytorch_dependencies(self):
        """Install PyTorch dependencies in correct order"""
        print("\n🔥 Instalando dependências PyTorch...")
        
        # Determine PyTorch source
        if self.platform == "windows":
            try:
                subprocess.run(["nvidia-smi"], check=True, capture_output=True)
                pytorch_source = "pytorch-cu121"
                wheel_url = "https://data.pyg.org/whl/torch-2.7.0+cu121.html"
                print("   🎮 Usando PyTorch GPU (CUDA)")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytorch_source = "pytorch-cpu"
                wheel_url = "https://data.pyg.org/whl/torch-2.7.0+cpu.html"
                print("   💻 Usando PyTorch CPU")
        else:
            pytorch_source = "pytorch-cpu"
            wheel_url = "https://data.pyg.org/whl/torch-2.7.0+cpu.html"
            print("   💻 Usando PyTorch CPU")
        
        # Step 1: Install main PyTorch packages via Poetry
        pytorch_core = ["torch", "torchvision", "torchaudio"]
        for pkg in pytorch_core:
            if pkg in self.pytorch_deps:
                print(f"   📦 Instalando {pkg} via Poetry...")
                success = False
                
                # Try with source first
                try:
                    subprocess.run(
                        ["poetry", "add", f"{pkg}=={self.pytorch_deps[pkg]}", "--source", pytorch_source],
                        check=True,
                        capture_output=True
                    )
                    success = True
                    print(f"   ✅ {pkg} instalado com fonte {pytorch_source}")
                except subprocess.CalledProcessError:
                    print(f"   ⚠️  Falha com fonte {pytorch_source}, tentando sem fonte...")
                
                # Fallback: try without source
                if not success:
                    try:
                        subprocess.run(
                            ["poetry", "run", "pip", "install", f"{pkg}=={self.pytorch_deps[pkg]}", 
                             "-f", wheel_url],
                            check=True,
                            capture_output=True
                        )
                        success = True
                        print(f"   ✅ {pkg} instalado via pip como fallback")
                    except subprocess.CalledProcessError as e:
                        print(f"   ❌ Erro ao instalar {pkg}: {e}")

        # Step 2: Install PyTorch extensions via pip with --no-build-isolation
        torch_extensions = list(self.PYTORCH_EXTENSION_PACKAGES)
        extensions_to_install = [pkg for pkg in torch_extensions if pkg in self.pip_deps]
        
        if extensions_to_install:
            print("   🔧 Instalando extensões PyTorch via pip...")

            # Install extensions
            for pkg in extensions_to_install:
                version_spec = f"{pkg}=={self.pip_deps[pkg]}" if self.pip_deps[pkg] != "*" else pkg
                cmd = [
                    "poetry", "run", "pip", "install",
                    version_spec,
                    "--no-build-isolation",
                    "-f", wheel_url
                ]
                
                print(f"   🔧 Instalando {pkg}...")
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"   ✅ {pkg} instalado com sucesso")
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️  Erro ao instalar {pkg}: {e}")
                    print("   💡 Tentando sem especificação de versão...")
                    try:
                        cmd_fallback = [
                            "poetry", "run", "pip", "install",
                            pkg,
                            "--no-build-isolation",
                            "-f", wheel_url
                        ]
                        subprocess.run(cmd_fallback, check=True, capture_output=True)
                        print(f"   ✅ {pkg} instalado na segunda tentativa")
                    except subprocess.CalledProcessError:
                        print(f"   ❌ Falha definitiva ao instalar {pkg}")
                        print("   💡 Pode ser necessário instalar manualmente após a instalação")

        print("   ✅ Instalação PyTorch concluída")

    def install_dependencies(self):
        """Install all dependencies"""
        print("\n📦 Instalando dependências...")

        # Install PyTorch dependencies first (they need special handling)
        if self.pytorch_deps:
            self.install_pytorch_dependencies()
        
        # Install pip-only dependencies early (before Poetry lock)
        if self.pip_deps:
            print("⚙️ Instalando dependências pip-only…")
            for name, version in self.pip_deps.items():
                if name.lower() in self.PIP_ONLY_PACKAGES:
                    print(f"   📦 Instalando {name} (workaround para conflito de nomes)")
                    try:
                        version_spec = f"{name}=={version}" if version != "*" else name
                        subprocess.run(
                            ["poetry", "run", "pip", "install", version_spec],
                            check=True,
                        )
                        print(f"   ✅ {name} instalado via pip")
                    except subprocess.CalledProcessError as e:
                        print(f"   ⚠️  Erro ao instalar {name}: {e}")

        # Install Git dependencies
        if self.git_deps:
            print("🔗 Instalando dependências Git…")
            for name, url in self.git_deps.items():
                self._install_git_dependency(name, url)

        # Try Poetry workflow (unless emergency mode is forced)
        if hasattr(self, '_force_emergency') and self._force_emergency:
            print("   🚨 Modo de emergência forçado - pulando Poetry...")
            self._poetry_failed = True
            self._emergency_pip_install()
        else:
            try:
                poetry_success = self._try_poetry_install()
                
                if not poetry_success:
                    print("\n🚨 Poetry falhou - ativando modo de emergência...")
                    self._poetry_failed = True
                    self._emergency_pip_install()
            except Exception as e:
                print(f"\n🚨 Erro inesperado no Poetry: {e}")
                print("   Ativando modo de emergência...")
                self._poetry_failed = True
                self._emergency_pip_install()

    def _try_poetry_install(self) -> bool:
        """Try Poetry installation workflow, return success status"""
        # Lock dependencies (this should work better now)
        print("   🔒 Criando poetry.lock...")
        lock_success = False
        
        # Try different lock strategies
        lock_commands = [
            ["poetry", "lock", "--no-update"],
            ["poetry", "lock"],
            ["poetry", "lock", "--no-cache"]
        ]
        
        for i, cmd in enumerate(lock_commands):
            try:
                _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("   ✅ poetry.lock criado com sucesso")
                lock_success = True
                break
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️  Tentativa {i+1} de lock falhou: {e.returncode}")
                if i == len(lock_commands) - 1:
                    print("   ❌ Todas as tentativas de lock falharam")
                    # e.stderr is already a string when text=True is used
                    error_msg = e.stderr if e.stderr else 'N/A'
                    print(f"   💡 Último erro: {error_msg}")
                    print("   ⚠️  Pulando Poetry install devido ao lock falhar")
                    return False

        # Try Poetry install
        if lock_success:
            return self._try_poetry_install_commands()
        else:
            return False
    
    def _try_poetry_install_commands(self) -> bool:
        """Try different Poetry install commands"""
        print(f"   📥 Instalando pacotes para {self.platform}...")
        
        install_commands = [
            # Try with platform extras first
            ["poetry", "install", "--no-root", "-E", self.platform],
            # Try without extras
            ["poetry", "install", "--no-root"],
            # Try sync instead of install
            ["poetry", "install", "--no-root", "--sync"],
            # Try without any flags
            ["poetry", "install"]
        ]
        
        for i, cmd in enumerate(install_commands):
            try:
                _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"   ✅ Poetry install executado com sucesso (estratégia {i+1})")
                return True
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️  Tentativa {i+1} falhou: {e.returncode}")
                if i == len(install_commands) - 1:
                    print("   ❌ Todas as tentativas de Poetry install falharam")
                    # e.stderr is already a string when text=True is used
                    error_msg = e.stderr if e.stderr else 'N/A'
                    print(f"   💡 Último erro: {error_msg}")
        
        return False

    def _emergency_pip_install(self):
        """Emergency pip installation when Poetry fails"""
        print("   🆘 Instalando dependências críticas via pip...")
        print("   💡 Isso pode demorar alguns minutos...")
        
        # Read requirements.txt and install essential packages
        if self.requirements_path.exists():
            essential_packages = []
            
            with open(self.requirements_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    
                    # Extract package name and version
                    result = self.parse_requirement(line)
                    if result and result[0] == "normal":
                        _, name, version = result
                        
                        # Skip packages we already installed
                        if name.lower() in self.PYTORCH_CORE_PACKAGES:
                            continue
                        if name.lower() in self.PYTORCH_EXTENSION_PACKAGES:
                            continue
                        if name.lower() in self.PIP_ONLY_PACKAGES:
                            continue
                        if name.lower() in {"pff", "production-fix-flow"}:
                            continue
                            
                        # Add to essential packages
                        if version and version != "*":
                            essential_packages.append(f"{name}=={version}")
                        else:
                            essential_packages.append(name)
            
            print(f"   📊 {len(essential_packages)} pacotes para instalar via pip")
            
            # Install in batches to avoid overwhelming pip
            batch_size = 10
            successful_installs = 0
            
            for i in range(0, len(essential_packages), batch_size):
                batch = essential_packages[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(essential_packages) + batch_size - 1) // batch_size
                
                print(f"   📦 Instalando lote {batch_num}/{total_batches}: {len(batch)} pacotes")
                
                try:
                    subprocess.run(
                        ["poetry", "run", "pip", "install"] + batch,
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    print(f"   ✅ Lote {batch_num} instalado com sucesso")
                    successful_installs += len(batch)
                except subprocess.CalledProcessError as e:
                    print(f"   ⚠️  Erro no lote {batch_num}: {e.returncode}")
                    # Try installing individually
                    print("   💡 Tentando instalação individual...")
                    for package in batch:
                        try:
                            subprocess.run(
                                ["poetry", "run", "pip", "install", package],
                                check=True,
                                capture_output=True
                            )
                            print(f"   ✅ {package}")
                            successful_installs += 1
                        except subprocess.CalledProcessError:
                            print(f"   ❌ {package}")
            
            print(f"   🎯 Modo de emergência concluído: {successful_installs}/{len(essential_packages)} instalados")
        else:
            print("   ⚠️  requirements.txt não encontrado - pulando modo de emergência")

    def _install_git_dependency(self, name: str, url: str):
        """Install a single Git dependency with fallback strategies"""
        tried = False
        candidate = url
        
        while True:
            try:
                print(f"   🔗 Tentando instalar {name} de {candidate}")
                subprocess.run(
                    ["poetry", "run", "pip", "install", candidate], 
                    check=True,
                    capture_output=True,
                    text=True
                )
                tried = True
                print(f"   ✅ {name} instalado com sucesso")
                break
            except subprocess.CalledProcessError as e:
                # SSH → HTTPS
                if candidate.startswith("git+ssh://"):
                    _, rest = candidate.split("git+ssh://", 1)
                    candidate = "git+https://" + rest.split("@", 1)[-1]
                    print(f"   ⚠️ SSH falhou, tentando HTTPS: {candidate}")
                    continue
                # HTTPS → HTTPS+TOKEN (for gitlab-devrocks)
                if "gitlab-devrocks.eisa.corp.com" in candidate:
                    token = os.getenv("GITLAB_TOKEN")
                    if token:
                        repo, rev = candidate.split(".git@", 1)
                        candidate = f"{repo}.git@{rev}".replace(
                            "git+https://",
                            f"git+https://gitlab-ci-token:{token}@",
                        )
                        print(f"   🔐 Tentando HTTPS+TOKEN: {candidate}")
                        continue
                print(f"   ❌ Falha ao instalar {name} ({candidate})")
                print(f"      Erro: {e.stderr if hasattr(e, 'stderr') else str(e)}")
                break
        
        if not tried:
            print(f"   ⚠️ {name} não foi instalado - todas as tentativas falharam")

    def create_distribution_files(self):
        """Create distribution files"""
        git_deps_list = list(self.git_deps.values())
        pip_deps_list = [f"{n}=={v}" for n, v in self.pip_deps.items() 
                        if n.lower() in self.PIP_ONLY_PACKAGES]
        pytorch_extensions = [f"{n}=={v}" for n, v in self.pip_deps.items() 
                             if n.lower() in self.PYTORCH_EXTENSION_PACKAGES]

        install_content = f'''#!/usr/bin/env python3
"""
Instalador universal do PFF
Funciona em Windows, Linux e macOS
"""
import subprocess
import sys
import os
import platform
from pathlib import Path

def detect_platform():
    system = platform.system().lower()
    if system == 'windows':
        return 'windows'
    elif system == 'darwin':
        return 'macos'
    else:
        return 'linux'

def check_cuda():
    """Check if CUDA is available"""
    try:
        subprocess.run(["nvidia-smi"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    print("🚀 Instalador PFF v4.0.0")
    print("=" * 50)
    
    plat = detect_platform()
    has_cuda = check_cuda() if plat == 'windows' else False
    
    print(f"🖥️  Plataforma detectada: {{plat}}")
    if plat == 'windows':
        print(f"🎮 CUDA disponível: {{'Sim' if has_cuda else 'Não'}}")
    
    # Instala Poetry se necessário
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
    except:
        print("📦 Instalando Poetry...")
        if plat == 'windows':
            subprocess.run([sys.executable, "-m", "pip", "install", "poetry"], check=True)
        else:
            subprocess.run(
                "curl -sSL https://install.python-poetry.org | python3 -",
                shell=True, check=True
            )
    
    # Configura e instala
    print("🔧 Configurando ambiente...")
    subprocess.run(["poetry", "config", "virtualenvs.in-project", "true"], check=True)
    subprocess.run(["poetry", "env", "use", sys.executable], check=True)
    
    print("📦 Instalando dependências principais...")
    install_cmd = ["poetry", "install"]
    
    # Adiciona extras da plataforma
    if plat == 'windows':
        install_cmd.extend(["-E", "windows"])
    elif plat == 'linux':
        install_cmd.extend(["-E", "linux"])
    elif plat == 'macos':
        install_cmd.extend(["-E", "macos"])
        
    subprocess.run(install_cmd, check=True)
    
    # ── PYTORCH EXTENSIONS ──
    pytorch_extensions = {pytorch_extensions!r}
    if pytorch_extensions:
        print("🔥 Instalando extensões PyTorch...")
        if plat == 'windows' and has_cuda:
            wheel_url = "https://data.pyg.org/whl/torch-2.7.0+cu121.html"
        else:
            wheel_url = "https://data.pyg.org/whl/torch-2.7.0+cpu.html"
            
        for ext in pytorch_extensions:
            subprocess.run([
                "poetry", "run", "pip", "install", ext,
                "--no-build-isolation", "-f", wheel_url
            ], check=True)
    
    # ── GIT DEPS (via pip) ──
    git_deps = {git_deps_list!r}
    if git_deps:
        print("🔗 Instalando dependências Git…")
        for url in git_deps:
            subprocess.run(
                ["poetry", "run", "pip", "install", url],
                check=True
            )
    
    # ── PIP DEPS excluídas do TOML ──
    pip_deps = {pip_deps_list!r}
    if pip_deps:
        print("⚙️ Instalando dependências pip-only...")
        for dep in pip_deps:
            subprocess.run(
                ["poetry", "run", "pip", "install", dep],
                check=True
            )
    
    # Instruções finais
    print("\\n✅ Instalação concluída!")
    print("\\n📝 Para usar o PFF:")
    print("   poetry run python -m pff")
    print("\\n💡 Ou ative o ambiente virtual:")
    if plat == 'windows':
        print("   .venv\\\\Scripts\\\\activate")
    else:
        print("   source .venv/bin/activate")
    print("   python -m pff")
    
if __name__ == "__main__":
    main()
'''

        with open(self.project_root / "install.py", "w", encoding="utf-8") as f:
            f.write(install_content)

        # Make executable on Unix
        if sys.platform != "win32":
            os.chmod(self.project_root / "install.py", 0o755)

        # 2. INSTALL.md
        readme_content = """# 📦 Instalação do PFF

## Instalação Rápida

```bash
python install.py
```

## Plataformas Suportadas

✅ Windows 10/11 (CPU/CUDA)
✅ Linux (Ubuntu, Debian, CentOS, etc.)
✅ macOS 10.15+

## Dependências por Plataforma

### 🪟 Windows

- Inclui: pywin32, windows-curses
- Python: 3.12+
- CUDA: Detectado automaticamente

### 🐧 Linux

- Inclui: python-daemon, systemd-python
- Python: 3.12+

### 🍎 macOS

- Inclui: pyobjc-core
- Python: 3.12+

## Solução de Problemas

### Erro: "Poetry não encontrado"
```bash
pip install poetry
```

### Erro: "torch-scatter build failed"
```bash
# Windows - instale Visual Studio Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Depois execute:
poetry run pip install torch-scatter --no-build-isolation
```

### Erro: "Dependência X não instalada"
```bash
# Windows
powershell -ExecutionPolicy Bypass -File install_special.ps1

# Linux/macOS
bash install_special.sh
```

## Build from Source

```bash
poetry build
# Arquivos em dist/
```

## Desenvolvimento

```bash
poetry install -E dev
poetry run pytest
poetry run black .
poetry run mypy .
```
"""

        with open(self.project_root / "INSTALL.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

        print("\n✅ Criados arquivos de distribuição:")
        print("   - install.py (instalador universal)")
        print("   - INSTALL.md (documentação)")

    def run(self, clean_done=False):
        """Run full sync"""
        print("🚀 Iniciando sincronização Poetry para PFF")
        print("=" * 50)

        # Clean up any old backup directories first
        self._cleanup_old_backups()

        self.ensure_poetry()
        self.sync_dependencies()
        
        print("\n🔧 Configurando Poetry para criar .venv no projeto...")
        try:
            subprocess.run(
                ["poetry", "config", "virtualenvs.in-project", "true", "--local"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            if b"No such option" not in e.stderr and b"is not defined" not in e.stderr:
                print(f"⚠️  Aviso ao configurar virtualenvs.in-project: {e.stderr.decode()}")

        if not clean_done:
            print("\n🧹 Limpando ambiente antigo (se existir)...")
            if self.poetry_lock_path.exists():
                os.remove(self.poetry_lock_path)
                print("   🗑️  poetry.lock removido")
            
            venv_path = self.project_root / ".venv"
            if venv_path.exists():
                self._robust_rmtree(venv_path)

        self.setup_environment()
        self.install_dependencies()
        self.create_distribution_files()
        
        # Final status check
        self._check_installation_status()

        print("\n✨ Sincronização concluída!")
        
        # Check if everything worked
        if hasattr(self, '_poetry_failed') and self._poetry_failed:
            print("\n⚠️  ATENÇÃO: Poetry falhou, mas modo de emergência foi ativado")
            print("   📦 Dependências críticas foram instaladas via pip")
            print("   💡 Para usar: poetry run python -m pff")
            print("   🔧 Se houver problemas, tente:")
            print("      • poetry shell  # ativa o ambiente")
            print("      • python -m pff  # executa diretamente")
            print("      • python sync.py --emergency  # força pip para tudo")
        else:
            print("   🎉 Todas as dependências foram instaladas com sucesso!")

        print("\n📝 Próximos passos:")
        print("   1. Para usar: poetry run python -m pff")
        print("   2. Para distribuir: python install.py")
        print("   3. Para build: poetry build")
        print("\n📦 Para instalar em outra máquina:")
        print("   1. Copie todo o projeto")
        print("   2. Execute: python install.py")
        
        if self.git_deps:
            print("\n⚠️  Não esqueça de configurar tokens para dependências Git privadas:")
            for name, url in self.git_deps.items():
                if "gitlab-devrocks" in url:
                    print(f"   export GITLAB_TOKEN=<seu_token>  # Para {name}")
        
        # Additional tips based on what happened
        if hasattr(self, '_poetry_failed') and self._poetry_failed:
            print("\n💡 Dicas para problemas com Poetry:")
            print("   • Se poetry run falhar: ative manualmente (.venv\\Scripts\\activate)")
            print("   • Para reinstalar: python sync.py --clean")
            print("   • Para debug: poetry install --no-root -vvv")
            print("   • Para usar pip direto: python sync.py --emergency")

    def _check_installation_status(self):
        """Check what packages were actually installed"""
        print("\n🔍 Verificando status da instalação...")
        
        # Check if we can run Python in the environment
        try:
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "import sys; print('Python OK')"],
                check=True,
                capture_output=True,
                text=True
            )
            print("   ✅ Ambiente Poetry funcional")
        except subprocess.CalledProcessError:
            print("   ❌ Ambiente Poetry com problemas")
            return
        
        # Check critical packages
        critical_packages = ["torch", "pysimdjson"]
        if self.git_deps:
            critical_packages.extend(self.git_deps.keys())
        
        working_packages = []
        failed_packages = []
        
        for package in critical_packages:
            try:
                # Try to import the package
                import_name = package.lower().replace("-", "_")
                if package == "PyClause":
                    import_name = "pyclause"
                    
                result = subprocess.run(
                    ["poetry", "run", "python", "-c", f"import {import_name}; print('{package} OK')"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"   ✅ {package} funcional")
                working_packages.append(package)
            except subprocess.CalledProcessError:
                print(f"   ⚠️  {package} não encontrado ou com problemas")
                failed_packages.append(package)
        
        # Check Poetry environment
        try:
            result = subprocess.run(
                ["poetry", "run", "pip", "list"],
                check=True,
                capture_output=True,
                text=True
            )
            installed_packages = result.stdout.count('\n')
            print(f"   📊 Total de pacotes instalados: ~{installed_packages}")
        except subprocess.CalledProcessError:
            print("   ⚠️  Não foi possível listar pacotes instalados")
        
        # Summary
        if failed_packages:
            print(f"\n   ⚠️  Pacotes com problemas: {', '.join(failed_packages)}")
            if hasattr(self, '_poetry_failed') and self._poetry_failed:
                print("   💡 Isso é esperado quando o Poetry falha - modo de emergência ativado")
            else:
                print("   💡 Eles podem ser instalados manualmente depois")
        
        if len(working_packages) >= len(critical_packages) // 2:
            print("   🎉 Instalação parece funcional!")
        elif hasattr(self, '_poetry_failed') and self._poetry_failed:
            print("   🔧 Ambiente básico criado - algumas dependências podem estar faltando")
        else:
            print("   ⚠️  Podem haver problemas na instalação")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sincronizar dependências Poetry para PFF",
        epilog="""
Exemplos de uso:
  python sync.py                           # Instalação normal
  python sync.py --clean                   # Remove .venv e reinstala tudo
  python sync.py --emergency               # Força instalação via pip
  python sync.py --clean --force-clean     # Limpeza forçada (Windows)
  python sync.py --clean --emergency       # Limpeza + modo emergência
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clean", action="store_true", 
                       help="Remove o ambiente virtual existente antes de recriar")
    parser.add_argument("--emergency", action="store_true",
                       help="Força o modo de emergência (instala via pip)")
    parser.add_argument("--force-clean", action="store_true",
                       help="Força limpeza mesmo com arquivos em uso (Windows)")
    args = parser.parse_args()
    
    syncer = PoetrySync()
    
    if args.emergency:
        print("🚨 Modo de emergência forçado pelo usuário")
        syncer._force_emergency = True
    
    clean_done = False
    if args.clean:
        print("🧹 Modo limpeza ativado - removendo ambiente existente...")
        
        if args.force_clean:
            print("   💪 Modo força ativado - tentativas mais agressivas")
        
        venv_path = syncer.project_root / ".venv"
        if venv_path.exists():
            # Use more attempts if force-clean is enabled
            max_attempts = 5 if args.force_clean else 3
            if syncer._robust_rmtree(venv_path, max_attempts):
                print("   ✅ Limpeza do .venv concluída")
            else:
                print("   ⚠️  Problemas na limpeza do .venv")
                if not args.force_clean:
                    print("   💡 Tente: python sync.py --clean --force-clean")
                print("   🔄 Continuando mesmo assim...")
        
        if syncer.poetry_lock_path.exists():
            syncer.poetry_lock_path.unlink()
            print("   🗑️  poetry.lock removido")
        clean_done = True
    
    syncer.run(clean_done=clean_done)