"""Lector V2 de archivos .imgmap mediante pywebview."""

import sys
import atexit
import hashlib
import functools
import json
import shutil
import threading
import traceback
import http.server
import socketserver
from pathlib import Path
from urllib.parse import urlparse, unquote

import webview

from mapformat import build_from_folder, prepare_map, EXTENSION
from library import get_library_files, read_cover

try:
    import settings
except ImportError:
    settings = None

if getattr(sys, "frozen", False):
    WEB_DIR = Path(sys._MEIPASS)
else:
    WEB_DIR = Path(__file__).resolve().parent

TEMPLATES_DIR = WEB_DIR / "templates"
SEARCH_DIRS = [TEMPLATES_DIR, WEB_DIR]
ERROR_LOG = Path.home() / "imgmap_reader_web_error.log"
DEBUG = "--debug" in sys.argv

STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".mjs": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
}


def find_web_file(rel):
    rel = rel.lstrip("/\\")
    if not rel or Path(rel).suffix.lower() not in STATIC_MIME:
        return None

    for base in SEARCH_DIRS:
        try:
            base_resolved = base.resolve()
            candidate = (base / rel).resolve()
        except (OSError, ValueError):
            continue

        if base_resolved != candidate and base_resolved not in candidate.parents:
            continue
        if candidate.is_file():
            return candidate
    return None


def is_allowed_library_file(path):
    """Solo admite archivos .imgmap directamente dentro de la biblioteca configurada."""
    try:
        path = Path(path).resolve()
        if path.suffix.lower() != EXTENSION or not path.is_file():
            return False
        library = settings.get_library_path() if settings else None
        return library is not None and path.parent == library.resolve()
    except Exception:
        return False


# ---------------- VELOCIDAD DE DESPLAZAMIENTO ----------------
DEFAULT_SCROLL_SPEED = 1300
MIN_SCROLL_SPEED = 100
MAX_SCROLL_SPEED = 5000


def _clamp_speed(value):
    return max(MIN_SCROLL_SPEED, min(MAX_SCROLL_SPEED, int(round(float(value)))))


def load_scroll_speed():
    if settings is None:
        return DEFAULT_SCROLL_SPEED
    try:
        return _clamp_speed(
            settings.load_settings().get("scroll_speed", DEFAULT_SCROLL_SPEED)
        )
    except Exception:
        return DEFAULT_SCROLL_SPEED


def save_scroll_speed(value):
    if settings is None:
        return
    data = settings.load_settings()
    data["scroll_speed"] = value
    settings.save_settings(data)


class MapState:
    """Mantiene un mapa abierto y su extracción temporal en disco."""

    def __init__(self):
        self.path = None
        self.manifest = None
        self.prepared = None
        self.mtime = 0
        self.lock = threading.Lock()

    def open(self, path):
        # La apertura no se completa hasta reconstruir todas las imágenes.
        # prepare_map trabaja por imagen y conserva los WebP en disco temporal.
        path = Path(path).resolve()
        prepared = prepare_map(path)

        with self.lock:
            previous = self.prepared
            self.path = path
            self.manifest = prepared.manifest
            self.prepared = prepared
            self.mtime = path.stat().st_mtime_ns

        if previous is not None:
            previous.close()

    def entry_path(self, arcname):
        with self.lock:
            if self.prepared is None:
                raise FileNotFoundError("No hay ningun archivo abierto")
            return self.prepared.entry_path(arcname)

    def close(self):
        with self.lock:
            previous = self.prepared
            self.prepared = None
            self.path = None
            self.manifest = None
            self.mtime = 0
        if previous is not None:
            previous.close()


STATE = MapState()
atexit.register(STATE.close)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            route = unquote(parsed.path)

            if route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if route in ("/", "/index.html"):
                return self._serve_static("index.html")
            if route == "/manifest":
                return self._serve_manifest()
            if route.startswith("/img/"):
                return self._serve_image(route[len("/img/"):])
            if route.startswith("/library-cover/"):
                return self._serve_library_cover(route[len("/library-cover/"):])
            return self._serve_static(route)

        except Exception:
            ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
            try:
                self.send_error(500)
            except Exception:
                pass

    def _serve_static(self, rel):
        path = find_web_file(rel)
        if path is None:
            self.send_error(404, f"No existe el archivo: {rel}")
            return

        data = path.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            STATIC_MIME.get(path.suffix.lower(), "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_manifest(self):
        if STATE.manifest is None:
            self.send_error(404, "No hay ningun archivo abierto")
            return

        body = dict(STATE.manifest)
        body["_path"] = str(STATE.path)
        data = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_cached(self, data, mime, etag):
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _serve_image(self, arcname):
        try:
            image_path = STATE.entry_path(arcname)
            image_size = image_path.stat().st_size
        except (KeyError, FileNotFoundError):
            self.send_error(404)
            return

        # Todos los elementos de V2 se reconstruyen como WebP lossless.
        # Se transmite desde disco para no cargar la imagen completa en RAM.
        key = f"{STATE.path}|{STATE.mtime}|{arcname}"
        etag = '"' + hashlib.md5(key.encode("utf-8")).hexdigest() + '"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/webp")
        self.send_header("Content-Length", str(image_size))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(image_path, "rb") as image_file:
            shutil.copyfileobj(image_file, self.wfile, 1024 * 1024)

    def _serve_library_cover(self, spec):
        map_path, _, cover = spec.rpartition("|")
        if not map_path or not cover:
            self.send_error(404)
            return
        if not is_allowed_library_file(map_path):
            self.send_error(403)
            return

        try:
            # Se usa la miniatura JPEG de la cabecera. No se reconstruye el mapa
            # entero cada vez que se muestra una tarjeta de la biblioteca.
            data = read_cover(map_path, cover)
            mtime = Path(map_path).stat().st_mtime_ns
        except Exception:
            self.send_error(404)
            return

        key = f"{map_path}|{mtime}|{cover}"
        etag = '"' + hashlib.md5(key.encode("utf-8")).hexdigest() + '"'
        self._send_cached(data, "image/jpeg", etag)


def _open_dialog_type():
    file_dialog = getattr(webview, "FileDialog", None)
    if file_dialog is not None and hasattr(file_dialog, "OPEN"):
        return file_dialog.OPEN
    return webview.OPEN_DIALOG


def _dialog_type(name):
    file_dialog = getattr(webview, "FileDialog", None)
    if file_dialog is not None and hasattr(file_dialog, name):
        return getattr(file_dialog, name)
    return getattr(webview, f"{name}_DIALOG")


def _first(result):
    if not result:
        return None
    if isinstance(result, (list, tuple)):
        return result[0]
    return result


def logged(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if DEBUG:
            print(f"[API] {fn.__name__}{args[1:]}")
        result = fn(*args, **kwargs)
        if DEBUG:
            print(f"[API]   -> {repr(result)[:300]}")
        return result
    return wrapper


class Api:
    """Métodos expuestos a JavaScript mediante window.pywebview.api."""

    def __init__(self):
        self._window = None

    def _open_path(self, path):
        STATE.open(path)
        if settings is not None:
            settings.set_last_file(path)
        return {"path": str(path)}

    @logged
    def choose_and_open(self):
        result = self._window.create_file_dialog(
            _open_dialog_type(),
            file_types=(
                f"Mapa de imagenes (*{EXTENSION})",
                "Todos los archivos (*.*)",
            ),
        )
        if not result:
            return None
        return self._open_path(result[0])

    @logged
    def create_new_map(self):
        """Pide la carpeta, crea el .imgmap V2 y lo abre."""
        try:
            folder = _first(
                self._window.create_file_dialog(_dialog_type("FOLDER"))
            )
            if not folder:
                return None
            folder = Path(folder)

            output = _first(
                self._window.create_file_dialog(
                    _dialog_type("SAVE"),
                    save_filename=f"{folder.name}{EXTENSION}",
                    file_types=(f"Mapa de imagenes (*{EXTENSION})",),
                )
            )
            if not output:
                return None

            output = Path(output)
            if output.suffix.lower() != EXTENSION:
                output = output.with_suffix(EXTENSION)

            build_from_folder(folder, output, title=folder.name)
            return self._open_path(output)

        except Exception as exc:
            ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
            return {"error": str(exc)}

    # ---------------- VELOCIDAD DE DESPLAZAMIENTO ----------------
    def _speed_info(self):
        return {
            "value": load_scroll_speed(),
            "default": DEFAULT_SCROLL_SPEED,
            "min": MIN_SCROLL_SPEED,
            "max": MAX_SCROLL_SPEED,
        }

    @logged
    def get_scroll_speed(self):
        return self._speed_info()

    @logged
    def set_scroll_speed(self, value):
        try:
            save_scroll_speed(_clamp_speed(value))
        except (TypeError, ValueError):
            return {"error": "La velocidad debe ser un numero"}
        return self._speed_info()

    @logged
    def reset_scroll_speed(self):
        save_scroll_speed(DEFAULT_SCROLL_SPEED)
        return self._speed_info()

    # ---------------- ZOOM ----------------
    @logged
    def get_zoom(self, path=None):
        if settings is None:
            return None
        try:
            return settings.get_zoom(path)
        except Exception:
            return None

    @logged
    def set_zoom(self, path, value):
        if settings is None:
            return None
        try:
            return settings.set_zoom(path, value)
        except Exception:
            return None

    # ---------------- CONTINUAR ----------------
    @logged
    def get_last_file(self):
        if settings is None:
            return None
        path = settings.get_last_file()
        if not path:
            return None
        return {"path": str(path), "name": Path(path).name}

    @logged
    def open_last_file(self):
        if settings is None:
            return {"error": "No hay un archivo reciente"}
        path = settings.get_last_file()
        if path is None:
            return {"error": "No hay un archivo reciente"}
        try:
            return self._open_path(path)
        except Exception as exc:
            ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
            return {"error": str(exc)}

    # ---------------- BIBLIOTECA ----------------
    @logged
    def get_library(self):
        folder = settings.get_library_path() if settings else None
        if folder is None:
            return {"path": None, "files": []}
        return {"path": str(folder), "files": get_library_files(folder)}

    @logged
    def choose_library_folder(self):
        try:
            folder = _first(
                self._window.create_file_dialog(_dialog_type("FOLDER"))
            )
            if not folder:
                return None
            if settings is None:
                return {"error": "Falta settings.py"}
            settings.set_library_path(folder)
            return self.get_library()
        except Exception as exc:
            ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
            return {"error": str(exc)}

    @logged
    def open_library_file(self, path):
        if not is_allowed_library_file(path):
            return {"error": "El archivo no pertenece a la biblioteca"}
        try:
            return self._open_path(path)
        except Exception as exc:
            ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
            return {"error": str(exc)}


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_server():
    httpd = Server(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return port


def is_registered():
    import winreg
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Classes\VECTOR.imgmap\shell\open\command",
        ) as key:
            return winreg.QueryValueEx(key, "")[0] == f'"{sys.executable}" "%1"'
    except OSError:
        return False


def register_file_type():
    import winreg
    import ctypes

    if not getattr(sys, "frozen", False):
        print("Ejecutalo desde el VECTOR.exe empaquetado.")
        return

    exe = sys.executable
    entries = [
        (r"Software\Classes\.imgmap", "VECTOR.imgmap"),
        (
            r"Software\Classes\.imgmap\ShellEx\{E357FCCD-A995-4576-B01F-234630154E96}",
            "{C7657C4A-9F68-40FA-A4DF-96BC08EB3551}",
        ),
        (r"Software\Classes\VECTOR.imgmap", "Mapa de imagenes VECTOR"),
        (r"Software\Classes\VECTOR.imgmap\DefaultIcon", f'"{exe}",0'),
        (r"Software\Classes\VECTOR.imgmap\shell\open\command", f'"{exe}" "%1"'),
    ]
    for path, value in entries:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path) as key:
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, value)

    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)


def main():
    if "--register" in sys.argv:
        register_file_type()
        return

    if getattr(sys, "frozen", False) and sys.platform == "win32":
        try:
            if not is_registered():
                register_file_type()
        except Exception:
            ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")

    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    if args:
        try:
            STATE.open(args[0])
        except Exception:
            ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")

    port = start_server()
    api = Api()
    window = webview.create_window(
        "Lector de mapas de imagenes",
        url=f"http://127.0.0.1:{port}/",
        js_api=api,
        width=1200,
        height=800,
        min_size=(600, 400),
        maximized=True,
    )
    api._window = window
    webview.start(debug=DEBUG)
    STATE.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERROR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise
