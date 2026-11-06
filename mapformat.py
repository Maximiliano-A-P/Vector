"""
Formato V2 de Vector (.imgmap).

El archivo contiene:
  - Cabecera JSON con el manifest y los metadatos de cada imagen.
  - Miniatura JPEG de la portada para la biblioteca.
  - Diccionario global RGBA comprimido con zlib.
  - Una secuencia de identificadores de color por imagen, también comprimida.

Las imágenes originales se decodifican a RGBA y sus colores se deduplican
contra un diccionario SQLite en disco. Al abrir el mapa se reconstruyen
imágenes WebP lossless en una carpeta temporal. Nunca se carga el mapa
completo en memoria.

Requisitos: Pillow y NumPy.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
import zlib
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageFile


FORMAT_VERSION = 2
EXTENSION = ".imgmap"
MAGIC = b"VECTOR-IMGMAP-V2\x00"
HEADER_LENGTH = struct.Struct("<I")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE = 512
TILE_HEIGHT = 128
IO_CHUNK_SIZE = 1024 * 1024
UINT32_MAX = (1 << 32) - 1

# Pillow protege contra imágenes con dimensiones anómalas. Los mapas pueden
# contener imágenes grandes legítimas; se desactiva el límite de advertencia
# y se valida el tamaño real al procesar cada imagen.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = False


# ---------------------------------------------------------------------------
# Descubrimiento y nombres internos
# ---------------------------------------------------------------------------


def _find_cover(dir_path: Path) -> Optional[Path]:
    for path in sorted(dir_path.iterdir()):
        if (
            path.is_file()
            and path.stem.lower() == "portada"
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            return path
    return None


def _extract_number(name: str) -> Optional[int]:
    matches = re.findall(r"\d+", name)
    if not matches:
        return None
    return int(matches[-1])


def _list_branch_images(dir_path: Path, cover_path: Path) -> list[Path]:
    candidates = []
    cover_resolved = cover_path.resolve()

    for path in dir_path.iterdir():
        if not path.is_file() or path.resolve() == cover_resolved:
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        candidates.append((_extract_number(path.stem), path))

    without_number = [path.name for number, path in candidates if number is None]
    if without_number:
        raise ValueError(
            "No se pudo determinar el número de orden de: "
            + ", ".join(sorted(without_number))
        )

    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def _build_asset_arcname(source_path: Path, root_folder: Path, used_names: set[str]) -> str:
    """Mantiene la carpeta y el nombre base, pero normaliza la extensión a .webp."""
    relative_parent = source_path.parent.relative_to(root_folder)
    stem = source_path.stem
    suffix = 2
    file_name = f"{stem}.webp"
    relative_name = (
        (PurePosixPath(*relative_parent.parts) / file_name).as_posix()
        if str(relative_parent) != "."
        else file_name
    )

    while relative_name.casefold() in used_names:
        file_name = f"{stem}__{suffix}.webp"
        relative_name = (
            (PurePosixPath(*relative_parent.parts) / file_name).as_posix()
            if str(relative_parent) != "."
            else file_name
        )
        suffix += 1

    used_names.add(relative_name.casefold())
    return relative_name


def _safe_asset_path(root: Path, arcname: str) -> Path:
    """Resuelve una ruta del manifest e impide salir de la carpeta temporal."""
    relative = PurePosixPath(arcname)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Ruta de imagen no válida en el manifest: {arcname}")

    destination = root.joinpath(*relative.parts).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise ValueError(f"Ruta de imagen fuera del directorio temporal: {arcname}")
    return destination


# ---------------------------------------------------------------------------
# Miniatura JPEG
# ---------------------------------------------------------------------------


def _make_thumbnail(image_path: Path) -> bytes:
    """Genera una miniatura JPEG independiente del contenido comprimido del mapa."""
    with Image.open(image_path) as image:
        image.seek(0)
        image.thumbnail((THUMB_SIZE, THUMB_SIZE))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
        return output.getvalue()


# ---------------------------------------------------------------------------
# Diccionario global de píxeles
# ---------------------------------------------------------------------------


def _create_palette_database(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(database_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")  # Aproximadamente 32 MiB.

    connection.execute(
        "CREATE TABLE palette ("
        "id INTEGER PRIMARY KEY, "
        "rgba BLOB NOT NULL UNIQUE)"
    )
    connection.execute(
        "CREATE TEMP TABLE tile_colors ("
        "local_id INTEGER PRIMARY KEY, "
        "rgba BLOB NOT NULL, "
        "global_id INTEGER)"
    )
    connection.commit()
    return connection


def _map_tile_to_global_ids(
    rgba_pixels: np.ndarray,
    connection: sqlite3.Connection,
) -> np.ndarray:
    """Convierte los RGBA de un bloque en IDs globales sin guardar el mapa en RAM."""
    flat_pixels = np.ascontiguousarray(rgba_pixels.reshape(-1, 4))
    rgba_dtype = np.dtype((np.void, 4))
    packed_pixels = flat_pixels.view(rgba_dtype).reshape(-1)

    # unique_values conserva los cuatro bytes exactos de cada color. No se
    # utiliza un hash como identidad, por lo que no existen colisiones.
    unique_values, inverse = np.unique(packed_pixels, return_inverse=True)
    unique_rgba = unique_values.view(np.uint8).reshape(-1, 4)

    cursor = connection.cursor()
    cursor.execute("DELETE FROM tile_colors")
    cursor.executemany(
        "INSERT INTO tile_colors(local_id, rgba) VALUES (?, ?)",
        (
            (local_id, row.tobytes())
            for local_id, row in enumerate(unique_rgba)
        ),
    )

    # Cada color nuevo recibe un ID que no se vuelve a utilizar.
    cursor.execute(
        "INSERT OR IGNORE INTO palette(rgba) "
        "SELECT rgba FROM tile_colors"
    )
    cursor.execute(
        "UPDATE tile_colors SET global_id = ("
        "SELECT palette.id FROM palette "
        "WHERE palette.rgba = tile_colors.rgba)"
    )

    # SQLite comienza los IDs en 1; el formato los guarda desde 0.
    global_ids = np.fromiter(
        (row[0] - 1 for row in cursor.execute(
            "SELECT global_id FROM tile_colors ORDER BY local_id"
        )),
        dtype=np.uint32,
        count=len(unique_values),
    )

    mapped = global_ids[inverse]
    return mapped.astype("<u4", copy=False)


def _process_source_image(
    source_path: Path,
    arcname: str,
    index_path: Path,
    connection: sqlite3.Connection,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> dict:
    """Deduplica una imagen y elige entre IDs directos y RLE de IDs."""
    raw_path = index_path.with_name(index_path.stem + ".u32.z")
    rle_path = index_path.with_name(index_path.stem + ".rle.z")

    with Image.open(source_path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError(
                f"La imagen animada no está admitida: {source_path.name}. "
                "Convertí sus fotogramas a imágenes independientes."
            )

        width, height = source.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensiones no válidas en {source_path.name}")

        pixel_count = width * height
        raw_compressor = zlib.compressobj(level=6)
        rle_compressor = zlib.compressobj(level=6)
        rle_uncompressed_size = 0
        total_run_count = 0
        rle_enabled = True

        with open(raw_path, "wb") as raw_file, open(rle_path, "wb") as rle_file:
            for top in range(0, height, TILE_HEIGHT):
                bottom = min(height, top + TILE_HEIGHT)
                tile = source.crop((0, top, width, bottom)).convert("RGBA")
                try:
                    pixels = np.asarray(tile, dtype=np.uint8)
                    indexes = _map_tile_to_global_ids(pixels, connection)

                    raw_chunk = raw_compressor.compress(indexes.tobytes(order="C"))
                    if raw_chunk:
                        raw_file.write(raw_chunk)

                    # RLE almacena pares (cantidad consecutiva, ID de color).
                    # Se prueba en paralelo y solo se conserva si ocupa menos.
                    changes = np.flatnonzero(indexes[1:] != indexes[:-1]) + 1
                    starts = np.empty(changes.size + 1, dtype=np.int64)
                    starts[0] = 0
                    starts[1:] = changes
                    run_lengths = np.diff(
                        np.append(starts, len(indexes))
                    ).astype("<u4", copy=False)
                    total_run_count += len(run_lengths)

                    # Si un bloque tiene demasiadas corridas, RLE no es útil
                    # para esta imagen: se descarta su temporal y se continúa
                    # únicamente con la secuencia normal de identificadores.
                    if rle_enabled and len(run_lengths) >= len(indexes) * 0.5:
                        rle_enabled = False
                        rle_file.seek(0)
                        rle_file.truncate(0)
                        rle_uncompressed_size = 0
                        total_run_count = 0
                    elif rle_enabled:
                        run_ids = indexes[starts].astype("<u4", copy=False)
                        runs = np.empty((len(run_lengths), 2), dtype="<u4")
                        runs[:, 0] = run_lengths
                        runs[:, 1] = run_ids

                        rle_data = runs.tobytes(order="C")
                        rle_uncompressed_size += len(rle_data)
                        rle_chunk = rle_compressor.compress(rle_data)
                        if rle_chunk:
                            rle_file.write(rle_chunk)
                finally:
                    tile.close()

                if progress_callback:
                    progress_callback(
                        f"Procesando {source_path.name}: {bottom}/{height} filas"
                    )

            raw_tail = raw_compressor.flush()
            if raw_tail:
                raw_file.write(raw_tail)
            if rle_enabled:
                rle_tail = rle_compressor.flush()
                if rle_tail:
                    rle_file.write(rle_tail)

    connection.commit()

    # Si hay menos de una corrida por cada dos píxeles y RLE realmente reduce
    # el archivo, se guardan las corridas. En imágenes muy detalladas se usa
    # la secuencia de IDs sin RLE para evitar trabajo y metadatos innecesarios.
    use_rle = (
        rle_enabled
        and total_run_count < pixel_count * 0.5
        and rle_path.stat().st_size < raw_path.stat().st_size
    )

    selected_path = rle_path if use_rle else raw_path
    os.replace(selected_path, index_path)
    raw_path.unlink(missing_ok=True)
    rle_path.unlink(missing_ok=True)

    return {
        "arcname": arcname,
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "index_encoding": "rle_u32" if use_rle else "u32",
        "index_uncompressed_size": (
            rle_uncompressed_size if use_rle else pixel_count * 4
        ),
        "index_compressed_size": index_path.stat().st_size,
        "source_size": source_path.stat().st_size,
    }


def _write_compressed_palette(
    connection: sqlite3.Connection,
    palette_path: Path,
) -> tuple[int, int]:
    """Exporta el diccionario en orden de ID y lo comprime en un archivo temporal."""
    count = connection.execute("SELECT COUNT(*) FROM palette").fetchone()[0]
    if count > UINT32_MAX:
        raise ValueError(
            "El diccionario supera la cantidad máxima de colores admitida (2^32-1)."
        )

    compressor = zlib.compressobj(level=6)
    with open(palette_path, "wb") as output:
        cursor = connection.execute("SELECT rgba FROM palette ORDER BY id")
        for (rgba,) in cursor:
            chunk = compressor.compress(rgba)
            if chunk:
                output.write(chunk)
        tail = compressor.flush()
        if tail:
            output.write(tail)

    return count, palette_path.stat().st_size


def _copy_file(source: Path, destination) -> None:
    with open(source, "rb") as input_file:
        shutil.copyfileobj(input_file, destination, IO_CHUNK_SIZE)


# ---------------------------------------------------------------------------
# Generación del .imgmap V2
# ---------------------------------------------------------------------------


def build_from_folder(
    folder_path: Path,
    output_path: Path,
    title: str = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Path:
    folder_path = Path(folder_path).resolve()
    output_path = Path(output_path).resolve()

    if not folder_path.is_dir():
        raise NotADirectoryError(f"No existe la carpeta de origen: {folder_path}")

    root_cover = _find_cover(folder_path)
    if root_cover is None:
        raise FileNotFoundError(
            "Falta la portada en la carpeta raíz "
            "(portada.jpg, portada.jpeg, portada.png o portada.webp)"
        )

    branch_dirs = [
        path for path in folder_path.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not branch_dirs:
        raise FileNotFoundError(
            "No se encontraron carpetas numeradas (ramas) dentro de la carpeta raíz"
        )
    branch_dirs.sort(key=lambda path: int(path.name))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="vector_build_", dir=str(output_path.parent)
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        indexes_dir = temporary_root / "indexes"
        indexes_dir.mkdir()
        database_path = temporary_root / "palette.sqlite"
        palette_path = temporary_root / "palette.z"

        assets_to_process: list[tuple[Path, str]] = []
        used_names: set[str] = set()

        root_cover_arcname = _build_asset_arcname(
            root_cover, folder_path, used_names
        )
        assets_to_process.append((root_cover, root_cover_arcname))

        branches = []
        for branch_dir in branch_dirs:
            branch_number = branch_dir.name
            branch_cover = _find_cover(branch_dir)
            if branch_cover is None:
                raise FileNotFoundError(
                    f"Falta la portada en la rama {branch_number} "
                    "(portada.jpg, portada.jpeg, portada.png o portada.webp)"
                )

            branch_cover_arcname = _build_asset_arcname(
                branch_cover, folder_path, used_names
            )
            assets_to_process.append((branch_cover, branch_cover_arcname))

            branch_images = []
            for image_path in _list_branch_images(branch_dir, branch_cover):
                image_arcname = _build_asset_arcname(
                    image_path, folder_path, used_names
                )
                branch_images.append(image_arcname)
                assets_to_process.append((image_path, image_arcname))

            branches.append({
                "id": int(branch_number),
                "name": branch_number,
                "cover": branch_cover_arcname,
                "images": branch_images,
            })

        # La base se abre después de validar toda la estructura de carpetas,
        # así no queda bloqueada si falta una portada o hay un error previo.
        connection = _create_palette_database(database_path)
        asset_metadata = []
        try:
            total_assets = len(assets_to_process)
            for index, (source_path, arcname) in enumerate(assets_to_process):
                if progress_callback:
                    progress_callback(
                        f"Codificando imagen {index + 1} de {total_assets}: {source_path.name}"
                    )

                index_path = indexes_dir / f"{index:08d}.z"
                metadata = _process_source_image(
                    source_path,
                    arcname,
                    index_path,
                    connection,
                    progress_callback,
                )
                asset_metadata.append(metadata)

            palette_count, palette_compressed_size = _write_compressed_palette(
                connection, palette_path
            )

            manifest = {
                "format_version": FORMAT_VERSION,
                "title": title or folder_path.name,
                "cover": root_cover_arcname,
                "branches": branches,
            }

            # WebP ya está comprimido. Se comparan los tamaños finales estimados
            # de ambas representaciones, incluida su cabecera, para no elegir
            # una deduplicación que termine aumentando el .imgmap.
            all_sources_webp = all(
                source_path.suffix.lower() == ".webp"
                for source_path, _ in assets_to_process
            )
            source_total_size = sum(
                source_path.stat().st_size
                for source_path, _ in assets_to_process
            )
            dictionary_total_size = palette_compressed_size + sum(
                item["index_compressed_size"] for item in asset_metadata
            )

            stored_webp_assets = [
                {
                    "arcname": item["arcname"],
                    "width": item["width"],
                    "height": item["height"],
                    "webp_size": source_path.stat().st_size,
                }
                for item, (source_path, _) in zip(
                    asset_metadata, assets_to_process
                )
            ]
            pixels_header = {
                "format_version": FORMAT_VERSION,
                "storage_mode": "pixels",
                "manifest": manifest,
                "palette_count": palette_count,
                "palette_compressed_size": palette_compressed_size,
                "assets": asset_metadata,
            }
            webp_header = {
                "format_version": FORMAT_VERSION,
                "storage_mode": "webp",
                "manifest": manifest,
                "palette_count": 0,
                "palette_compressed_size": 0,
                "assets": stored_webp_assets,
            }

            thumbnail = _make_thumbnail(root_cover)
            fixed_size = len(thumbnail) + len(MAGIC) + HEADER_LENGTH.size
            pixels_header_bytes = json.dumps(
                pixels_header, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            webp_header_bytes = json.dumps(
                webp_header, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")

            pixels_estimated_size = (
                fixed_size + len(pixels_header_bytes) + dictionary_total_size
            )
            webp_estimated_size = (
                fixed_size + len(webp_header_bytes) + source_total_size
            )
            storage_mode = (
                "webp"
                if all_sources_webp and webp_estimated_size < pixels_estimated_size
                else "pixels"
            )

            if storage_mode == "webp":
                header = webp_header
                header_bytes = webp_header_bytes
            else:
                header = pixels_header
                header_bytes = pixels_header_bytes
            if len(header_bytes) > 64 * 1024 * 1024:
                raise ValueError("Los metadatos del mapa son demasiado grandes.")

            temporary_output = output_path.with_name(output_path.name + ".tmp")

            try:
                with open(temporary_output, "wb") as output:
                    # La miniatura JPEG sigue siendo el prefijo del archivo para
                    # conservar la vista previa del Explorador de Windows.
                    output.write(thumbnail)
                    output.write(MAGIC)
                    output.write(HEADER_LENGTH.pack(len(header_bytes)))
                    output.write(header_bytes)

                    if storage_mode == "pixels":
                        _copy_file(palette_path, output)
                        for index in range(total_assets):
                            _copy_file(indexes_dir / f"{index:08d}.z", output)
                    else:
                        # Conserva exactamente los archivos WebP de entrada.
                        for source_path, _ in assets_to_process:
                            _copy_file(source_path, output)

                    output.flush()
                    os.fsync(output.fileno())

                os.replace(temporary_output, output_path)
            finally:
                if temporary_output.exists():
                    temporary_output.unlink()

        finally:
            connection.close()

    if progress_callback:
        progress_callback(f"Archivo generado: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Lectura de cabecera y miniatura
# ---------------------------------------------------------------------------


def _read_header(file_path: Path) -> tuple[dict, int, bytes]:
    file_path = Path(file_path)
    with open(file_path, "rb") as source:
        # La miniatura JPEG está al principio. Se identifica su marcador EOI
        # para localizar la cabecera binaria que comienza inmediatamente después.
        jpeg_buffer = bytearray()
        thumbnail = None
        while len(jpeg_buffer) <= 64 * 1024 * 1024:
            chunk = source.read(64 * 1024)
            if not chunk:
                break
            jpeg_buffer.extend(chunk)
            end_marker = jpeg_buffer.find(b"\xff\xd9")
            if end_marker >= 0:
                thumbnail = bytes(jpeg_buffer[:end_marker + 2])
                unread = len(jpeg_buffer) - (end_marker + 2)
                if unread:
                    source.seek(-unread, os.SEEK_CUR)
                break

        if not thumbnail or not thumbnail.startswith(b"\xff\xd8"):
            raise ValueError("El archivo no contiene una miniatura JPEG válida.")
        if source.read(len(MAGIC)) != MAGIC:
            raise ValueError(
                "El archivo no tiene el formato Vector V2 o está dañado."
            )

        raw_length = source.read(HEADER_LENGTH.size)
        if len(raw_length) != HEADER_LENGTH.size:
            raise ValueError("Cabecera incompleta en el archivo .imgmap")

        header_length = HEADER_LENGTH.unpack(raw_length)[0]
        if header_length <= 0 or header_length > 64 * 1024 * 1024:
            raise ValueError("Tamaño de cabecera no válido.")

        header_bytes = source.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("Cabecera incompleta en el archivo .imgmap")
        header = json.loads(header_bytes.decode("utf-8"))
        data_offset = source.tell()

    if header.get("format_version") != FORMAT_VERSION:
        raise ValueError("Versión del formato .imgmap no admitida.")
    if not isinstance(header.get("manifest"), dict):
        raise ValueError("El archivo no contiene un manifest válido.")
    if not isinstance(header.get("assets"), list):
        raise ValueError("El archivo no contiene la lista de imágenes.")

    return header, data_offset, thumbnail


def read_manifest(map_path: Path) -> dict:
    """Lee solamente los metadatos; no descomprime las imágenes."""
    header, _, _ = _read_header(Path(map_path))
    return header["manifest"]


def read_thumbnail(map_path: Path) -> bytes:
    """Devuelve la miniatura JPEG guardada en la cabecera."""
    _, _, thumbnail = _read_header(Path(map_path))
    return thumbnail


# ---------------------------------------------------------------------------
# Preparación / primera capa de descompresión
# ---------------------------------------------------------------------------


def _decompress_exact(
    source,
    compressed_size: int,
    destination: Path,
    expected_size: int,
) -> None:
    """Descomprime un segmento zlib a disco y valida tamaño y checksum."""
    decompressor = zlib.decompressobj()
    remaining = compressed_size
    written = 0

    with open(destination, "wb") as output:
        while remaining:
            chunk = source.read(min(IO_CHUNK_SIZE, remaining))
            if not chunk:
                raise ValueError("El contenido comprimido del .imgmap está incompleto.")
            remaining -= len(chunk)
            pending = chunk

            # max_length limita la memoria aunque un bloque comprimido
            # represente muchos megabytes de píxeles repetidos.
            while pending:
                decoded = decompressor.decompress(pending, IO_CHUNK_SIZE)
                pending = decompressor.unconsumed_tail
                if decoded:
                    output.write(decoded)
                    written += len(decoded)
                    if written > expected_size:
                        raise ValueError(
                            "El contenido descomprimido supera el tamaño esperado."
                        )

        tail = decompressor.flush()
        if tail:
            output.write(tail)
            written += len(tail)

    if not decompressor.eof:
        raise ValueError("Flujo zlib incompleto o dañado.")
    if written != expected_size:
        raise ValueError(
            f"Tamaño descomprimido incorrecto: se esperaban {expected_size} bytes "
            f"y se obtuvieron {written}."
        )


def _decode_rle_indexes(rle_path: Path, indexes_path: Path, pixel_count: int) -> None:
    """Expande los pares RLE (longitud, ID) a un archivo de IDs uint32."""
    file_size = rle_path.stat().st_size
    if file_size == 0 or file_size % 8:
        raise ValueError("Flujo RLE no válido: tamaño incorrecto.")

    run_count = file_size // 8
    runs = np.memmap(rle_path, dtype="<u4", mode="r", shape=(run_count, 2))
    written_pixels = 0
    try:
        with open(indexes_path, "wb") as output:
            # Se procesa por lotes de corridas; una corrida larga se escribe
            # en fragmentos para mantener un uso de memoria acotado.
            for run_start in range(0, run_count, 65536):
                run_end = min(run_count, run_start + 65536)
                batch = np.asarray(runs[run_start:run_end], dtype=np.uint32)
                for run_length, color_id in batch:
                    length = int(run_length)
                    color = int(color_id)
                    if length <= 0 or written_pixels + length > pixel_count:
                        raise ValueError("Longitud RLE fuera del rango de la imagen.")

                    remaining = length
                    while remaining:
                        amount = min(remaining, 262144)
                        output.write(
                            np.full(amount, color, dtype="<u4").tobytes(order="C")
                        )
                        remaining -= amount
                    written_pixels += length
    finally:
        runs._mmap.close()

    if written_pixels != pixel_count:
        raise ValueError(
            f"El flujo RLE reconstruyó {written_pixels} píxeles; "
            f"se esperaban {pixel_count}."
        )


def _rebuild_webp_from_indexes(
    indexes_path: Path,
    palette_path: Path,
    output_path: Path,
    width: int,
    height: int,
    palette_count: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Reconstruye una imagen RGBA usando archivos mapeados en disco."""
    pixel_count = width * height
    expected_index_size = pixel_count * 4
    if indexes_path.stat().st_size != expected_index_size:
        raise ValueError("La cantidad de índices de una imagen no coincide con sus dimensiones.")

    palette = np.memmap(
        palette_path,
        dtype=np.uint8,
        mode="r",
        shape=(palette_count, 4),
    )
    indexes = np.memmap(
        indexes_path,
        dtype="<u4",
        mode="r",
        shape=(pixel_count,),
    )

    rgba_path = output_path.with_suffix(output_path.suffix + ".rgba.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(rgba_path, "wb") as rgba_output:
            tile_pixels = max(width, 1) * TILE_HEIGHT
            for offset in range(0, pixel_count, tile_pixels):
                end = min(pixel_count, offset + tile_pixels)
                block_indexes = np.asarray(indexes[offset:end], dtype=np.uint32)
                if block_indexes.size and int(block_indexes.max()) >= palette_count:
                    raise ValueError("El archivo contiene un índice fuera del diccionario de colores.")
                block_rgba = palette[block_indexes]
                rgba_output.write(block_rgba.tobytes(order="C"))
                if progress_callback:
                    progress_callback(
                        f"Reconstruyendo {output_path.name}: {end}/{pixel_count} píxeles"
                    )

        # Image.frombuffer puede leer desde mmap sin crear una segunda copia
        # completa del bitmap en Python. La codificación WebP requiere Pillow.
        import mmap

        with open(rgba_path, "r+b") as rgba_file:
            with mmap.mmap(rgba_file.fileno(), 0, access=mmap.ACCESS_READ) as rgba_map:
                image = Image.frombuffer(
                    "RGBA",
                    (width, height),
                    rgba_map,
                    "raw",
                    "RGBA",
                    0,
                    1,
                )
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(
                        output_path,
                        format="WEBP",
                        lossless=True,
                        quality=100,
                        method=6,
                        exact=True,
                    )
                finally:
                    image.close()
    finally:
        try:
            indexes._mmap.close()
            palette._mmap.close()
        except Exception:
            pass
        if rgba_path.exists():
            rgba_path.unlink()


def prepare_map(
    map_path: Path,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> "PreparedMap":
    """Descomprime y reconstruye todas las imágenes antes de devolver el mapa listo."""
    map_path = Path(map_path).resolve()
    header, data_offset, _ = _read_header(map_path)

    # Preferimos extraer junto al .imgmap para utilizar el espacio libre de
    # su misma unidad. Si la carpeta no permite escribir, usamos TEMP del SO.
    try:
        temporary_directory = Path(
            tempfile.mkdtemp(prefix="vector_open_", dir=str(map_path.parent))
        )
    except OSError:
        temporary_directory = Path(tempfile.mkdtemp(prefix="vector_open_"))
    assets_directory = temporary_directory / "images"
    assets_directory.mkdir()
    palette_path = temporary_directory / "palette.rgba"

    try:
        assets = header["assets"]
        if not assets:
            raise ValueError("El mapa no contiene imágenes.")

        storage_mode = header.get("storage_mode", "pixels")
        with open(map_path, "rb") as source:
            source.seek(data_offset)
            total_assets = len(assets)

            if storage_mode == "webp":
                # En este modo el diccionario no habría reducido el tamaño.
                # Se extraen los WebP originales sin alterar sus bytes.
                for index, asset in enumerate(assets):
                    arcname = str(asset["arcname"])
                    webp_size = int(asset["webp_size"])
                    if webp_size <= 0:
                        raise ValueError(f"Tamaño WebP no válido en {arcname}.")
                    output_path = _safe_asset_path(assets_directory, arcname)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    remaining = webp_size
                    with open(output_path, "wb") as image_output:
                        while remaining:
                            chunk = source.read(min(IO_CHUNK_SIZE, remaining))
                            if not chunk:
                                raise ValueError(f"Imagen WebP incompleta: {arcname}")
                            image_output.write(chunk)
                            remaining -= len(chunk)
                    if progress_callback:
                        progress_callback(
                            f"Imagen {index + 1} de {total_assets} extraída: {arcname}"
                        )

            elif storage_mode == "pixels":
                palette_count = int(header["palette_count"])
                palette_compressed_size = int(header["palette_compressed_size"])
                if palette_count <= 0 or palette_count > UINT32_MAX:
                    raise ValueError("Cantidad de colores no válida en el diccionario.")

                _decompress_exact(
                    source,
                    palette_compressed_size,
                    palette_path,
                    palette_count * 4,
                )

                for index, asset in enumerate(assets):
                    arcname = str(asset["arcname"])
                    width = int(asset["width"])
                    height = int(asset["height"])
                    compressed_size = int(asset["index_compressed_size"])
                    uncompressed_size = int(asset["index_uncompressed_size"])
                    pixel_count = width * height
                    if width <= 0 or height <= 0:
                        raise ValueError(f"Dimensiones no válidas en {arcname}.")
                    if pixel_count != int(asset["pixel_count"]):
                        raise ValueError(f"Metadatos de píxeles no válidos en {arcname}.")
                    if compressed_size <= 0 or uncompressed_size <= 0:
                        raise ValueError(f"Tamaño comprimido no válido en {arcname}.")

                    encoded_indexes_path = temporary_directory / f"indexes_{index:08d}.encoded"
                    indexes_path = temporary_directory / f"indexes_{index:08d}.raw"
                    _decompress_exact(
                        source,
                        compressed_size,
                        encoded_indexes_path,
                        uncompressed_size,
                    )

                    encoding = asset.get("index_encoding", "u32")
                    if encoding == "rle_u32":
                        _decode_rle_indexes(
                            encoded_indexes_path, indexes_path, pixel_count
                        )
                        encoded_indexes_path.unlink(missing_ok=True)
                    elif encoding == "u32":
                        if uncompressed_size != pixel_count * 4:
                            raise ValueError(f"Tamaño de índices incorrecto en {arcname}.")
                        os.replace(encoded_indexes_path, indexes_path)
                    else:
                        raise ValueError(f"Codificación de índices desconocida en {arcname}.")

                    output_path = _safe_asset_path(assets_directory, arcname)
                    _rebuild_webp_from_indexes(
                        indexes_path,
                        palette_path,
                        output_path,
                        width,
                        height,
                        palette_count,
                        progress_callback,
                    )
                    indexes_path.unlink(missing_ok=True)

                    if progress_callback:
                        progress_callback(
                            f"Imagen {index + 1} de {total_assets} reconstruida: {arcname}"
                        )
            else:
                raise ValueError(f"Modo de almacenamiento desconocido: {storage_mode}")

        return PreparedMap(
            source_path=map_path,
            manifest=header["manifest"],
            temporary_directory=temporary_directory,
            assets_directory=assets_directory,
        )

    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    finally:
        if palette_path.exists():
            palette_path.unlink()


class PreparedMap:
    """Mapa ya reconstruido en disco; sus imágenes permanecen en WEBP lossless."""

    def __init__(
        self,
        source_path: Path,
        manifest: dict,
        temporary_directory: Path,
        assets_directory: Path,
    ):
        self.source_path = Path(source_path)
        self.manifest = manifest
        self.temporary_directory = Path(temporary_directory)
        self.assets_directory = Path(assets_directory)
        self._closed = False

    def entry_path(self, arcname: str) -> Path:
        if self._closed:
            raise RuntimeError("El mapa temporal ya fue cerrado.")
        path = _safe_asset_path(self.assets_directory, arcname)
        if not path.is_file():
            raise FileNotFoundError(f"No existe la imagen reconstruida: {arcname}")
        return path

    def read_entry(self, arcname: str) -> bytes:
        return self.entry_path(arcname).read_bytes()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutil.rmtree(self.temporary_directory, ignore_errors=True)


# ---------------------------------------------------------------------------
# API pequeña para lectores antiguos y utilidades
# ---------------------------------------------------------------------------

_PREPARED_CACHE: dict[tuple[str, int], PreparedMap] = {}


def read_entry(map_path: Path, arcname: str) -> bytes:
    """Lee una imagen reconstruida. Mantiene una caché temporal para llamadas repetidas."""
    map_path = Path(map_path).resolve()
    key = (str(map_path), map_path.stat().st_mtime_ns)
    prepared = _PREPARED_CACHE.get(key)

    if prepared is None:
        # Si el mismo archivo fue reemplazado, libera su extracción anterior.
        stale_keys = [item for item in _PREPARED_CACHE if item[0] == str(map_path)]
        for stale_key in stale_keys:
            _PREPARED_CACHE.pop(stale_key).close()
        prepared = prepare_map(map_path)
        _PREPARED_CACHE[key] = prepared

    return prepared.read_entry(arcname)


def total_image_count(manifest: dict) -> int:
    return sum(len(branch["images"]) for branch in manifest["branches"])


def _cleanup_prepared_cache() -> None:
    for prepared in list(_PREPARED_CACHE.values()):
        prepared.close()
    _PREPARED_CACHE.clear()


import atexit
atexit.register(_cleanup_prepared_cache)
