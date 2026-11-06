"""Funciones de biblioteca para archivos .imgmap V2."""

from pathlib import Path

from mapformat import read_manifest, read_thumbnail, EXTENSION


def list_imgmaps(folder):
    """Devuelve los .imgmap de primer nivel y sus metadatos de portada."""
    folder = Path(folder)
    if not folder.is_dir():
        return []

    files = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == EXTENSION
    ]
    files.sort(key=lambda path: path.name.lower())

    result = []
    for path in files:
        try:
            manifest = read_manifest(path)
            cover = manifest.get("cover")
            result.append({
                "name": path.name,
                "path": str(path.resolve()),
                "cover": cover,
                "coverUrl": (
                    "/library-cover/"
                    + str(path.resolve()).replace("\\", "/")
                    + "|"
                    + str(cover)
                ) if cover else None,
            })
        except Exception as error:
            # Un archivo dañado no impide mostrar los demás elementos.
            result.append({
                "name": path.name,
                "path": str(path.resolve()),
                "cover": None,
                "error": str(error),
            })

    return result


def read_cover(path, cover_arcname):
    """Devuelve la miniatura JPEG de la cabecera para la tarjeta de biblioteca."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if not cover_arcname:
        raise FileNotFoundError("El manifest no contiene portada.")

    manifest = read_manifest(path)
    if manifest.get("cover") != cover_arcname:
        raise ValueError("La portada solicitada no coincide con el manifest.")

    return read_thumbnail(path)


def get_library_files(folder):
    return list_imgmaps(folder)
