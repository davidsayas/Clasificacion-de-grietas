"""Corre todos los tests sin necesitar pytest:   python tests/run_all.py

Cada archivo tests/test_*.py tiene funciones test_*(). Si pytest está instalado,
también funciona con:   python -m pytest tests
"""
import importlib
import sys
import tempfile
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "tests"))


def main(filtro=None):
    ok, fallos = 0, []
    for archivo in sorted((RAIZ / "tests").glob("test_*.py")):
        if filtro and filtro not in archivo.stem:
            continue
        modulo = importlib.import_module(archivo.stem)
        for nombre in sorted(n for n in dir(modulo) if n.startswith("test_")):
            funcion = getattr(modulo, nombre)
            try:
                if "tmp" in funcion.__code__.co_varnames[:funcion.__code__.co_argcount]:
                    with tempfile.TemporaryDirectory() as d:
                        funcion(Path(d))
                else:
                    funcion()
                ok += 1
                print(f"  ok     {archivo.stem}.{nombre}")
            except Exception:
                fallos.append(f"{archivo.stem}.{nombre}")
                print(f"  FALLA  {archivo.stem}.{nombre}")
                traceback.print_exc()
    print(f"\n{ok} pruebas correctas, {len(fallos)} con fallas")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
