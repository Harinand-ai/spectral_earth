import uvicorn
import pyrootutils

pyrootutils.setup_root(r"d:\Projects\spectral_earth", indicator=".project-root", pythonpath=True)

if __name__ == "__main__":
    print("=" * 80)
    print("LAUNCHING SPECTRALEARTH ACTIVE LEARNING VISUALIZATION APPLICATION")
    print("Open http://127.0.0.1:8000 in your browser")
    print("=" * 80)
    uvicorn.run("src.visualization_app.app:app", host="127.0.0.1", port=8000, reload=False)
