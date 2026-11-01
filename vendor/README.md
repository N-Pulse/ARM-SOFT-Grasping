# `vendor/` — bundled OpenGL libraries

Open3D links against `libEGL.so.1` / `libGL.so.1` even when no window is ever
opened, so importing it fails on a bare Linux machine that has no OpenGL
runtime installed.

If you have root, the clean fix is the system package:

```bash
sudo apt install libegl1 libgl1 libglib2.0-0
```

If you do not, these unpacked Ubuntu 22.04 (amd64) libraries stand in. The
`.venv` in this folder adds them to `LD_LIBRARY_PATH` automatically when you
activate it, so normally there is nothing to do.

To use them without the venv:

```bash
export LD_LIBRARY_PATH="$PWD/vendor/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```

On macOS, Windows, or a non-amd64 Linux these files are ignored — delete the
folder if it is in your way, and install OpenGL the usual way for your platform.
