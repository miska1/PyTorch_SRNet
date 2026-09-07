# Dataset location

Place the already-separated grayscale image pairs in:

```text
train/cover/  train/stego/
val/cover/    val/stego/
test/cover/   test/stego/
```

Cover and stego images are paired by their relative path without the extension.
All images must be 256x256 unless `--image-size` is intentionally changed.
