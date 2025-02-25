## cmd
```
python convert.py --mode <モード> --input <入力パス> [--output <出力パス>]
```

## eg
```
// i2b
python convert.py --mode i2b --input ./images --output ./output/converted_data.idx3-ubyte

// b2i
python convert.py --mode b2i --input ./output/converted_data.idx3-ubyte --output ./output/images

```
